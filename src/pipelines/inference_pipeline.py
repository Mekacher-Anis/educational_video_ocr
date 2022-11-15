import os
import queue
import random
import time
from typing import Dict, List

import cv2
import numpy as np
from mmocr.ocr import MMOCR
from mmocr.utils.bbox_utils import is_on_same_line
from torch import LongTensor, multiprocessing
from torch.nn.functional import softmax
from transformers import (BertForNextSentencePrediction, BertTokenizer)

from post_processing.dict_correct import SymSpellCorrect
from post_processing.misspelling_detection import MisspellingDetection
from . import models


try:
    multiprocessing.set_start_method('spawn')
except:
    pass

class OCRPipeline:
    def __init__(
        self,
        rec_model = 'PARSeq',
        det_model = 'DBPP_r50',
        merge_into_lines = True,
        run_err_correction = True,
        merge_max_x_dist = 50,
        merge_min_x_overlap = .5,
        merge_max_y_dist = 10,
        merge_min_y_overlap = .5,
        merge_min_confidence = .9,
        use_bert_for_merging=False,
        merge_pool_size = 4,
        bert_pool_size = 2,
        lang:str='en',
        **kwargs
    ):
        self.configs_dir = os.environ['MMOCR_HOME']+'/configs' if 'MMOCR_HOME' in os.environ else './configs'
        det_config = self.__get_model_config(det_model)
        recog_config = self.__get_model_config(rec_model)
        self.ocr = MMOCR(
            det=det_model,
            det_ckpt=det_config['ckpt'],
            recog=rec_model,
            recog_ckpt=recog_config['ckpt'],
            config_dir=self.configs_dir,
            log_level='CRITICAL',
            **kwargs)
        
        self.merge_into_lines = merge_into_lines
        self.use_bert_for_merging = use_bert_for_merging
        self.run_err_correction = run_err_correction
        self.lang = lang
        if self.run_err_correction:
            self.error_detection = MisspellingDetection(lang=lang)
            self.error_correction = SymSpellCorrect(lang=lang)
            self.metrics = {
                'detection_count': 0,
                'corrected_count': 0,
                'wrong_count': 0
            }

        if self.merge_into_lines:
            self.merge_pool_size = merge_pool_size
            self.bert_pool_size = bert_pool_size
            self.line_merging_pool = multiprocessing.Pool(self.merge_pool_size)
        if self.use_bert_for_merging:
            self.bert_pool = multiprocessing.Pool(self.bert_pool_size)
            self.bert_pool_manager = multiprocessing.Manager()
            self.bert_input_queues = [self.bert_pool_manager.Queue() for _ in range(self.bert_pool_size)]
            self.bert_output_queues = [self.bert_pool_manager.Queue() for _ in range(self.bert_pool_size)]
            self.bert_processes_locks = [self.bert_pool_manager.Lock() for _ in range(self.bert_pool_size)]
            for i in range(self.bert_pool_size):
                self.bert_pool.apply_async(
                    OCRPipeline.is_next_sentence,
                    [self.bert_input_queues[i], self.bert_output_queues[i]],
                )
                        
        self.merge_max_x_dist = merge_max_x_dist
        self.merge_min_x_overlap = merge_min_x_overlap
        self.merge_max_y_dist = merge_max_y_dist
        self.merge_min_y_overlap = merge_min_y_overlap
        self.merge_min_confidence = merge_min_confidence
        
        
    def __call__(self, img, **kwargs):
        try:
            # run detection and recognition
            result = self.ocr.readtext(img, **kwargs)
            result = [[
                {
                    'box': x[0],
                    'box_score': x[1],
                    'text_score': x[2],
                    'text': x[3],
                }
                for x in
                zip(result[i]['det_polygons'],result[i]['det_scores'],result[i]['rec_scores'], result[i]['rec_texts'])
            ] for i in range(len(result))]
                        
            # run error detection and correction
            detection_count = 0
            wrong_count = 0
            corrected_count = 0
            if self.run_err_correction:
                for img in result:
                    for box in img:
                        detection_count += 1
                        # print(box['text'], self.error_detection.check(box['text']))
                        if not self.error_detection.check(box['text']):
                            wrong_count += 1
                            candidates = self.error_correction.get_candidates(box['text'])
                            if candidates and candidates[0].term != box['text']:
                                corrected_count += 1
                                box['text'] = candidates[0].term
                                
                self.metrics['detection_count'] += detection_count
                self.metrics['wrong_count'] += wrong_count
                self.metrics['corrected_count'] += corrected_count
                
            # avg = sum([len(x) for x in result])
            # avg /= len(result)
            # print(f'Average words per frame : {avg}')
            # join words into lines if requested
            if self.merge_into_lines:
                return self.line_merging_pool.starmap(
                    OCRPipeline.merge_lines,
                    [(
                        x,
                        self.use_bert_for_merging,
                        self.bert_input_queues if self.use_bert_for_merging else None,
                        self.bert_output_queues if self.use_bert_for_merging else None,
                        self.bert_processes_locks if self.use_bert_for_merging else None
                    ) for x in result],
                    chunksize=1
                )
            else:
                return result
            
        except Exception as e:
            print(e)
            return []
    
    def __get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        if model_name not in models.model_dict:
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return models.model_dict[model_name]
        
    def print_metrics(self):
        if self.run_err_correction:
            detection_count = self.metrics['detection_count']
            wrong_count = self.metrics['wrong_count']
            corrected_count = self.metrics['corrected_count']
            if not detection_count:
                print('[INFO] Inference pipeline stats: No text has been detected.')
                return
            else:
                print(f'[INFO] Inference pipeline Stats:\n\
                        \tDetected word: {int(detection_count)}\n\
                        \tWords not in dict: {wrong_count} ({round((wrong_count/detection_count)*100, 2)}%)\n\
                        \tWords corrected: {corrected_count} ({round((corrected_count/detection_count)*100, 2)}%)\n'
                    )
    
    def is_next_sentence(
        input_queue: multiprocessing.Queue,
        bert_output_queue: multiprocessing.Queue,
        conf_thresh=.8
    ):
        # we avoid creating bert models repeadtly upon calling this function 
        # by creating multiple processes that wait for input in the input queue
        # and process data when it's available
        print(f"[INFO] Starting BERT processor {multiprocessing.current_process().ident}")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        print(f'[INFO] BERT processor {multiprocessing.current_process().ident} Ready.')
        
        
        while True:
            try:
                task = input_queue.get(block=True) # (first_sentence, second_sentence, output_queue)
            except queue.Empty:
                time.sleep(.05 + (random.random() * .1)) # avoid multiple processes waking up at the same time
            else:
                encoding = tokenizer(task[0], task[1], return_tensors="pt")
                outputs = model(**encoding, labels=LongTensor([1]))
                prob = softmax(outputs.logits, dim=1)[0,0].item()
                bert_output_queue.put(prob > conf_thresh)

    def visualize_joined(img, name, res):
        shapes = np.zeros_like(img, np.uint8)
        out = img.copy()
        for t in res:
            pts = np.array(t['box'], dtype=np.int32).reshape((-1,2))
            cv2.fillPoly(shapes,[pts],color=(random.randint(80, 255) ,random.randint(80, 255),random.randint(80, 255)))
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
        for t in res:
            pts = np.array(t['box'], dtype=np.int32).reshape((-1,2))
            # cv2.putText(out, t['text'], (pts[0][0]+10,pts[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow(name, out)
        while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(1000) == ord('f'):
                break
        cv2.destroyAllWindows()
        
    def visualize(self, inputs, preds, **kwargs):
        self.ocr.inferencer.visualize(inputs, preds,**kwargs)
        
    def merge_lines(
        preds,
        use_bert_for_merging,
        bert_input_queues: List[multiprocessing.Queue],
        bert_output_queues: List[multiprocessing.Queue],
        bert_processes_locks: List[multiprocessing.Lock],
        merge_max_x_dist = 100,
        merge_min_x_overlap = .5,
        merge_max_y_dist = 10,
        merge_min_y_overlap = .5,
    ):
        # merge into lines
        stitched = OCRPipeline.stitch_boxes_into_lines(
            preds,
            use_bert_for_merging,
            bert_input_queues,
            bert_output_queues,
            bert_processes_locks,
            max_dist=merge_max_x_dist,
            min_overlap_ratio=merge_min_y_overlap,
        )
        
        # flip x,y and stitch based on y (merge longer lines/paragraphs)
        for s in stitched:
            x_max = max(s['box'][::2])
            x_min = min(s['box'][::2])
            y_max = max(s['box'][1::2])
            y_min = min(s['box'][1::2])
            s['box'] = [y_min, x_min, y_max, x_min, y_max, x_max, y_min, x_max]
        stitched = OCRPipeline.stitch_boxes_into_lines(
            stitched,
            use_bert_for_merging,
            bert_input_queues,
            bert_output_queues,
            bert_processes_locks,
            max_dist=merge_max_y_dist,
            min_overlap_ratio=merge_min_x_overlap,
        )
        
        # flip back
        for s in stitched:
            x_max = max(s['box'][1::2])
            x_min = min(s['box'][1::2])
            y_max = max(s['box'][::2])
            y_min = min(s['box'][::2])
            s['box'] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            
        return stitched
        
    def stitch_boxes_into_lines(
        boxes,
        use_bert_for_merging,
        bert_input_queues: List[multiprocessing.Queue],
        bert_output_queues: List[multiprocessing.Queue],
        bert_processes_locks: List[multiprocessing.Lock],
        max_dist=10,
        min_overlap_ratio=0.8,
    ):
        """Stitch fragmented boxes of words into lines.

        Note: part of its logic is inspired by @Johndirr
        (https://github.com/faustomorales/keras-ocr/issues/22)

        Args:
            boxes (list): List of ocr results to be stitched
            max_x_dist (int): The maximum horizontal distance between the closest
                        edges of neighboring boxes in the same line
            min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                        allowed for any pairs of neighboring boxes in the same line

        Returns:
            merged_boxes(list[dict]): List of merged boxes and texts
        """
        if len(boxes) <= 1:
            return boxes

        merged_boxes = []
    
        num_bert_processor = len(bert_input_queues) if use_bert_for_merging else 0
        
        
        # sort groups based on the x_min coordinate of boxes
        x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
        # store indexes of boxes which are already parts of other lines
        skip_idxs = set()

        i = 0
        # locate lines of boxes starting from the leftmost one
        for i in range(len(x_sorted_boxes)):
            if use_bert_for_merging:
                bert_processs_id = random.randint(0, num_bert_processor - 1)
                # print(f'Merge process {multiprocessing.current_process().ident} waiting for lock on Bert process num {bert_processs_id}')
                bert_processes_locks[bert_processs_id].acquire()
                # print(f'Merge process {multiprocessing.current_process().ident} aquired lock on Bert process num {bert_processs_id}')
            try:
                if i in skip_idxs:
                    continue
                # the rightmost box in the current line
                rightmost_box_idx = i
                line = [rightmost_box_idx]
                for j in range(i + 1, len(x_sorted_boxes)):
                    if j in skip_idxs:
                        continue
                    if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                                    x_sorted_boxes[j]['box'], min_overlap_ratio):
                        line.append(j)
                        skip_idxs.add(j)
                        rightmost_box_idx = j
                
                # split line into lines if the distance between two neighboring
                # sub-lines' is greater than max_x_dist
                lines = []
                line_idx = 0
                lines.append([line[0]])
                for k in range(1, len(line)):
                    curr_box = x_sorted_boxes[line[k]]
                    prev_box = x_sorted_boxes[line[k - 1]]
                    dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
                    current_line = ' '.join([x_sorted_boxes[i]['text'] for i in lines[line_idx]])
                    current_word = x_sorted_boxes[line[k]]['text']
                    if dist > max_dist:
                        line_idx += 1
                        lines.append([])
                    elif use_bert_for_merging:
                        # print(f'Process {process_id} Posting job')
                        bert_input_queues[bert_processs_id].put((current_line, current_word))
                        res = bert_output_queues[bert_processs_id].get(block=True)
                        # print(f'Process {process_id} got response : ', res)
                        if not res:
                            line_idx += 1
                            lines.append([])
                    lines[line_idx].append(line[k])

                # Get merged boxes
                for box_group in lines:
                    merged_box = {}
                    merged_box['text'] = ' '.join(
                        [x_sorted_boxes[idx]['text'] for idx in box_group])
                    x_min, y_min = float('inf'), float('inf')
                    x_max, y_max = float('-inf'), float('-inf')
                    for idx in box_group:
                        x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                        x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                        y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                        y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
                    merged_box['box'] = [
                        x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
                    ]
                    merged_boxes.append(merged_box)
            finally:
                if use_bert_for_merging:
                    bert_processes_locks[bert_processs_id].release()
            # print(f'Merges process {multiprocessing.current_process().ident} released lock on Bert process num {bert_processs_id}')

        return merged_boxes