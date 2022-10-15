from json import detect_encoding
from pprint import pprint
from . import models
from post_processing.misspelling_detection import MisspellingDetection
from post_processing.dict_correct import SymSpellCorrect
from typing import Dict
from mmocr.ocr import MMOCR
from mmocr.utils.bbox_utils import stitch_boxes_into_lines
import os
import random
import numpy as np
import cv2

class OCRPipeline:
    def __init__(
        self,
        rec_model = 'SATRN',
        det_model = 'DBPP_r50',
        merge_into_lines = True,
        run_err_correction = True,
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
                            if candidates:
                                corrected_count += 1
                                box['text'] = candidates[0].term
                                
            self.metrics['detection_count'] += detection_count
            self.metrics['wrong_count'] += wrong_count
            self.metrics['corrected_count'] += corrected_count
            
            # join words into lines if requested
            if self.merge_into_lines:
                return [stitch_boxes_into_lines(result[i], max_x_dist=20) for i in range(len(result))]
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

    def visualize_joined(self, img, name, res):
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
            cv2.putText(out, t['text'], (pts[0][0]+10,pts[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow(name, out)
        if cv2.waitKey(0)  == ord('q'):
            exit(0)
        cv2.destroyAllWindows()
        
    def visualize(self, inputs, preds, **kwargs):
        self.ocr.inferencer.visualize(inputs, preds,**kwargs)