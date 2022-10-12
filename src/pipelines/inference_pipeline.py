from . import models
from typing import Dict
from mmocr.ocr import MMOCR
from mmocr.utils.bbox_utils import stitch_boxes_into_lines
import os
import random
import numpy as np
import cv2

class OCRPipeline:
    def __init__(self, rec_model = 'SATRN', det_model = 'DBPP_r50', **kwargs):
        self.configs_dir = os.environ['MMOCR_HOME']+'/configs' if 'MMOCR_HOME' in os.environ else './configs'
        det_config = self.__get_model_config(det_model)
        recog_config = self.__get_model_config(rec_model)
        self.ocr = MMOCR(
            det=det_model,
            det_ckpt=det_config['ckpt'],
            recog=rec_model,
            recog_ckpt=recog_config['ckpt'],
            config_dir=self.configs_dir,
            **kwargs)

    def __call__(self, img, **kwargs):
        try:
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
            return [stitch_boxes_into_lines(result[i], max_x_dist=20) for i in range(len(result))]
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