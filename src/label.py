from pathlib import Path
import timeit
import mmcv.video
import mmcv
from tqdm import tqdm
from export.str_writer import SubtitleWriter
from itertools import product
from mmocr.ocr import MMOCR
import os

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


class OCRPipeline:
    def __init__(self, rec_model = 'SATRN', det_model = 'DBPP_r50', **kwargs):
        self.configs_dir = os.environ['MMOCR_HOME']+'/configs' if 'MMOCR_HOME' in os.environ else './configs'
        self.ocr = MMOCR(recog=rec_model, det=det_model, config_dir=self.configs_dir, **kwargs)

    def __call__(self, img):
        try:
            return self.ocr.readtext(img)
        except:
            return []

def main():
    # configuration
    video_path = 'samples/normal_text_sample.mp4'
    batch_size = 4 # how many elements to process at a time
    FPS = 2 # number of frames per second of video to process
    det_model = 'DBPP_r50' # name of detection model to use
    rec_model = 'SATRN' # name of reognition model to use
    absolute_video_path = Path.cwd()/video_path
    absolute_subtitle_path = absolute_video_path.parent / f'{absolute_video_path.stem}__{det_model}__{rec_model}.srt'
    
    # read video
    reader = mmcv.video.VideoReader(video_path)
    print(f'Processing {video_path} : \nFrames:{reader.frame_cnt} \nFPS: {reader.fps} \nWidth: {reader.width}px \nHeight: {reader.height}px')
    
    # define OCR pipeline
    ocr = OCRPipeline(
        det_model=det_model,
        det_ckpt='/home/anis/Documents/AI/ML/OCR/educational_video_ocr/dbnetpp_resnet50-dcnv2_fpnc_epoch_20.pth',
        rec_model=rec_model,
        recog_ckpt='/home/anis/Documents/AI/ML/OCR/educational_video_ocr/SATRN_lvdb_epoch_2.pth',
    )
    
    # define the frame batches and run them through the pipeline
    skip_frames = int(reader.fps // FPS)
    frame_indexes = range(0, reader.frame_cnt, skip_frames)
    arr_chunks = [(i, i + batch_size)
                for i in range(0, len(frame_indexes), batch_size)]
    
    print(f'Using text det model: {det_model}\nUsing text rec model: {rec_model}\nProcessing {FPS} frames per one second of video\nBatch Size: {batch_size}')

    with open(absolute_subtitle_path, mode='w+') as f:
        sub = SubtitleWriter(f, FPS)
        for start, end in tqdm(arr_chunks):
            batch = [reader[i] for i in frame_indexes[start:end]]
            result = ocr(batch)
            for i, res in enumerate(result):
                sub.addSubtitle(res['rec_texts'], start+i)
        sub.finish()



if __name__ == '__main__':
    main()