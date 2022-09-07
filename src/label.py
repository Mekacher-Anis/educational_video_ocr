from pathlib import Path
import timeit
import mmcv.video
from det_rec import MMOCR, textdet_models, textrecog_models
import mmcv
from tqdm import tqdm
from export.str_writer import SubtitleWriter
from itertools import product

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
    def __init__(self, batch_mode=False, batch_size = 10, rec_model = 'ABINet_Vision', det_model = 'PANet_IC15'):
        self.ocr = MMOCR(recog=rec_model, det=det_model)
        self.batch_mode = batch_mode
        self.batch_size = batch_size

    def __call__(self, img):
        try:
            return self.ocr.readtext(
                img,
                batch_mode=self.batch_mode,
                recog_batch_size=self.batch_size,
                det_batch_size=self.batch_size,
                single_batch_size=self.batch_size,
                merge=True
            )
        except:
            return []

def main():
    # configuration
    video_path = 'samples/normal_text_sample.mp4'
    rec_model = 'ABINet_Vision'
    batch_size = 1000
    det_model = 'PANet_IC15'
    absolute_video_path = Path.cwd()/video_path
    absolute_subtitle_path = absolute_video_path.parent / f'{absolute_video_path.stem}__{det_model}__{rec_model}.srt'
    
    # read video
    reader = mmcv.video.VideoReader(video_path)
    print(f'Processing {video_path} : \nFrames:{reader.frame_cnt} \nFPS: {reader.fps} \nWidth: {reader.width}px \nHeight: {reader.height}px')
    
    # define OCR pipeline
    ocr = OCRPipeline(batch_mode=(batch_size!=1), batch_size=batch_size, det_model=det_model, rec_model=rec_model)
    
    # define the frame batches and run them through the pipeline
    arr_chunks = [(i, i + batch_size)
                for i in range(0, reader.frame_cnt, batch_size)]

    with open(absolute_subtitle_path, mode='w+') as f:
        sub = SubtitleWriter(f, reader.fps)
        for start, end in arr_chunks:
            result = ocr(reader[start:end][0])
            for i, res in enumerate(result):
                sub.addSubtitle(res['text'], start+i)
        sub.finish()



if __name__ == '__main__':
    main()