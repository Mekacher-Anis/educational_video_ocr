from pathlib import Path
import mmcv.video
import mmcv
from tqdm import tqdm
from export.str_writer import SubtitleWriter
from dotenv import load_dotenv
import os
import argparse
from itertools import product
from pipelines.inference_pipeline import OCRPipeline
from pipelines.test_pipeline import TestPipeline
import logging
from pprint import pprint

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument(
        'mode',
        help='What mode to run the script in. Can be either "infer" or "test"',
        choices=['test', 'infer'],
        default='infer'
    )
    parser.add_argument(
        'det_model_name',
        default='DBPP_r50',
        help='Detection model names seperated by comma.',
    )
    parser.add_argument(
        'rec_model_name',
        default='MASTER',
        help='Recognition model name seperated by comma.',
    )
    parser.add_argument(
        'video_path',
        nargs='*',
        default='samples/normal_text_sample.mp4',
        help='Relative video paths. One or multiple paths maybe specified.',
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help='Batch size of processd images. (Decrease this if you get out of memory errors)',
    )
    parser.add_argument(
        '--inference-fps',
        default=2,
        type=float,
        help='Number of frames per one second of video to be processed.',
    )
    parser.add_argument(
        '--error-correction',
        action='store_true',
        default=False,
        help='Run error correction after text recognition.'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def inference(args):
    # configuration
    video_path = args.video_path
    batch_size = args.batch_size # how many elements to process at a time
    inference_fps = args.inference_fps # number of frames per second of video to process
    det_model = args.det_model_name # name of detection model to use
    rec_model = args.rec_model_name # name of reognition model to use
    absolute_video_path = Path.cwd()/video_path
    absolute_subtitle_path = absolute_video_path.parent / f'{absolute_video_path.stem}__{det_model}__{rec_model}.srt'
    visualization_out_path = absolute_video_path.parent / f'{absolute_video_path.stem}_images'
    not visualization_out_path.exists() and os.mkdir(visualization_out_path)
    
    # read video
    reader = mmcv.video.VideoReader(video_path)
    print(f'Processing {video_path} : \nFrames:{reader.frame_cnt} \nFPS: {reader.fps} \nWidth: {reader.width}px \nHeight: {reader.height}px\n\n\n')
    
    # define OCR pipeline
    pipeline = OCRPipeline(
        det_model=det_model,
        rec_model=rec_model,
        run_err_correction=args.error_correction,
        launcher=args.launcher
    )
    
    # define the frame batches and run them through the pipeline
    skip_frames = int(reader.fps // inference_fps)
    frame_indexes = range(0, reader.frame_cnt, skip_frames)
    arr_chunks = [(i, i + batch_size)
                for i in range(0, len(frame_indexes), batch_size)]
    
    print(f'[INFO] Using text det model: {det_model}\n[INFO] Using text rec model: {rec_model}\n[INFO] Processing {inference_fps} frames per one second of video\n[INFO] Batch Size: {batch_size}\n\n\n')

    with open(absolute_subtitle_path, mode='w+') as f:
        sub = SubtitleWriter(f, inference_fps)
        for start, end in tqdm(arr_chunks):
            batch = [reader[i] for i in frame_indexes[start:end]]
            result = pipeline(batch, show=False, img_out_dir='')
            for i, res in enumerate(result):
                sub.write('\n'.join([e['text'] for e in res]), start+i)
        sub.finish()
    
    pipeline.print_metrics()
        
def test(args):
    pipeline = TestPipeline(
        det=args.det_model_name,
        recog=args.rec_model_name,
        batch_size=args.batch_size,
        run_err_correction=args.error_correction,
        launcher=args.launcher
    )
    metrics = pipeline.start_test()
    pprint(metrics)
    pipeline.write_metrics()

def run_fn_with_args(fn, args):
    while True:
        try:
            return fn(args)
        except Exception as e:
            if 'out of memory' in str(e) and args.batch_size > 1:
                args.batch_size -= 1
                print(f'[ERROR] Cuda out of memory error, retrying with batch size {args.batch_size}')
            else:
                print(f'[ERROR] Error occured while running {fn} on with {det} and {rec}')
                logging.error(e, exc_info=True)
                return None

if __name__ == '__main__':
    args = parse_args()
    
    # if multiple models have been chosen
    det_models = args.det_model_name.split(',')
    rec_models = args.rec_model_name.split(',')
    
    if args.mode == 'infer':    
        for det, rec in product(det_models, rec_models):
            args.det_model_name = det if det != 'None' else None
            args.rec_model_name = rec if rec != 'None' else None
            print (f"""\n\n\n\n{'*':*^74}\n*{'Inference':^72}*\n*{det:>35}--{rec:<35}*\n{'*':*^74}""")
            run_fn_with_args(inference, args)
    elif args.mode == 'test':
        # Pad the list with the minimum leingth with None
        max_len = max(len(det_models), len(rec_models))
        if len(det_models) < max_len:
            det_models.extend([None for _ in range(max_len-len(det_models))])
        if len(rec_models) < max_len:
            rec_models.extend([None for _ in range(max_len-len(rec_models))])
            
        for det, rec in zip(det_models, rec_models):
            args.det_model_name = det if det != 'None' else None
            args.rec_model_name = rec if rec != 'None' else None
            print (f"""\n\n\n\n{'*':*^74}\n*{'Testing':^72}*\n*{det or 'None':>35}--{rec or 'None':<35}*\n{'*':*^74}""")
            run_fn_with_args(test, args)
    else:
        raise f'Uknown mode. Expected "infer" or "test" Got {args.mode}'