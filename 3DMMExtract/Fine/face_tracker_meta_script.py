import os
import glob
import argparse
import torch

from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, set_start_method

def run(data):
    filename, opt, device_id = data
    torch.cuda.empty_cache()
    keypoint_filename = filename.replace(opt.input_dir, opt.keypoint_dir).replace('mp4', 'txt')
    debug_filename = filename.replace(opt.input_dir, opt.debug_dir).replace('mp4', 'mp4')
    output_filename = filename.replace(opt.input_dir, opt.output_dir).replace('mp4', 'mat')
    cmd = 'python face_tracker_ours.py' + \
            ' --input_video ' + filename + \
            ' --output_file ' + output_filename + \
            ' --keypoint_file ' + keypoint_filename + \
            ' --debug_video ' + debug_filename + \
            ' --img_h ' + str(opt.img_h) + \
            ' --img_w ' + str(opt.img_w) + \
            ' --device_id ' + str(device_id) 
    os.system(cmd)

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--keypoint_dir', type=str, help='the folder to the output files')
    parser.add_argument('--output_dir', type=str, help='the folder to the output files')
    parser.add_argument('--debug_dir', type=str, help='the folder to the output files')
    parser.add_argument('--img_h', type=int, default=256, help='image height')
    parser.add_argument('--img_w', type=int, default=256, help='image width')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device_ids', type=str, default='1')
    opt = parser.parse_args()

    # search files
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    filenames = []
    for ext in extensions:
        filenames += sorted(glob.glob(f'{opt.input_dir}/**/*.{ext}'))
    keypoint_filenames = sorted(glob.glob(f'{opt.keypoint_dir}/**/*.txt', recursive=True))

    # # particular operation
    # filenames = [f for f in filenames if 'test' in f]
    # keypoint_filenames = [f for f in keypoint_filenames if 'test' in f]

    print('Total number of videos:', len(filenames))
    device_ids = opt.device_ids.split(",")

    # run((filenames[0], opt, device_ids[0]))

    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None


