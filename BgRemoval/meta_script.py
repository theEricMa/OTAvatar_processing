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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    output_filename = filename.replace(opt.input_dir, opt.output_dir)
    # if os.path.exists(output_filename):
    #     exit(0)
    cmd = "python -m demo.video_matting.custom.run" + \
        " --video " + filename + \
        " --output " + output_filename + \
        " --result-type " + opt.result_type + \
        " --fps " + str(opt.fps)
    os.system(cmd)

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder to the output files')
    parser.add_argument('--result-type', type=str, default='fg', choices=['fg', 'matte'], 
                        help='matte - save the alpha matte; fg - save the foreground')
    parser.add_argument('--fps', type=int, default=30, help='fps of the result video')
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
        # filenames += sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))

    # # particular operations
    # filenames = [f for f in filenames if 'test' in f]
    
    print('Total number of videos:', len(filenames))
    device_ids = opt.device_ids.split(",")

    # run((filenames[0], opt, device_ids[0]))

    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None