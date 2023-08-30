import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import torch
from torch.multiprocessing import Pool, Process, set_start_method

class KeypointExtractor():
    def __init__(self):
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)   

    def extract_keypoint(self, images, name = None):
        # escape overlapped processing
        if name is not None:
            path = os.path.splitext(name)[0]+'.txt'
            # if os.path.exists(path):
            #     return None
                
        if isinstance(images, list):
            keypoints = []
            for image in tqdm(images, desc=path):
                current_kp = self.extract_keypoint(image, )
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(path, keypoints.reshape(-1))
            return keypoints
        else:
            try:
                keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
            except TypeError:
                print('No face detected in this image')
                shape = [68, 2]
                keypoints = -1. * np.ones(shape)                    
            if name is not None:
                np.savetxt(path, keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def run(data):
    torch.set_num_threads(1)
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    name = os.path.join(opt.output_dir, name[-2], name[-1])

    # path = os.path.splitext(name)[0]+'.txt'
    # if os.path.exists(path):
    #     return None
    images = read_video(filename)

    kp_extractor.extract_keypoint(images, name = name)

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    for ext in extensions:
        filenames = sorted(glob.glob(f'{opt.input_dir}/**/*.{ext}'))
        filenames = sorted(glob.glob(f'{opt.input_dir}/**/*.{ext}'))
        filenames = sorted(glob.glob(f'{opt.input_dir}/**/*.{ext}'))

    # filenames = [f for f in filenames if 'test' in f]
    # filenames = [f for f in filenames if 'WRA_VickyHartzler_000' in f]


    print('Total number of videos:', len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    # run((filenames[0], opt, opt.device_ids.split(",")[0]))
    torch.set_num_threads(1)

    # for data in tqdm(zip(filenames, args_list, device_ids)):
    #     run(data)

    # particular operations
    
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
