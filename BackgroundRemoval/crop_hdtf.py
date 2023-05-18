import numpy as np
import pandas as pd
import imageio
from skimage.transform import resize
from argparse import ArgumentParser
from skimage import img_as_ubyte
import os
import subprocess
from multiprocessing import Process
import warnings
import glob
from tqdm import tqdm
import face_alignment
from util import bb_intersection_over_union, join, scheduler, crop_bbox_from_frames, save

warnings.filterwarnings("ignore")

TEST_PERSONS = [
    'WDA_CatherineCortezMasto_000',
    'WDA_MartinHeinrich_000',
    'WDA_MikeDoyle_000',
    'WRA_DeanHeller_000',
    'WRA_EricCantor_000',
    'WRA_ErikPaulsen_000',
    'WRA_GeoffDavis_000',
    'WRA_JoePitts_000',
    'WRA_JohnThune_000',
    'WRA_OlympiaSnowe_000',
    'WRA_ReincePriebus_000',
    'WRA_ReneeEllmers_000',
    'WRA_RichardShelby_000',
    'WRA_RobPortman_000',
    'WRA_RoyBlunt_000',
    'WRA_TimScott_000',
    'WRA_ToddYoung_000',
    'WRA_TomCotton_000',
    'WRA_TomPrice_000',
    'WRA_VickyHartzler_000'
    ]

def extract_bbox(frame, fa):
    return fa.face_detector.detect_from_image(frame[..., ::-1])[0]
    # mult = frame.shape[0] / REF_FRAME_SIZE
    # frame = resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True)
    # bbox = fa.face_detector.detect_from_image(frame[..., ::-1])[0]    
    # return bbox


def store(frame_list, tube_bbox, video_id, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=0,
                                            image_shape=args.image_shape, min_size=0, 
                                            increase_area=args.increase)
    if out is None:
        return []

    name = video_id
    partition = 'test' if any([person in name for person in TEST_PERSONS]) else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format)

def process_video(video_path, args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    video_id = os.path.splitext(os.path.split(video_path)[-1])[0]
    reader = imageio.get_reader(video_path)
    tube_bbox = None
    frame_list = []
    for i, frame in tqdm(enumerate(reader), desc=video_id):
        if i == 0:
            try:
                bbox = extract_bbox(frame, fa)
                #left, top, right, bot, _ = bbox
                tube_bbox = bbox[:-1]
            except:
                return {}
        frame_list.append(frame)
    return store(frame_list, tube_bbox, video_id, args)

def run(params):
    video_id, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    return process_video(video_id, args)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--in_folder", default = 'hdtf_videos')
    parser.add_argument("--out_folder", default = 'hdtf_videos_aligned')
    parser.add_argument("--increase", default = 0.3, type=float, help='Increase bbox by this amount') 
    parser.add_argument("--format", default='.png', help='Store format (.png, .mp4)')
    parser.add_argument("--chunks_metadata", default='nemo-metadata.csv', help='Path to store metadata')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")

    parser.add_argument("--workers", default=1, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0", help="Names of the devices comma separated.")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        os.makedirs(args.out_folder + '/train')
        os.makedirs(args.out_folder + '/test')


    ids = sorted(
            sorted([os.path.join(args.in_folder + '/train', f) for f in os.listdir(args.in_folder + '/train')]) + \
            sorted([os.path.join(args.in_folder + '/test', f) for f in os.listdir(args.in_folder + '/test')])
        )

    ids = [f for f in ids if 'test' in f]

    scheduler(ids, run, args)