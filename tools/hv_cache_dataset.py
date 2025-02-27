import math
import multiprocessing as mp
import os
import pprint
import traceback
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from typing import Tuple, Union

import cv2
import decord
import numpy as np
import pandas as pd
from PIL import Image
from streaming import MDSWriter
from tqdm.auto import tqdm

decord.bridge.set_bridge("native")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--base_dir", default=None, type=str)
    parser.add_argument("--video_column", default="video", type=str)
    parser.add_argument("--caption_column", default="caption", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--bucket_reso", default=None, type=str, nargs="+")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--min_bucket_count", type=int, default=1)
    parser.add_argument("--head_frame", type=int, default=0)

    args = parser.parse_args()
    print("args:", pprint.pformat(args, sort_dicts=True, compact=True))
    return args


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso
    if bucket_width == image_width or bucket_height == image_height:
        image = np.array(image) if is_pil_image else image
    else:
        # resize the image to the bucket resolution to match the short side
        scale_width = bucket_width / image_width
        scale_height = bucket_height / image_height
        scale = max(scale_width, scale_height)
        image_width = int(image_width * scale + 0.5)
        image_height = int(image_height * scale + 0.5)

        if scale > 1:
            image = Image.fromarray(image) if not is_pil_image else image
            image = image.resize((image_width, image_height), Image.LANCZOS)
            image = np.array(image)
        else:
            image = np.array(image) if is_pil_image else image
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


class BucketSelector:
    RESOLUTION_STEPS_HUNYUAN = 16

    def __init__(self, resolution: Tuple[int, int], enable_bucket: bool = True, no_upscale: bool = False):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.reso_steps = BucketSelector.RESOLUTION_STEPS_HUNYUAN

        if not enable_bucket:
            # only define one bucket
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]


def load_video(video_path, bucket_selector=None, start_frame=None, end_frame=None):
    vr = decord.VideoReader(uri=video_path)
    video_num_frames = len(vr)
    _start_frame, _end_frame = 0, video_num_frames

    if start_frame is not None:
        _start_frame = start_frame
    if end_frame is not None:
        _end_frame = min(_end_frame, end_frame)

    frames = vr.get_batch(range(_start_frame, _end_frame)).asnumpy()
    f, h, w, _ = frames.shape
    bucket_reso = bucket_selector.get_bucket_resolution(image_size=(w, h))
    frames = [resize_image_to_bucket(frame, bucket_reso=bucket_reso) for frame in frames]
    return frames


class BucketBatchManager:
    def __init__(self, bucketed_item_info, min_bucket_count=0):
        self.buckets = bucketed_item_info
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()

        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            if len(bucket) < min_bucket_count:
                print(
                    f"bucket {bucket_reso!r} (n={len(bucket)!r}) has less than {min_bucket_count!r} items, remove it..."
                )
                del self.buckets[bucket_reso]

        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            print(f"bucket: {bucket_reso}, count: {len(bucket)}")


def convert_and_make_shards(args, base_df, bucket_batch_manager, bucket_reso, b_idx):
    columns = {
        "idx": "int32",
        "item_key": "str",
        "item": "str",
        "frame_count": "int32",
        "bucket_width": "int32",
        "bucket_height": "int32",
        "original_width": "int32",
        "original_height": "int32",
        "caption_str": "str",
        "video_arr": "ndarray",
    }
    print(f"Starting converter processs for bucket {bucket_reso!r}...")
    output_path = os.path.join(args.output_dir, "x".join(list(map(str, bucket_reso))))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    bucket = bucket_batch_manager.buckets[bucket_reso]

    writer = MDSWriter(out=output_path, columns=columns, size_limit=256 * (2**20), max_workers=os.cpu_count())

    for item_info in tqdm(bucket, dynamic_ncols=True, position=b_idx, leave=False, desc=f"bucket {bucket_reso}"):
        item_key = item_info["item_key"]
        frame_count = item_info["frame_count"]
        frame_crop_pos = item_info["frame_crop_pos"]
        idx = item_info["idx"]
        bucket_reso_wh = (item_info["bucket_width"], item_info["bucket_height"])

        row = base_df.iloc[idx]
        video_path, caption = row[args.video_column], row[args.caption_column]

        try:
            vr = decord.VideoReader(uri=video_path)
            video = vr.get_batch(range(frame_crop_pos, frame_crop_pos + frame_count)).asnumpy()
            original_width, original_height = video.shape[2], video.shape[1]
            video = [resize_image_to_bucket(frame, bucket_reso=bucket_reso_wh) for frame in video]
            video = np.stack(video, axis=0)
        except Exception as e:
            print(f"Failed to load video {video_path!r} : {e!r}")
            print(traceback.format_exc())
            continue

        sample = {}
        sample["idx"] = idx
        sample["item_key"] = str(item_key)
        sample["item"] = str(Path(video_path).name)
        sample["frame_count"] = frame_count
        sample["bucket_width"] = bucket_reso_wh[0]
        sample["bucket_height"] = bucket_reso_wh[1]
        sample["original_width"] = original_height
        sample["original_height"] = original_width
        sample["caption_str"] = caption
        sample["video_arr"] = video

        writer.write(sample)
    writer.finish()
    print(f"Converter process finished for bucket {bucket_reso!r} !!!")


def main(args):
    if str(args.csv).endswith(".csv"):
        df = pd.read_csv(args.csv)
    elif str(args.csv).endswith(".json"):
        df = pd.read_json(args.csv)
    elif str(args.csv).endswith(".parquet"):
        df = pd.read_parquet(args.csv)
    elif str(args.csv).endswith(".jsonl"):
        df = pd.read_json(args.csv, lines=True, orient="records")
    else:
        raise ValueError(f"Invalid csv path: {args.csv!r}")
    if args.base_dir is not None:
        df[args.video_column] = df[args.video_column].apply(lambda x: os.path.join(args.base_dir, x))

    if args.debug:
        df = df.sample(n=10).reset_index(drop=True, inplace=False)

    print("Total number of samples: ", len(df))

    bucket_selectors = []
    for res in args.bucket_reso:
        w, h, f = res.split("x")
        bs = BucketSelector(resolution=(int(w), int(h), int(f)), enable_bucket=True, no_upscale=False)
        bucket_selectors.append(bs)

    batches = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True, desc="Generating buckets"):
        video_path = row[args.video_column]

        vr = decord.VideoReader(uri=video_path)
        frame_count = len(vr)
        video = vr.get_batch(range(0, 1)).asnumpy()
        frame_height, frame_width = video.shape[1], video.shape[2]
        frame_size = (frame_width, frame_height)

        for bs in bucket_selectors:
            target_frame = bs.resolution[-1]
            bucket_reso = bs.get_bucket_resolution(frame_size)

            if frame_count >= (target_frame + args.head_frame):
                crop_pos_and_frames = [args.head_frame, target_frame]
                body, ext = os.path.splitext(Path(video_path).name)
                item_key = f"{body}_{crop_pos_and_frames[0]:05d}-{target_frame:05d}{ext}"
                batch_key = (*bucket_reso, target_frame)
                item_info = {
                    "item_key": item_key,
                    "batch_key": batch_key,
                    "frame_count": target_frame,
                    "frame_crop_pos": crop_pos_and_frames[0],
                    "idx": idx,
                    "bucket_height": bucket_reso[1],
                    "bucket_width": bucket_reso[0],
                }
                batch = batches.get(batch_key, [])
                batch.append(item_info)
                batches[batch_key] = batch

    bucket_manager = BucketBatchManager(batches, min_bucket_count=args.min_bucket_count)
    bucket_manager.show_bucket_info()

    bucket_resos = bucket_manager.bucket_resos

    exporters = []
    for bucket_idx, bucket_reso in enumerate(bucket_resos):
        op = mp.Process(target=convert_and_make_shards, args=(args, df, bucket_manager, bucket_reso, bucket_idx))
        op.start()
        exporters.append(op)

    for op in exporters:
        op.join()
        op.close()


if __name__ == "__main__":
    main(parse_args())
