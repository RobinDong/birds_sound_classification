import os
import pathlib
import argparse
import numpy as np

import torch
import torch.nn as nn

import pycls.core.builders as builders
from pycls.core.config import cfg

from utils import augment

SEGMENT_SIZE = 312
MAX_SIZE = int(600 // 5 * SEGMENT_SIZE)  # 10 mins
sample_rate = 32000


def load_label_file():
    map = {}
    with open("labelmap.csv") as fp:
        for line in fp.readlines():
            arr = line.split(",")
            id = arr[0]
            name = arr[1]
            map[int(id)] = name
    return map


def load_models(model_files: str):
    cfg.MODEL.TYPE = "regnet"
    # RegNetY-3.2GF
    cfg.REGNET.DEPTH = 18
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 200
    cfg.REGNET.WA = 106.23
    cfg.REGNET.WM = 2.48
    cfg.REGNET.GROUP_W = 112
    cfg.BN.NUM_GROUPS = 4
    cfg.ANYNET.STEM_CHANNELS = 1
    cfg.MODEL.NUM_CLASSES = 10958

    models = []
    for file in model_files.split(","):
        net = builders.build_model()
        net = net.cuda(device=torch.cuda.current_device())
        net.load_state_dict(torch.load(file))
        net.eval()
        models.append(net)
    return models


def process(sound_path, label_path, models, softmax, aug):
    if sound_path.endswith(".mp3.npy"):
        audio = np.load(sound_path)
        audio_len = min(MAX_SIZE, audio.shape[1])
        segs = audio_len // SEGMENT_SIZE
        for index in range(segs):
            begin = index * SEGMENT_SIZE
            end = begin + SEGMENT_SIZE
            sample = audio[:, begin:end].copy()
            sample = torch.from_numpy(sample)
            sample = sample.unsqueeze(0).unsqueeze(3)
            sample = aug(sample)
            sample = sample.permute(0, 3, 1, 2).float().cuda()
            results = []
            invalid_sample = False
            for model in models:
                result = model(sample)
                result = softmax(result)
                if result.isnan().any() or not result.isfinite().all():
                    print(f"{sound_path} has nan")
                    invalid_sample = True
                    break
                results.append(result.cpu().detach().numpy()[0])
            if invalid_sample:
                continue
            np.save(label_path + f".{index}", np.array(results).mean(axis=0))


def travel(args):
    models = load_models(args.model_files)
    softmax = nn.Softmax(dim=1)
    aug = augment.Augment(training=False)

    for directory in os.walk(args.sound_path):
        for dir_name in directory[1]:  # All subdirectories
            # create directory in destination
            pathlib.Path(os.path.join(args.label_path, dir_name)).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(os.path.join(args.sound_path, dir_name)):
                sound_path = os.path.join(args.sound_path, dir_name, file)
                label_path = os.path.join(args.label_path, dir_name, file)
                process(sound_path, label_path, models, softmax, aug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_files", default=None, type=str, help="Pathes of model files, separated by comma"
    )
    parser.add_argument(
        "--sound_path", default="/media/data2/song/V7.npy", type=str, help="Root path of sounds",
    )
    parser.add_argument(
        "--label_path", default="/media/data2/label/V7.npy", type=str, help="Root path of sounds",
    )
    args = parser.parse_args()

    travel(args)
