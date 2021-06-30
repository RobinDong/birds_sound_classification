import librosa
import argparse
import numpy as np

import torch
import torch.nn as nn

import pycls.core.builders as builders
from pycls.core.config import cfg

from utils import augment

from dataset.generate import CFG

SEGMENT_SIZE = 312
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


def predict(args):
    cfg.MODEL.TYPE = "regnet"
    cfg.REGNET.DEPTH = 21
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 80
    cfg.REGNET.WA = 42.63
    cfg.REGNET.WM = 2.66
    cfg.REGNET.GROUP_W = 24
    cfg.BN.NUM_GROUPS = 4
    cfg.ANYNET.STEM_CHANNELS = 1
    cfg.MODEL.NUM_CLASSES = 10958
    net = builders.build_model()
    net.load_state_dict(torch.load(args.classify_model, map_location="cpu"))
    net.eval()

    softmax = nn.Softmax(dim=1)
    label_map = load_label_file()

    # Load audio file to np.array
    #audio, sr = librosa.load(args.sound_file, mono=True, offset=1.1, sr=CFG.sample_rate)
    #logmel = librosa.feature.melspectrogram(audio, sr, n_mels=CFG.n_mels, fmax=CFG.fmax)
    #S_dB = librosa.power_to_db(logmel, ref=np.max)
    S_dB = np.load(args.sound_file)

    aug = augment.Augment(training=False)
    segs = S_dB.shape[1] // SEGMENT_SIZE
    for index in range(segs):
        begin = index * SEGMENT_SIZE
        end = begin + SEGMENT_SIZE
        if end > S_dB.shape[1]:
            print(f"{end} is out of range {S_dB.shape[1]} [{args.sound_file}]")
            continue
        sample = S_dB[:, begin:end].copy()
        sample = torch.from_numpy(sample)
        sample = sample.unsqueeze(0).unsqueeze(3)
        sample = aug(sample)
        sample = sample.permute(0, 3, 1, 2).float()
        result = net(sample)
        result = softmax(result)
        values, indices = torch.topk(result, 5)
        print("-----------------------------------------------")
        for ind, val in zip(indices[0], values[0]):
            ind = ind.item()
            # if ind > 0 and ind < 10950:
            print(ind, label_map[ind], f"({val.item()*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sound_file", default=None, type=str, help="Sound file to be predicted"
    )
    parser.add_argument(
        "--classify_model",
        default="ckpt/bird_cls_0.pth",
        type=str,
        help="Trained ckpt file path to open",
    )
    args = parser.parse_args()

    predict(args)
