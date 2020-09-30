import mfcc
import argparse
import torch
import torch.nn as nn
import numpy as np

import pycls.core.builders as builders
from pycls.core.config import cfg
from scipy.io import wavfile

PERIOD = 5  # seconds
SHIFT_LEN = 20  # micro-seconds
PERIOD_LEN = int(PERIOD * 1000 / SHIFT_LEN)


def load_label_file():
    map = {}
    with open("V7.new.txt") as fp:
        for line in fp.readlines():
            arr = line.split(",")
            id = arr[0]
            name = arr[1]
            eng_name = arr[2]
            map[int(id)] = name + " " + eng_name
    return map


def predict(args):
    cfg.MODEL.TYPE = "regnet"
    cfg.REGNET.DEPTH = 20
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 96
    cfg.BN.NUM_GROUPS = 8
    cfg.ANYNET.STEM_CHANNELS = 1
    cfg.MODEL.NUM_CLASSES = 1500
    net = builders.build_model()
    net.load_state_dict(torch.load(args.classify_model, map_location="cpu"))
    net.eval()

    softmax = nn.Softmax(dim=1)

    sr, audio = wavfile.read(args.sound_file)
    output = mfcc.mfcc(sr, audio)

    result = torch.from_numpy(np.zeros((1, cfg.MODEL.NUM_CLASSES), np.float32))
    # Split sound to a few pieces and predict them one by one, then add up the results
    for index in range(output.shape[0] // PERIOD_LEN):
        out = output[index * PERIOD_LEN : (index + 1) * PERIOD_LEN, :]
        tensor_sound = torch.from_numpy(out)
        result += net(
            tensor_sound.unsqueeze(0).unsqueeze(3).permute(0, 3, 1, 2).float()
        )
    result = softmax(result)
    print("result:", result, result.shape)
    values, indices = torch.topk(result, 10)

    label_map = load_label_file()
    print(values, indices)
    for ind, val in zip(indices[0], values[0]):
        print(ind.item(), label_map[ind.item()], f"({val.item()*100:.2f}%)")


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
