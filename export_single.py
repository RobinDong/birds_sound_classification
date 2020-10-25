import sys
import mfcc
import librosa
import argparse
import traceback
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=None, type=str, help="Input path for mp3 file"
    )
    parser.add_argument(
        "--output", default=None, type=str, help="Output path for numpy array"
    )
    args = parser.parse_args()

    try:
        audio, sr = librosa.load(args.input, sr=16000, mono=True, dtype=np.float32)
    except Exception as ex:
        print("Load autio failed:", ex)
        traceback.print_exc(file=sys.stdout)
        sys.exit(-1)

    audio = (audio * 32768).astype(np.int16)
    output = mfcc.mfcc(sr, audio)

    np.save(args.output, output)
