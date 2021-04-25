import os
import pathlib
import librosa
import multiprocessing
import numpy as np


class CFG:
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    n_mels = 128
    fmin = 20
    fmax = 16000
    period = 5  # seconds
    cpus = 4


class Generator:
    def __init__(self):
        pass

    def _transform(self, arguments):
        full_path, new_full_path = arguments
        try:
            audio, sr = librosa.load(full_path, mono=True, sr=CFG.sample_rate)
        except Exception:
            print(f"Couldn't open {full_path}")
            return
        duration = librosa.get_duration(y=audio, sr=sr)
        logmel = librosa.feature.melspectrogram(
                audio, sr,
                n_mels=CFG.n_mels,
                fmax=CFG.fmax)
        S_dB = librosa.power_to_db(logmel, ref=np.max)
        height, width = S_dB.shape
        with open(new_full_path, "wb") as fp:
            np.save(fp, S_dB)
        segment_step = int(CFG.period * width / duration)
        print("segment_step:", segment_step)

    def generate(self, src_dir, dest_dir, num_classes):
        task_list = []
        for directory in os.walk(src_dir):
            for dir_name in directory[1]:  # All subdirectories
                # create directory in destination
                pathlib.Path(
                    os.path.join(dest_dir, dir_name)
                ).mkdir(parents=True, exist_ok=True)
                for file in os.listdir(os.path.join(src_dir, dir_name)):
                    if file.endswith(".ogg"):
                        full_path = os.path.join(src_dir, dir_name, file)
                        new_full_path = os.path.join(
                            dest_dir,
                            dir_name,
                            file + ".npy")
                        task_list.append((full_path, new_full_path))

        with multiprocessing.Pool(processes=CFG.cpus) as pool:
            pool.map(self._transform, task_list)


if __name__ == "__main__":
    generator = Generator()
    generator.generate("/media/data2/sanbai/bird2021/train_short_audio/", "/media/data2/sanbai/bird2021.npy/", 400)
