import os
import cv2
import pathlib
import numpy as np

SEGMENT_SIZE = 312
MAX_SIZE = int(600 // 5 * SEGMENT_SIZE)  # 10 mins

src_dir = '/media/data2/song/V7.npy/'
dest_dir = '/media/data2/song/V7.jpeg/'
count = 0

for directory in os.walk(src_dir):
    for dir_name in directory[1]:  # All subdirectories
        # create directory in destination
        pathlib.Path(os.path.join(dest_dir, dir_name)).mkdir(parents=True, exist_ok=True)
        for file in os.listdir(os.path.join(src_dir, dir_name)):
            full_path = os.path.join(src_dir, dir_name, file)
            new_full_path = os.path.join(dest_dir, dir_name, file)
            if file.endswith(".mp3.npy"):
                audio = np.load(full_path)
                audio = (audio / 80 + 1) * 255  # Convert to pixel (0~255)
                audio = audio.astype(np.int16)
                audio_len = min(MAX_SIZE, audio.shape[1])
                segs = audio_len // SEGMENT_SIZE
                for index in range(segs):
                    begin = index * SEGMENT_SIZE
                    end = begin + SEGMENT_SIZE
                    if end > audio_len:
                        print(f"{end} is out of range {audio_len} [{full_path}]")
                        continue
                    # new_name = new_full_path[:-8] + "_" + str(index) + ".jpeg"
                    # cv2.imwrite(new_name, audio[:, begin:end])
                    new_name = new_full_path[:-8] + "_" + str(index) + ".npz"
                    cv2.imwrite(new_name, audio[:, begin:end])
            else:
                raise Exception(f"Wrong file {full_path}")

            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} files {count*100/575794:.2f}%")
