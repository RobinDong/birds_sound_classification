import os
import cv2
import shutil
import pathlib
import numpy as np

src_dir = '/media/data2/song/V2.training/'
dest_dir = '/media/data2/sanbai/V2.training/'
count = 0

for directory in os.walk(src_dir):
    for dir_name in directory[1]:  # All subdirectories
        # create directory in destination
        pathlib.Path(os.path.join(dest_dir, dir_name)).mkdir(parents=True, exist_ok=True)
        for file in os.listdir(os.path.join(src_dir, dir_name)):
            full_path = os.path.join(src_dir, dir_name, file)
            new_full_path = os.path.join(dest_dir, dir_name, file)
            if file.endswith(".npy"):
                audio = np.load(full_path)
                height, width = audio.shape
                audio = cv2.resize(audio, (width, 64))
                with open(new_full_path, 'wb') as fp:
                    np.save(fp, audio)
            elif file.endswith(".segments"):
                shutil.copyfile(full_path, new_full_path)
            else:
                raise Exception(f"Wrong file {full_path}")

            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} files {count*100/575794:.2f}%")
