import os
import cv2
import numpy as np

import torch.utils.data as data

from collections import Counter

SEED = 20200729
EVAL_RATIO = 0.05
FILE_PATTERN = ".npy"
JPEG_PATTERN = ".jpeg"
SEGMENT_SIZE = 312
MAX_SIZE = int(600 // 5 * SEGMENT_SIZE)  # 10 mins


class ListLoader(object):
    def __init__(self, root_path, num_classes, distill_mode=False, label_path=None):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of files for each category
        self.train_sound_list = []
        self.eval_sound_list = []
        self.file_list = []  # don't get train/eval data from the same file
        self.eval_files = set()
        self.labelmap = {}
        dir_count = 0
        for directory in os.walk(root_path):
            for dir_name in directory[1]:  # All subdirectories
                pos = dir_name.find(".")
                type_id = int(dir_name[0:pos])
                type_name = dir_name[pos + 1:]
                if type_id < 0 or type_id >= num_classes:
                    print("Wrong directory: {}!".format(dir_name))
                    continue
                self.labelmap[type_id] = type_name
                for file in os.listdir(os.path.join(root_path, dir_name)):
                    if file.endswith(FILE_PATTERN) or file.endswith(JPEG_PATTERN):
                        full_path = os.path.join(root_path, dir_name, file)
                        self.file_list.append(full_path)
                        self.category_count[type_id] += 1

                dir_count += 1
                # Only leave eval files in eval_files
                np.random.shuffle(self.file_list)
                nr_files = len(self.file_list)
                self.eval_files = {file for file in self.file_list[0:int(nr_files * EVAL_RATIO)]}

                for file in os.listdir(os.path.join(root_path, dir_name)):
                    full_path = os.path.join(root_path, dir_name, file)
                    if full_path in self.eval_files:
                        sound_list = self.eval_sound_list
                    else:
                        sound_list = self.train_sound_list

                    if file.endswith(FILE_PATTERN):
                        audio = np.load(full_path, mmap_mode="r")
                        audio_len = min(MAX_SIZE, audio.shape[1])
                        segs = audio_len // SEGMENT_SIZE
                        for index in range(segs):
                            begin = index * SEGMENT_SIZE
                            end = begin + SEGMENT_SIZE
                            if end > audio_len:
                                print(f"{end} is out of range {audio_len} [{full_path}]")
                                continue
                            if distill_mode:
                                label_file = os.path.join(label_path, dir_name, file) + f".{index}.npy"
                                sound_list.append((full_path, begin, end, type_id, label_file))
                            else:
                                sound_list.append((full_path, begin, end, type_id))
                    elif file.endswith(JPEG_PATTERN):
                        sound_list.append((full_path, type_id))

        np.random.shuffle(self.train_sound_list)
        avg_count = sum(self.category_count.values()) / len(
            self.category_count
        )
        print("Active categories:", dir_count)
        print("Avg count per category:", avg_count)
        minimum = min(self.category_count, key=self.category_count.get)
        print("Min count category:", self.category_count[minimum])
        maximum = max(self.category_count, key=self.category_count.get)
        print("Max count category:", self.category_count[maximum])
        print("Train sounds:", len(self.train_sound_list))
        print("Eval sounds:", len(self.eval_sound_list))

    def sound_lists(self):
        """Return train/eval sound files' list"""
        return self.train_sound_list, self.eval_sound_list

    def export_labelmap(self, path="labelmap.csv"):
        with open(path, "w") as fp:
            for type_id, type_name in self.labelmap.items():
                count = self.category_count[type_id]
                fp.write(
                    str(type_id) + "," + type_name + "," + str(count) + "\n"
                )


class BirdsDataset(data.Dataset):
    """ All sounds and classes for birds through the world """

    def __init__(self, sound_list, train=True, finetune=False):
        self.sound_list = sound_list
        self._train = train
        self._finetune = finetune
        # Create category map so we can uniformly
        # pick up samples from different categorys.
        self._category_map = {}
        if finetune and train:
            for index in range(len(self.sound_list)):
                item = self.sound_list[index]
                if len(item) > 2:
                    if len(item) == 5:  # add distilling labels
                        full_path, begin, end, type_id, _ = item
                    else:
                        full_path, begin, end, type_id = item
                else:
                    full_path, type_id = item
                if type_id in self._category_map:
                    self._category_map[type_id].append(index)
                else:
                    new_list = [index]
                    self._category_map[type_id] = new_list
            self._category_list = list(self._category_map.keys())

    def export_samples(self, path="eval_list.txt"):
        with open(path, "w") as fp:
            for index in range(len(self.sound_list)):
                fp.write(str(self.sound_list[index]) + "\n")

    def _inflight_aug(self, sample):
        # First dice throwing
        '''dice = np.random.randint(3)
        if dice == 1:
            sample = cv2.blur(sample, (3, 3))
        elif dice == 2:
            sample = cv2.blur(sample, (5, 5))'''

        # Second dice throwing
        dice = np.random.randint(5)
        if dice == 0:
            return sample

        height = sample.shape[0]
        width = sample.shape[1]

        cut_ratio = 5
        if dice == 1:
            # Time masking
            time_slot = np.random.randint(width // cut_ratio)
            time_begin = np.random.randint(0, width - time_slot)
            sample[:, time_begin: time_begin + time_slot] = sample.min()
            # Frequency masking
            freq_slot = np.random.randint(height // cut_ratio)
            freq_begin = np.random.randint(0, height - freq_slot)
            sample[freq_begin: freq_begin + freq_slot, :] = sample.min()
            return sample

        if dice == 2:
            # Time warping
            time_slot = np.random.randint(width // cut_ratio)
            if np.random.randint(2):  # From head
                sample = sample[:, time_slot:]
            else:  # From tail
                sample = sample[:, : width - time_slot]
            sample = cv2.resize(sample, (width, height))
            return sample

        if dice == 3:
            # Frequency warping
            freq_slot = np.random.randint(height // cut_ratio)
            if np.random.randint(2):  # From head
                sample = sample[freq_slot:, :]
            else:  # From tail
                sample = sample[: height - freq_slot, :]
            sample = cv2.resize(sample, (width, height))
            return sample

        if dice == 4:
            # Change loundness
            if np.random.randint(2):
                sample = sample * 1.10
            else:
                sample = sample * 0.90
            return sample

        return sample

    def __getitem__(self, index):
        if self._finetune and self._train:
            # Rotately choose a category
            idx = index % len(self._category_list)
            lst = self._category_map[self._category_list[idx]]
            # Randomly choose a index in this category
            ind = np.random.randint(0, len(lst))
            index = lst[ind]

        item = self.sound_list[index]

        if len(item) > 2:
            if len(item) == 5:  # add distilling labels
                full_path, begin, end, type_id, label_file = item
                if not os.path.isfile(label_file):
                    return None
                label = np.load(label_file)
                if np.isnan(label).any() or not np.isfinite(label).all():
                    return None
            else:
                full_path, begin, end, type_id = item
            audio = np.load(full_path, mmap_mode="r")
            # temporary augmentation
            sample = audio[:, begin:end].copy()
            if self._train:
                sample = self._inflight_aug(sample)
        else:
            full_path, type_id = item
            img = cv2.imread(full_path)
            sample = img[:, :, 0]

        if len(item) == 5:  # add distilling labels
            return sample, int(type_id), label
        else:
            return sample, int(type_id)

    def __len__(self):
        return len(self.sound_list)

    @staticmethod
    def my_collate(batch):
        batch = filter(lambda sound: sound is not None, batch)
        return data.dataloader.default_collate(list(batch))

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)


if __name__ == "__main__":
    list_loader = ListLoader("/media/data2/song/V7.npy/", 20000)
    train_lst, eval_lst = list_loader.sound_lists()
    print("train_lst", train_lst, len(train_lst))
    print("eval_lst", eval_lst, len(eval_lst))

    bd = BirdsDataset(eval_lst)
    sound, type_id = bd[3]
    print("sound", sound.shape, sound)
    print("type_id", type_id, type(type_id))
