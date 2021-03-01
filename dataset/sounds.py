import os
import cv2
import numpy as np

import torch.utils.data as data

from collections import Counter

SEED = 20200729
EVAL_RATIO = 0.2


class ListLoader(object):
    def __init__(self, root_path, num_classes):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of files for each category
        self.sound_list = []
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
                    if file.endswith("A_full.npy"):
                        self.category_count[type_id] += 1

                dir_count += 1

                for file in os.listdir(os.path.join(root_path, dir_name)):
                    if not file.endswith("A_full.npy"):
                        continue
                    full_path = os.path.join(root_path, dir_name, file)

                    # Find corresponding segments file
                    seg_files = os.path.join(
                        root_path, dir_name, file + ".segments"
                    )
                    splits = []
                    with open(seg_files, "r") as fp:
                        splits = eval(fp.read())
                    for begin, end in splits:
                        if (end - begin) != 78:
                            print(f"The part range is not 78. [{seg_files}]")
                            continue
                        self.sound_list.append(
                            (full_path, begin, end, type_id)
                        )

        np.random.shuffle(self.sound_list)
        np.random.shuffle(self.sound_list)
        print("sound_list:", self.sound_list[:10])

        avg_count = sum(self.category_count.values()) / len(
            self.category_count
        )
        print("Active categories:", dir_count)
        print("Avg count per category:", avg_count)
        minimum = min(self.category_count, key=self.category_count.get)
        print("Min count category:", self.category_count[minimum])
        maximum = max(self.category_count, key=self.category_count.get)
        print("Max count category:", self.category_count[maximum])

    def sound_indices(self):
        """Return train/eval sound files' list"""
        length = len(self.sound_list)
        indices = np.random.permutation(length)
        point = int(length * EVAL_RATIO)
        eval_indices = indices[0:point]
        train_indices = indices[point:]

        return self.sound_list, train_indices, eval_indices

    def export_labelmap(self, path="labelmap.csv"):
        with open(path, "w") as fp:
            for type_id, type_name in self.labelmap.items():
                count = self.category_count[type_id]
                fp.write(
                    str(type_id) + "," + type_name + "," + str(count) + "\n"
                )


class BirdsDataset(data.Dataset):
    """ All sounds and classes for birds through the world """

    def __init__(self, sound_list, sound_indices, train=True, finetune=False):
        self.sound_list = sound_list
        self.sound_indices = sound_indices
        self._train = train
        self._finetune = finetune
        # Create category map so we can uniformly
        # pick up samples from different categorys.
        self._category_map = {}
        for index in range(len(self.sound_indices)):
            full_path, begin, end, type_id = self.sound_list[
                self.sound_indices[index]
            ]
            if type_id in self._category_map:
                self._category_map[type_id].append(index)
            else:
                new_list = [index]
                self._category_map[type_id] = new_list
        self._category_list = list(self._category_map.keys())

    def export_samples(self, path="eval_list.txt"):
        with open(path, "w") as fp:
            for ind in self.sound_indices:
                fp.write(str(self.sound_list[ind]) + "\n")

    def _inflight_aug(self, sample):
        # First dice throwing
        dice = np.random.randint(3)
        if dice == 1:
            sample = cv2.blur(sample, (3, 3))
        elif dice == 2:
            sample = cv2.blur(sample, (5, 5))

        # Second dice throwing
        dice = np.random.randint(5)
        if dice == 0:
            return sample

        height = sample.shape[0]
        width = sample.shape[1]

        if dice == 1:
            # Time masking
            time_slot = np.random.randint(width // 10)
            time_begin = np.random.randint(0, width - time_slot)
            sample[:, time_begin: time_begin + time_slot] = sample.min()
            # Frequency masking
            freq_slot = np.random.randint(height // 10)
            freq_begin = np.random.randint(0, height - freq_slot)
            sample[freq_begin: freq_begin + freq_slot, :] = sample.min()
            return sample

        if dice == 2:
            # Time warping
            time_slot = np.random.randint(width // 10)
            if np.random.randint(2):  # From head
                sample = sample[:, time_slot:]
            else:  # From tail
                sample = sample[:, : width - time_slot]
            sample = cv2.resize(sample, (width, height))
            return sample

        if dice == 3:
            # Frequency warping
            freq_slot = np.random.randint(height // 10)
            if np.random.randint(2):  # From head
                sample = sample[freq_slot:, :]
            else:  # From tail
                sample = sample[: height - freq_slot, :]
            sample = cv2.resize(sample, (width, height))
            return sample

        if dice == 4:
            # Change loundness
            if np.random.randint(2):
                sample = sample * 1.05
            else:
                sample = sample * 0.95
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

        full_path, begin, end, type_id = self.sound_list[
            self.sound_indices[index]
        ]
        audio = np.load(full_path, mmap_mode="r")
        sample = audio[:, begin:end].copy()
        sample = cv2.resize(sample, (78, 1250 // 5 // 2))
        if self._train:
            sample = self._inflight_aug(sample)
        return sample, int(type_id)

    def __len__(self):
        return len(self.sound_indices)

    @staticmethod
    def my_collate(batch):
        batch = filter(lambda sound: sound is not None, batch)
        return data.dataloader.default_collate(list(batch))


if __name__ == "__main__":
    list_loader = ListLoader("V1", 100)
    sound_list, train_lst, eval_lst = list_loader.sound_indices()
    print("train_lst", train_lst, len(train_lst))
    print("eval_lst", eval_lst, len(eval_lst))

    bd = BirdsDataset(sound_list, eval_lst)
    sound, type_id = bd[3]
    print("sound", sound.shape, sound)
    print("type_id", type_id, type(type_id))
