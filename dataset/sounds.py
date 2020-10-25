import os
import numpy as np

import torch.utils.data as data

from collections import Counter

SEED = 20200729
EVAL_RATIO = 0.1
PERIOD = 5  # seconds
SHIFT_LEN = 20  # micro-seconds
PERIOD_LEN = int(PERIOD * 1000 / SHIFT_LEN)
SEGMENT_LEN = int(4 * 1000 / SHIFT_LEN)  # 4ms overlap with previous period


class ListLoader(object):
    def __init__(self, root_path, num_classes, finetune=False):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of files for each category
        self.sound_list = []
        for directory in os.walk(root_path):
            for dir_name in directory[1]:  # All subdirectories
                type_id = int(dir_name)
                if type_id < 0 or type_id > num_classes:
                    print("Wrong directory: {}!".format(dir_name))
                    continue
                for _ in os.listdir(os.path.join(root_path, dir_name)):
                    self.category_count[type_id] += 1

                if not finetune and self.category_count[type_id] < 5:
                    continue

                count = 0
                for npy_file in os.listdir(os.path.join(root_path, dir_name)):
                    # Only choose A grade bird audio
                    if npy_file.split(".")[-2] != "A":
                        continue
                    full_path = os.path.join(root_path, dir_name, npy_file)
                    audio = np.load(full_path)

                    if audio.shape[0] < PERIOD_LEN:
                        continue

                    count += 1

                    for seg_index in range(audio.shape[0] // SEGMENT_LEN):
                        """sample = audio[index * period_len: (1 + index) * period_len]
                        print("sample:", sample.shape, sample.dtype)
                        if sample.shape[0] == period_len:"""
                        if seg_index * SEGMENT_LEN + PERIOD_LEN <= audio.shape[0]:
                            self.sound_list.append((full_path, seg_index, type_id))

        avg_count = sum(self.category_count.values()) / len(self.category_count)
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


class BirdsDataset(data.Dataset):
    """ All sounds and classes for birds through the world """

    def __init__(self, sound_list, sound_indices):
        self.sound_list = sound_list
        self.sound_indices = sound_indices

    def __getitem__(self, index):
        full_path, seg_index, type_id = self.sound_list[self.sound_indices[index]]
        audio = np.load(full_path)
        sample = audio[seg_index * SEGMENT_LEN: seg_index * SEGMENT_LEN + PERIOD_LEN]
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
