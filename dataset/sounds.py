import os
import librosa
import numpy as np

import torch.utils.data as data

from collections import Counter

SEED = 20200729
EVAL_RATIO = 0.1
PERIOD = 5  # seconds
MFCCS = 80


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
                for sound_file in os.listdir(os.path.join(root_path, dir_name)):
                    self.category_count[type_id] += 1

                if not finetune and self.category_count[type_id] < 100:
                    continue

                for sound_file in os.listdir(os.path.join(root_path, dir_name)):
                    full_path = os.path.join(root_path, dir_name, sound_file)
                    audio, sample_rate = librosa.load(full_path)
                    seconds = audio.shape[0] / sample_rate
                    if seconds < PERIOD:
                        continue
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCCS)
                    ratio = int(mfccs.shape[1] / seconds)
                    for index in range(int(seconds) // PERIOD):
                        sample = mfccs[
                            :, index * PERIOD * ratio : (1 + index) * PERIOD * ratio
                        ]
                        if sample.shape[1] == PERIOD * ratio:
                            self.sound_list.append((sample, type_id))

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
        sample, type_id = self.sound_list[self.sound_indices[index]]
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
