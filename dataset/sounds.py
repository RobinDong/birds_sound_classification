import os
import numpy as np

import torch.utils.data as data

from random import shuffle
from collections import Counter

SEED = 20200729
EVAL_RATIO = 0.2


class ListLoader(object):
    def __init__(self, root_path, num_classes, finetune=False):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of files for each category
        self.sound_list = []
        self.labelmap = {}
        dir_count = 0
        for directory in os.walk(root_path):
            for dir_name in directory[1]:  # All subdirectories
                pos = dir_name.find(".")
                type_id = int(dir_name[0:pos])
                type_name = dir_name[pos+1:]
                if type_id < 0 or type_id >= num_classes:
                    print("Wrong directory: {}!".format(dir_name))
                    continue
                self.labelmap[type_id] = type_name
                for file in os.listdir(os.path.join(root_path, dir_name)):
                    if file.endswith("npy"):
                        self.category_count[type_id] += 1

                if not finetune and self.category_count[type_id] < 20:
                    continue

                dir_count += 1

                enough = False
                count = 0
                for file in os.listdir(os.path.join(root_path, dir_name)):
                    if not file.endswith("npy"):
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
                        count += 1
                        if count > 50:
                            enough = True
                            break
                    if enough:
                        break

        shuffle(self.sound_list)
        shuffle(self.sound_list)

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

    def __init__(self, sound_list, sound_indices):
        self.sound_list = sound_list
        self.sound_indices = sound_indices

    def __getitem__(self, index):
        full_path, begin, end, type_id = self.sound_list[
            self.sound_indices[index]
        ]
        audio = np.load(full_path)
        sample = audio[:, begin:end].copy()
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
