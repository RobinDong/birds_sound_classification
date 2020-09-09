import os
import sys
import time
import argparse
import traceback
import subprocess


def process(full_path, args):
    sub_dir = full_path.split("/")[-2]
    output_path = os.path.join(args.output_root, sub_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for _ in range(10):
        try:
            file_size = os.stat(full_path).st_size
        except Exception as ex:
            print("Stat file failed:", ex)
            traceback.print_exc(file=sys.stdout)
            time.sleep(10)
            continue
        break

    if file_size <= 4096 or file_size > (100 * 1048576):
        return

    print("full_path:", full_path)
    sound_name, _ = os.path.splitext(full_path.split("/")[-1])
    new_path = os.path.join(output_path, f"{sound_name}.npy")

    for _ in range(10):
        result = subprocess.run(
            f"python3 -u export_single.py --input \"{full_path}\" --output \"{new_path}\"",
            shell=True)
        if result.returncode != 0:
            print("retry")
        else:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpus", default=6, type=int, help="Number of cpu cores"
    )
    parser.add_argument(
        "--output_root", default="/media/data2/sanbai/mfcc_sound", type=str, help="Root directory of output"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    finish_list = set()
    if os.path.exists("finish.lst"):
        with open("finish.lst") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                finish_list.add(line)

    finish_file = open("finish.lst", "a+")

    with open("task.lst") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line in finish_list:
                continue
            process(line.strip(), args)
            finish_file.write(line)
