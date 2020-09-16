import os
import sys
import time
import argparse
import traceback
import subprocess

from functools import partial
from multiprocessing import Pool, current_process


RETRY = 10
RETRY_WAIT = 10  # seconds

TOO_SMALL_SIZE = 4096
TOO_BIG_SIZE = 100 * 1048576


def process(full_path, args):
    sub_dir = full_path.split("/")[-2]
    output_path = os.path.join(args.output_root, sub_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for _ in range(RETRY):
        try:
            file_size = os.stat(full_path).st_size
        except Exception as ex:
            print("Stat file failed:", ex)
            traceback.print_exc(file=sys.stdout)
            time.sleep(RETRY_WAIT)
            continue
        break

    if file_size <= TOO_SMALL_SIZE or file_size > TOO_BIG_SIZE:
        return

    print("full_path:", full_path)
    sound_name, _ = os.path.splitext(full_path.split("/")[-1])
    new_path = os.path.join(output_path, f"{sound_name}.npy")

    for _ in range(RETRY):
        result = subprocess.run(
            f"python3 -u export_single.py --input \"{full_path}\" --output \"{new_path}\"",
            shell=True)
        if result.returncode != 0:
            print("retry")
        else:
            break

    job_index = current_process().name.split("-")[-1]
    job_index = int(job_index) - 1
    with open(f"finish.{job_index}.lst", "a+") as fp:
        fp.write(full_path + "\n")


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
    for cpu in range(args.cpus):
        if os.path.exists(f"finish.{cpu}.lst"):
            with open(f"finish.{cpu}.lst") as fp:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    finish_list.add(line)

    need_to_do = []
    with open("task.lst") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line in finish_list:
                continue
            need_to_do.append(line.strip())

    pool = Pool(processes=args.cpus)
    pool.map(
        partial(
            process,
            args=args,
        ),
        need_to_do
    )
