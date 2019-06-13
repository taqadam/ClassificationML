import random
import shutil
from glob import glob
import os
from pathlib import Path
import argparse

random.seed(100)


def build_dataset(src_dir, n_samples=1000):
    os.makedirs(src_dir, exist_ok=True)
    filenames = glob(f"{src_dir}/*.jpg")
    filenames.sort()
    random.shuffle(filenames)
    samples = filenames[:n_samples]
    return samples


def copy_images(fps, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fp in fps:
        fname = Path(fp).name
        dp = f"{dest_dir}/{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("GrainClassification")
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()
    n = args.n
    ########################################
    # Grain Bins
    ########################################
    grain_bins = build_dataset("data/processed/grain_bin", n_samples=n)
    grain_bins_split_1 = int(0.8 * len(grain_bins))
    grain_bins_split_2 = int(0.9 * len(grain_bins))

    train_grain_bins = grain_bins[:grain_bins_split_1]
    os.makedirs("data/samples/train/grain_bin", exist_ok=True)
    for fp in train_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/train/grain_bin/grain_bin_{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()

    val_grain_bins = grain_bins[grain_bins_split_1:grain_bins_split_2]
    os.makedirs("data/samples/val/grain_bin", exist_ok=True)
    for fp in val_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/val/grain_bin/grain_bin_{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()

    test_grain_bins = grain_bins[grain_bins_split_2:]
    os.makedirs("data/samples/test/grain_bin", exist_ok=True)
    for fp in test_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/test/grain_bin/grain_bin_{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()

    ########################################
    # No Grain Bins
    ########################################
    no_grain_bins = build_dataset("data/processed/no_grain_bin", n_samples=n)
    no_grain_bins_split_1 = int(0.8 * len(no_grain_bins))
    no_grain_bins_split_2 = int(0.9 * len(no_grain_bins))

    train_no_grain_bins = no_grain_bins[:no_grain_bins_split_1]
    os.makedirs("data/samples/train/no_grain_bin", exist_ok=True)
    for fp in train_no_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/train/no_grain_bin/{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()

    val_no_grain_bins = no_grain_bins[no_grain_bins_split_1:no_grain_bins_split_2]
    os.makedirs("data/samples/val/no_grain_bin", exist_ok=True)
    for fp in val_no_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/val/no_grain_bin/{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()

    test_no_grain_bins = no_grain_bins[no_grain_bins_split_2:]
    os.makedirs("data/samples/test/no_grain_bin", exist_ok=True)
    for fp in test_no_grain_bins:
        fname = Path(fp).name
        dp = f"data/samples/test/no_grain_bin/{fname}"
        print(f"Saving {fp} to {dp}")
        shutil.copy(fp, dp)
    print()
