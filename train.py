import os
import numpy as np
import pandas as pd
import random
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/train", help="Path to the training data")

    return parser.parse_args()

def main():
    args = parse_arguments()
    IMAGE_DIR = os.path.join(args.train_dir, "train_images")
    TRAIN_EXCEL_FILE = os.path.join(args.train_dir, "train_labels.xlsx")
    df = pd.read_excel(TRAIN_EXCEL_FILE)

    stem_to_file = {}
    for f in os.listdir(IMAGE_DIR):
        stem, _ = os.path.splitext(f)
        stem_to_file[stem] = f

    stems = df['Image_id'].astype(str).str.zfill(3)
    df['image_path'] = stems.map(stem_to_file)

    print(df.head())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trying with multiple seeds - to measure how stable our model is with different random initializations, data shuffling, and train test splits
    seeds = [7, 10, 42, 56, 100]

    for SEED in seeds:

        # setting seed for reproducibility of the results
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # print(IMAGE_DIR, TRAIN_EXCEL_FILE)


if __name__ == '__main__':
    main()