"""
duplicated image checking.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.utils.data as tu_data
import torchvision.transforms as transforms
from utils import log_info, read_lines, get_time_ttl_and_eta
from tifffile import tifffile

class DatasetDupCheck(tu_data.Dataset):

    def __init__(self, bcc_list, ns_list, image_transform=None, log_fn=log_info):
        self.image_transform = image_transform
        self.log_fn = log_fn
        self.ipath_list = []  # image path list
        self.ipath_list.extend(bcc_list)
        self.ipath_list.sort()
        self.log_fn(f"Dataset")
        self.log_fn(f"  Found {len(self.ipath_list)} images for BCC.")
        self.log_fn(f"    first BCC image: {self.ipath_list[0]}") if self.ipath_list else None
        self.log_fn(f"    last  BCC image: {self.ipath_list[-1]}") if self.ipath_list else None

        self.ipath_list.extend(ns_list)
        self.log_fn(f"  Found {len(ns_list)} images for Non-BCC")
        self.log_fn(f"    first NS image: {ns_list[0]}") if ns_list else None
        self.log_fn(f"    last  NS image: {ns_list[-1]}") if ns_list else None
        self.ipath_list.sort()
        self.np2pil_fn = transforms.ToPILImage() # numpy nparray to PIL image

    def __len__(self):
        return len(self.ipath_list)

    def __getitem__(self, index):
        # im = Image.open(self.ipath_list[index]).convert("RGB")
        # Notes. the input *.tif file has mode "I;16", which means it uses 16 bits for single value.
        img = tifffile.imread(self.ipath_list[index])
        img = img / 256             # 16 bits to 8 bits. result: float64
        img = img.astype(np.uint8)  # float64 to uint8
        img = self.np2pil_fn(img)   # nparray to PIL image
        if self.image_transform:
            img = self.image_transform(img)
        return img, self.ipath_list[index]
# class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--scale", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--image_size", nargs="+", type=int, default=[224, 224])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./dataset/harddisk")
    _args = parser.parse_args()

    # add device
    gpu_ids = _args.gpu_ids
    log_info(f"gpu_ids : {gpu_ids}")
    _args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    return _args

def get_data_loaders(args):
    train_tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])

    # from dir list to file list. Each dir is a leaf dir, containing *.tif images
    def dirs2files(dir_list):
        f_list = []
        for d_idx, d in enumerate(dir_list):
            dd = os.path.join(args.data_dir, d)
            f_names = os.listdir(dd)
            f_names = [f for f in f_names if str(f).endswith(".tif")]
            [f_list.append(os.path.join(dd, f)) for f in f_names]
            # log_info(f"  {d_idx:4d}/{d_cnt}: {d}\t{len(f_names)}")
        f_list.sort()
        return f_list

    # get file paths. The file contains dir.
    file_train_bcc   = os.path.join(args.data_dir, "dir_train_bcc.txt")
    file_train_other = os.path.join(args.data_dir, "dir_train_other.txt")
    file_test_bcc    = os.path.join(args.data_dir, "dir_val_bcc.txt")
    file_test_other  = os.path.join(args.data_dir, "dir_val_other.txt")

    # get dir list from file
    train_bcc_dir_list   = read_lines(file_train_bcc)
    train_other_dir_list = read_lines(file_train_other)
    test_bcc_dir_list    = read_lines(file_test_bcc)
    test_other_dir_list  = read_lines(file_test_other)

    # get file list from dir list
    train_bcc_list   = dirs2files(train_bcc_dir_list)
    train_other_list = dirs2files(train_other_dir_list)
    test_bcc_list    = dirs2files(test_bcc_dir_list)
    test_other_list  = dirs2files(test_other_dir_list)
    train_dataset = DatasetDupCheck(train_bcc_list, train_other_list, image_transform=train_tf)
    test_dataset  = DatasetDupCheck(test_bcc_list, test_other_list, image_transform=train_tf)

    train_loader = tu_data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    log_info(f"train dataset and data loader ==================")
    log_info(f"  bcc index      : {file_train_bcc}")
    log_info(f"  bcc dir cnt    : {len(train_bcc_dir_list)}")
    log_info(f"  bcc file cnt   : {len(train_bcc_list)}")
    log_info(f"  other index    : {file_train_other}")
    log_info(f"  other dir cnt  : {len(train_other_dir_list)}")
    log_info(f"  other file cnt : {len(train_other_list)}")
    log_info(f"  total count    : {len(train_dataset)}")
    log_info(f"  batch_count    : {len(train_loader)}")
    log_info(f"  batch_size     : {args.batch_size}")
    log_info(f"  num_workers    : {args.num_workers}")

    test_loader = tu_data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    log_info(f"test dataset and data loader ==================")
    log_info(f"  bcc index      : {file_test_bcc}")
    log_info(f"  bcc dir cnt    : {len(test_bcc_dir_list)}")
    log_info(f"  bcc file cnt   : {len(test_bcc_list)}")
    log_info(f"  other index    : {file_test_other}")
    log_info(f"  other dir cnt  : {len(test_other_dir_list)}")
    log_info(f"  other file cnt : {len(test_other_list)}")
    log_info(f"  total count    : {len(test_dataset)}")
    log_info(f"  batch_count    : {len(test_loader)}")
    log_info(f"  batch_size     : {args.batch_size}")
    log_info(f"  num_workers    : {args.num_workers}")

    return train_loader, test_loader

def check_dup(train_img_arr, train_path_arr, test_img_arr, test_path_arr):
    train_cnt = train_img_arr.shape[0]
    test_cnt = test_img_arr.shape[0]
    dup_cnt = 0
    for tr_idx in range(train_cnt):
        for te_idx in range(test_cnt):
            if torch.all(torch.eq(train_img_arr[tr_idx], test_img_arr[te_idx])):
                dup_cnt += 1
                log_info(f"Found dup: {train_path_arr[tr_idx]}\t{test_path_arr[te_idx]}")
        # for
    # for
    return dup_cnt

def main():
    args = parse_args()
    train_loader, test_loader = get_data_loaders(args)
    b_cnt = len(train_loader)
    dup_cnt = 0
    scale = args.scale
    start_time = time.time()
    log_info(f"batch size : {args.batch_size}")
    log_info(f"train b_cnt: {b_cnt}")
    log_info(f"test b_cnt : {len(test_loader)}")
    log_info(f"scale      : {scale}")
    log_info(f"start duplication checking . . .")
    for b_idx, (train_img_arr, train_path_arr) in enumerate(train_loader):
        elp, eta = get_time_ttl_and_eta(start_time, b_idx, b_cnt)
        log_info(f"train: B{b_idx:04d}/{b_cnt} ----- dup_cnt={dup_cnt}. elp:{elp}, eta:{eta}")
        log_info(f"  file[0] : {train_path_arr[0]}")
        log_info(f"  file[-1]: {train_path_arr[-1]}")
        train_img_arr = train_img_arr.to(args.device)
        train_img_arr = (train_img_arr * scale).int()
        for idx, (test_img_arr, test_path_arr) in enumerate(test_loader):
            test_img_arr = test_img_arr.to(args.device)
            test_img_arr = (test_img_arr * scale).int()
            d = check_dup(train_img_arr, train_path_arr, test_img_arr, test_path_arr)
            dup_cnt += d
        # for
    # for
    log_info(f"dup_cnt: {dup_cnt}")

if __name__ == '__main__':
    main()
