import os
import random
import re
import tifffile
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import log_info, read_lines


class DatasetHarddisk(data.Dataset):

    def __init__(self, bcc_list, ns_list, image_transform=None, target_transform=None, log_fn=log_info):
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.log_fn = log_fn
        self.ipath_list = []  # image path list
        self.label_list = []  # image label list
        self.ipath_list.extend(bcc_list)
        self.label_list.extend([1] * len(bcc_list)) # BCC image has label 1
        self.ipath_list.sort()
        self.log_fn(f"Dataset")
        self.log_fn(f"  Found {len(self.ipath_list)} images for BCC. Assign label 1.")
        self.log_fn(f"    first BCC image: {self.ipath_list[0]}") if self.ipath_list else None
        self.log_fn(f"    last  BCC image: {self.ipath_list[-1]}") if self.ipath_list else None

        self.ipath_list.extend(ns_list)
        self.label_list.extend([0] * len(ns_list))
        self.log_fn(f"  Found {len(ns_list)} images. Assign label 0.")
        self.log_fn(f"    first NS image: {ns_list[0]}") if ns_list else None
        self.log_fn(f"    last  NS image: {ns_list[-1]}") if ns_list else None
        self.log_fn(f"  total images: {len(self.ipath_list)}, labels: {len(self.label_list)}")
        if len(self.ipath_list) != len(self.label_list):
            raise ValueError(f"label count not match image count!")
        self.np2pil_fn = transforms.ToPILImage()  # numpy nparray to PIL image

    def __len__(self):
        return len(self.ipath_list)

    def __getitem__(self, index):
        # img = Image.open(self.ipath_list[index]).convert("RGB")
        # Notes. the input *.tif file has mode "I;16", which means it uses 16 bits for single value.
        img = tifffile.imread(self.ipath_list[index])
        img = img / 256             # 16 bits to 8 bits. result: float64
        img = img.astype(np.uint8)  # float64 to uint8
        img = self.np2pil_fn(img)   # nparray to PIL image
        img = img.convert("RGB")
        lbl = self.label_list[index]
        if self.image_transform:
            img = self.image_transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl
# class

s_ns = 'normal skin'
s_bcc = 'bcc'
s_mel = 'melanocytic'
s_len = 'lentigo'

ignore_str1 = 'Macroscopic Images'
ignore_str2 = 'VivaBlock'
ignore_str1_cnt = 0
ignore_str2_cnt = 0

def categorize_dirs(data_dir: str, bcc_dir_list: [], other_dir_list: [], unknown_dir_list: []):
    def append_all_leaf_dir(dir_root, dir_list: [], stat_map: {}):
        global ignore_str1_cnt, ignore_str2_cnt
        dir_counter = 0
        for dir_path, subdir_list, f_list in os.walk(dir_root):
            if len(subdir_list) > 0: continue   # has subdir, then dir_path is not leaf dir.
            tif_cnt = 0
            for f in f_list:
                if f.endswith('tif'): tif_cnt += 1
            # for
            if tif_cnt == 0: continue
            if ignore_str1 in dir_path:
                ignore_str1_cnt += 1
                continue
            if ignore_str2 in dir_path:
                ignore_str2_cnt += 1
                continue
            dir_counter += 1
            dir_list.append(dir_path)  # this is leaf dir.
            stat_map[dir_path] = tif_cnt
        # for
        return dir_counter

    def handle_ns_bcc(root_path):
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir:
                append_all_leaf_dir(dir_path, other_dir_list, m_ns)
            elif s_bcc in base_dir and s_ns not in base_dir:
                append_all_leaf_dir(dir_path, bcc_dir_list, m_bcc)
        # for

    def handle_bcc_mel(root_path):
        """
        Handle dir having both BCC and Melanocytic images, such as dir:
        "1428 melanocytic nevus 6 lesions, 1 BCC"
        "1438 melanocytic nevus 7 lesions, BCC 1 lesion"
        """
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_bcc in base_dir and s_mel not in base_dir:
                append_all_leaf_dir(dir_path, bcc_dir_list, m_bcc)
            elif s_mel in base_dir and s_bcc not in base_dir:
                append_all_leaf_dir(dir_path, other_dir_list, m_mel)
        # for

    def handle_lesions(root_path):
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir and s_mel not in base_dir:
                append_all_leaf_dir(dir_path, other_dir_list, m_ns)
            elif s_bcc in base_dir and s_ns not in base_dir and s_mel not in base_dir:
                append_all_leaf_dir(dir_path, bcc_dir_list, m_bcc)
            elif s_mel in base_dir and s_ns not in base_dir and s_bcc not in base_dir:
                append_all_leaf_dir(dir_path, other_dir_list, m_mel)
        # for

    item_list = os.listdir(data_dir)
    item_list.sort()
    m_ns, m_bcc, m_mel, m_ltg, m_seb = {}, {}, {}, {}, {}
    for i, item in enumerate(item_list):
        # log_fn(f"{i: 3d}: {item}")
        sub_path = os.path.join(data_dir, item)
        if not os.path.isdir(sub_path): continue
        sub_lo = item.lower()
        if s_ns in sub_lo and s_bcc not in sub_lo and s_mel not in sub_lo:      # NS only
            append_all_leaf_dir(sub_path, other_dir_list, m_ns)
        elif s_bcc in sub_lo and s_ns not in sub_lo and s_mel not in sub_lo:    # BCC only
            append_all_leaf_dir(sub_path, bcc_dir_list, m_bcc)
        elif s_mel in sub_lo and s_ns not in sub_lo and s_bcc not in sub_lo:    # Melanocytic only
            append_all_leaf_dir(sub_path, other_dir_list, m_mel)
        elif s_len in sub_lo:                                                   # lentigo
            append_all_leaf_dir(sub_path, other_dir_list, m_ltg)
        elif 'seb k' in sub_lo:                                                 # "seb K"
            append_all_leaf_dir(sub_path, other_dir_list, m_seb)
        elif s_ns in sub_lo and s_bcc in sub_lo and s_mel not in sub_lo:        # NS and BCC
            handle_ns_bcc(sub_path)
        elif s_bcc in sub_lo and s_mel in sub_lo and s_ns not in sub_lo:        # BCC and Melanocytic
            handle_bcc_mel(sub_path)
        elif 'normalskin' in sub_lo:                                            # "normalskin"
            append_all_leaf_dir(sub_path, other_dir_list, m_ns)
        elif 'normal' in sub_lo:                                                # "normal"
            append_all_leaf_dir(sub_path, other_dir_list, m_ns)
        elif re.match(r"^\d{4} NI\d+ \d+ lesions$", item):                      # "1519 NI038 3 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} \d+ lesions$", item):                            # "1602 2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} {2}\d+ lesions$", item):                         # "1495  2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} NI\d+\w \d+ lesions$", item):                    # "1504 NI025a 5 lesions"
            handle_lesions(sub_path)
        else:
            unknown_dir_list.append(sub_path)
    # for
    map_of_stat_map = {
        'ns'         : m_ns,
        'bcc'        : m_bcc,
        'melanocytic': m_mel,
        'lentigo'    : m_ltg,
        'sebK'       : m_seb,
    }
    return map_of_stat_map

def dir_load(data_dir):
    log_info(f"dir_load({data_dir})...")
    root_dir = os.path.join(data_dir, 'dataset')
    bcc_dir_list, other_dir_list, unknown_dir_list = [], [], []
    map_of_stat_map = categorize_dirs(root_dir, bcc_dir_list, other_dir_list, unknown_dir_list)
    pre_len = len(data_dir)
    bcc_dir_list     = [d[pre_len:] for d in bcc_dir_list]
    other_dir_list   = [d[pre_len:] for d in other_dir_list]
    unknown_dir_list = [d[pre_len:] for d in unknown_dir_list]
    log_info(f"loaded dir from: {root_dir}")
    log_info(f"  bcc_dir_list    : {len(bcc_dir_list)}")
    log_info(f"  other_dir_list  : {len(other_dir_list)}")
    log_info(f"  unknown_dir_list: {len(unknown_dir_list)}")

    f_path = os.path.join(data_dir, f"load_bcc_dir_list.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp}\r\n") for fp in bcc_dir_list]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_other_dir_list.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp}\r\n") for fp in other_dir_list]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_unknown_dir_list.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp}\r\n") for fp in unknown_dir_list]
    log_info(f"saved: {f_path}")

    for type_name, stat_map in map_of_stat_map.items():
        keys = list(stat_map.keys())
        keys.sort()
        f_path = os.path.join(data_dir, f"type_{type_name}.txt")
        with open(f_path, 'w') as fptr:
            for k, v in stat_map.items():
                fptr.write(f"{k}\t{v}\n")
        # with
        log_info(f"saved detail file: {f_path}")
    # for
    return bcc_dir_list, other_dir_list

def dir_reshuffle(root_dir, bcc_dir_list, other_dir_list):
    log_info(f"dir_reshuffle()...")
    bcc_cnt, other_cnt = len(bcc_dir_list), len(other_dir_list)
    random.shuffle(bcc_dir_list)
    random.shuffle(other_dir_list)
    bcc_test_cnt, other_test_cnt = int(bcc_cnt/6), int(other_cnt/6)
    bcc4test_list, bcc4train_list = bcc_dir_list[:bcc_test_cnt], bcc_dir_list[bcc_test_cnt:]
    other4test_list, other4train_list = other_dir_list[:other_test_cnt], other_dir_list[other_test_cnt:]
    log_info(f"  bcc_cnt    : {bcc_cnt:5d}")
    log_info(f"  bcc train  : {len(bcc4train_list):5d}")
    log_info(f"  bcc test   : {len(bcc4test_list):5d}")
    log_info(f"  other_cnt  : {other_cnt:5d}")
    log_info(f"  other train: {len(other4train_list):5d}")
    log_info(f"  other test : {len(other4test_list):5d}")

    bcc4train_list.sort()
    bcc4test_list.sort()
    other4train_list.sort()
    other4test_list.sort()
    file_arr = []
    f_path = os.path.join(root_dir, f"dir_train_bcc.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f}\r\n") for f in bcc4train_list]
    log_info(f"saved: {f_path}")
    file_arr.append(f_path)

    f_path = os.path.join(root_dir, f"dir_train_other.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f}\r\n") for f in other4train_list]
    log_info(f"saved: {f_path}")
    file_arr.append(f_path)

    f_path = os.path.join(root_dir, f"dir_val_bcc.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f}\r\n") for f in bcc4test_list]
    log_info(f"saved: {f_path}")
    file_arr.append(f_path)

    f_path = os.path.join(root_dir, f"dir_val_other.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f}\r\n") for f in other4test_list]
    log_info(f"saved: {f_path}")
    file_arr.append(f_path)
    return file_arr

def check_dir_files(file_arr):
    log_info(f"check_dir_files()...")
    main_map = {} # file-name to line-set mapping
    for f_name in file_arr:
        lines = read_lines(f_name)
        log_info(f"  Read {len(lines):4d} lines for: {f_name}")
        main_map[f_name] = set(lines)
    # for
    for f_name in file_arr:
        line_set = main_map[f_name]
        for fn, ls in main_map.items(): # each file-name and line-set
            if f_name == fn: continue
            log_info(f"  checking {f_name} vs {fn}")
            for line in line_set:
                if line in ls:
                    log_info(f"[Error] duplicate line: \"{line}\". Found in both:")
                    log_info(f"  {f_name}")
                    log_info(f"  {fn}")
                    return False
            # for
        # for
    # for
    log_info(f"check_dir_files()...Done. No duplicate line found.")
    return True

def main():
    hd_dir = './dataset/harddisk/'
    bcc_dir_list, other_dir_list = dir_load(hd_dir)
    file_arr = dir_reshuffle(hd_dir, bcc_dir_list, other_dir_list)
    check_dir_files(file_arr)
    log_info(f"Ignore some directories who containing specific keywords:")
    log_info(f"  {ignore_str1}: {ignore_str1_cnt}")
    log_info(f"  {ignore_str2}: {ignore_str2_cnt}")

if __name__ == '__main__':
    main()
