"""
This is to screen, or check, the training/validation data from portable hard disk.
The "portable hard disk" means the disk from AK, where has 158G image data.
But the data is not well classified. It's mixed together with different folder structure.
And it also includes multiple types including:
  NS: normal skin;
  BCC: Basal cell carcinoma;
  melanocytic;
  lentigo;
  ...
"""
import os
import re

import utils

log_fn = utils.log_info
root_dir = "D:/Supervisor/2023-08-11-Classification-Task/2023-07-31-dataset"
s_ns = 'normal skin'
s_bcc = 'bcc'
s_mel = 'melanocytic'
s_len = 'lentigo'
ns_filepath_list = []
bcc_filepath_list = []
mel_filepath_list = []
len_filepath_list = []
sbk_filepath_list = []
ns_avipath_list = []
bcc_avipath_list = []
mel_avipath_list = []
len_avipath_list = []
sbk_avipath_list = []
other_dir_list = []

counter_file_hash = {}
counter_avi_hash = {}
counter_dir_hash = {}
def counter_plus(key, file_counters, dir_counter=1):
    if key in counter_file_hash:
        counter_file_hash[key] += file_counters[0]
    else:
        counter_file_hash[key] = file_counters[0]
    if key in counter_avi_hash:
        counter_avi_hash[key] += file_counters[1]
    else:
        counter_avi_hash[key] = file_counters[1]
    if key in counter_dir_hash:
        counter_dir_hash[key] += dir_counter
    else:
        counter_dir_hash[key] = dir_counter

def list_all_files(dir_path, tif_file_list: [], avi_file_list: []):
    counter_tif = 0
    counter_avi = 0
    for dir_path, subdir_list, file_list in os.walk(dir_path):
        for fname in file_list:
            if fname.endswith('.tif'):
                counter_tif += 1
                tif_file_list.append(os.path.join(dir_path, fname))
            elif fname.endswith('.avi'):
                avi_file_list.append(os.path.join(dir_path, fname))
                counter_avi += 1
        # for
    # for
    return counter_tif, counter_avi

def handle_bcc_mel(root_path):
    """
    Handle dir having both BCC and Melanocytic images, such as dir:
    "1428 melanocytic nevus 6 lesions, 1 BCC"
    "1438 melanocytic nevus 7 lesions, BCC 1 lesion"
    """
    for dir_path, subdir_list, file_list in os.walk(root_path):
        base_dir = os.path.basename(dir_path)
        base_dir = base_dir.lower()
        if s_bcc in base_dir and s_mel not in base_dir:
            c = list_all_files(dir_path, bcc_filepath_list, bcc_avipath_list)
            counter_plus('bcc_mel.bcc', c, 1)
        elif s_mel in base_dir and s_bcc not in base_dir:
            c = list_all_files(dir_path, mel_filepath_list, mel_avipath_list)
            counter_plus('bcc_mel.mel', c, 1)
    # for

def handle_ns_bcc(root_path):
    for dir_path, subdir_list, file_list in os.walk(root_path):
        base_dir = os.path.basename(dir_path)
        base_dir = base_dir.lower()
        if s_ns in base_dir and s_bcc not in base_dir:
            c = list_all_files(dir_path, ns_filepath_list, ns_avipath_list)
            counter_plus('ns_bcc.ns', c, 1)
        elif s_bcc in base_dir and s_ns not in base_dir:
            c = list_all_files(dir_path, bcc_filepath_list, bcc_avipath_list)
            counter_plus('ns_bcc.bcc', c, 1)
    # for

def handle_lesions(root_path):
    for dir_path, subdir_list, file_list in os.walk(root_path):
        base_dir = os.path.basename(dir_path)
        base_dir = base_dir.lower()
        if s_ns in base_dir and s_bcc not in base_dir and s_mel not in base_dir:
            c = list_all_files(dir_path, ns_filepath_list, ns_avipath_list)
            counter_plus('lesions.ns', c, 1)
        elif s_bcc in base_dir and s_ns not in base_dir and s_mel not in base_dir:
            c = list_all_files(dir_path, bcc_filepath_list, bcc_avipath_list)
            counter_plus('lesions.bcc', c, 1)
        elif s_mel in base_dir and s_ns not in base_dir and s_bcc not in base_dir:
            c = list_all_files(dir_path, mel_filepath_list, mel_avipath_list)
            counter_plus('lesions.mel', c, 1)
    # for

def main():
    log_fn(f"root_dir: {root_dir}")
    item_list = os.listdir(root_dir)
    item_list.sort()
    log_fn(f"item count: {len(item_list)}")
    for i, item in enumerate(item_list):
        # log_fn(f"{i: 3d}: {item}")
        sub_path = os.path.join(root_dir, item)
        if not os.path.isdir(sub_path): continue
        sub_lo = item.lower()
        if s_ns in sub_lo and s_bcc not in sub_lo and s_mel not in sub_lo:      # NS only
            c = list_all_files(sub_path, ns_filepath_list, ns_avipath_list)
            counter_plus('ns', c, 1)
        elif s_bcc in sub_lo and s_ns not in sub_lo and s_mel not in sub_lo:    # BCC only
            c = list_all_files(sub_path, bcc_filepath_list, bcc_avipath_list)
            counter_plus('bcc', c, 1)
        elif s_mel in sub_lo and s_ns not in sub_lo and s_bcc not in sub_lo:    # Melanocytic only
            c = list_all_files(sub_path, mel_filepath_list, mel_avipath_list)
            counter_plus('mel', c, 1)
        elif s_len in sub_lo:                                                   # lentigo
            c = list_all_files(sub_path, len_filepath_list, len_avipath_list)
            counter_plus('lentigo', c, 1)
        elif 'seb k' in sub_lo:                                                 # "seb K"
            c = list_all_files(sub_path, sbk_filepath_list, sbk_avipath_list)
            counter_plus('sebK', c, 1)
        elif s_ns in sub_lo and s_bcc in sub_lo and s_mel not in sub_lo:        # NS and BCC
            handle_ns_bcc(sub_path)
        elif s_bcc in sub_lo and s_mel in sub_lo and s_ns not in sub_lo:        # BCC and Melanocytic
            handle_bcc_mel(sub_path)
        elif 'normalskin' in sub_lo:                                            # "normalskin"
            c = list_all_files(sub_path, ns_filepath_list, ns_avipath_list)
            counter_plus('normalskin', c, 1)
        elif 'normal' in sub_lo:                                                # "normal"
            c = list_all_files(sub_path, ns_filepath_list, ns_avipath_list)
            counter_plus('normal', c, 1)
        elif re.match(r"^\d{4} NI\d+ \d+ lesions$", item):                      # "1519 NI038 3 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} \d+ lesions$", item):                            # "1602 2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} {2}\d+ lesions$", item):                         # "1495  2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} NI\d+\w \d+ lesions$", item):                    # "1504 NI025a 5 lesions"
            handle_lesions(sub_path)
        else:
            other_dir_list.append(item)
    # for
    log_fn(f"other dir ******************************************")
    [log_fn(f"{p}") for p in other_dir_list]
    log_fn(f"dir_list: {len(item_list)}")
    keys = list(counter_dir_hash.keys())
    keys.sort()
    for key in keys:
        dc = counter_dir_hash[key]
        fc = counter_file_hash[key]
        ac = counter_avi_hash[key]
        log_fn(f"{key:12s}: {dc:3d}, {fc:5d}, {ac:3d}")
    log_fn(f"others dir: {len(other_dir_list):3d}")
    log_fn(f"ns_filepath_list  : {len(ns_filepath_list):5d} {len(ns_avipath_list):4d}")
    log_fn(f"bcc_filepath_list : {len(bcc_filepath_list):5d} {len(bcc_avipath_list):4d}")
    log_fn(f"mel_filepath_list : {len(mel_filepath_list):5d} {len(mel_avipath_list):4d}")
    log_fn(f"len_filepath_list : {len(len_filepath_list):5d} {len(len_avipath_list):4d}")
    log_fn(f"sbk_filepath_list : {len(sbk_filepath_list):5d} {len(sbk_avipath_list):4d}")


if __name__ == '__main__':
    main()
