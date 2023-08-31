import os
import random
import re
from utils import log_info, read_lines

s_ns = 'normal skin'
s_bcc = 'bcc'
s_mel = 'melanocytic'
s_len = 'lentigo'
s_ignore1 = 'Macroscopic Images'    # image too big to load
s_ignore2 = 'VivaBlock'             # image too big to load
s_ignore31 = '1640 melanocytic nevus 6 lesions'
s_ignore32 = '1640(2)'              # duplicated image

# ignore it because duplication.
#   "1419 normal skin central back/1419/central back"
#   "1419 R elbow and central back normal/1419/central back"
# the latter has other folders, so we ignore the former.
ign_patient_dir1 = "1419 normal skin central back"

basal_patient_dirs   = []
other_patient_dirs   = []
unknown_patient_dirs = []
basal_patient2leaf_dir_map = {} # key: patient dir. value: a map of leaf_dir => image_count
other_patient2leaf_dir_map = {}

# leaf to image-count map. Such maps are only for statistic
l2ic_bcc_map = {}   # BCC
l2ic_nsk_map = {}   # normal skin
l2ic_mel_map = {}   # melanocytic
l2ic_ltg_map = {}   # lentigo
l2ic_seb_map = {}   # seb K

def classify_dirs_by_patient(data_dir):
    def get_leaf_dir_map(dir_root: str):
        leaf_dir_map = {}
        for dir_path, subdir_list, f_list in os.walk(dir_root):
            if len(subdir_list) > 0: continue  # has subdir, then dir_path is not leaf dir.
            if s_ignore1 in dir_path: continue
            if s_ignore2 in dir_path: continue
            if s_ignore31 in dir_path and s_ignore32 in dir_path: continue
            tif_arr = [f for f in f_list if f.endswith('tif')]
            if len(tif_arr) == 0: continue
            leaf_dir_map[dir_path] = len(tif_arr)  # this is leaf dir.
        # for
        return leaf_dir_map

    def handle_bcc_specific(root_path):
        # Handle the folder 'BCC_23 July 2020/'
        for x in os.listdir(root_path):
            if x == '57 -  no images': continue
            full_path = os.path.join(root_path, x)
            if not os.path.isdir(full_path): continue
            basal_patient_dirs.append(full_path)
            mm = get_leaf_dir_map(full_path)
            basal_patient2leaf_dir_map[full_path] = mm
            l2ic_bcc_map.update(mm)

    def handle_ns_bcc(root_path):
        # Sample folder: '1450 normal skin, BCC/'
        ns_map, bcc_map = {}, {}
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir:
                ns_map.update(get_leaf_dir_map(dir_path))
            elif s_bcc in base_dir and s_ns not in base_dir:
                bcc_map.update(get_leaf_dir_map(dir_path))
        # for
        if ns_map:
            other_patient_dirs.append(root_path)
            other_patient2leaf_dir_map[root_path] = ns_map
            l2ic_nsk_map.update(ns_map)
        if bcc_map:
            basal_patient_dirs.append(root_path)
            basal_patient2leaf_dir_map[root_path] = bcc_map
            l2ic_bcc_map.update(bcc_map)

    def handle_bcc_mel(root_path):
        # Handle dir having both BCC and Melanocytic images, such as dir:
        #   "1428 melanocytic nevus 6 lesions, 1 BCC"
        #   "1438 melanocytic nevus 7 lesions, BCC 1 lesion"
        bcc_map, mel_map = {}, {}
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_bcc in base_dir and s_mel not in base_dir:
                bcc_map.update(get_leaf_dir_map(dir_path))
            elif s_mel in base_dir and s_bcc not in base_dir:
                mel_map.update(get_leaf_dir_map(dir_path))
        # for
        if bcc_map:
            basal_patient_dirs.append(root_path)
            basal_patient2leaf_dir_map[root_path] = bcc_map
            l2ic_bcc_map.update(bcc_map)
        if mel_map:
            other_patient_dirs.append(root_path)
            other_patient2leaf_dir_map[root_path] = mel_map
            l2ic_mel_map.update(mel_map)

    def handle_lesions(root_path):
        ns_map, bcc_map, mel_map = {}, {}, {}
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir and s_mel not in base_dir:
                ns_map.update(get_leaf_dir_map(dir_path))
            elif s_bcc in base_dir and s_ns not in base_dir and s_mel not in base_dir:
                bcc_map.update(get_leaf_dir_map(dir_path))
            elif s_mel in base_dir and s_ns not in base_dir and s_bcc not in base_dir:
                mel_map.update(get_leaf_dir_map(dir_path))
        # for
        # update the statistical maps
        if ns_map:
            l2ic_nsk_map.update(ns_map)
        if bcc_map:
            l2ic_bcc_map.update(bcc_map)
        if mel_map:
            l2ic_mel_map.update(mel_map)

        if bcc_map:
            basal_patient_dirs.append(root_path)
            basal_patient2leaf_dir_map[root_path] = bcc_map

        if ns_map or mel_map:
            # this is tricky. NS and melanocytic are both "other" type.
            # To avoid duplicate dir, we only add to one list
            other_patient_dirs.append(root_path)
            ns_map.update(mel_map)
            other_patient2leaf_dir_map[root_path] = ns_map

    log_info(f"classify_dirs_by_patient({data_dir})...")
    item_list = os.listdir(data_dir)
    item_list.sort()
    for i, item in enumerate(item_list):
        if item == ign_patient_dir1:
            log_info(f"Ignore patient dir: {ign_patient_dir1}")
            continue
        sub_path = os.path.join(data_dir, item)
        if not os.path.isdir(sub_path): continue
        item_lw = item.lower()
        if item_lw == 'BCC_23 July 2020'.lower():
            handle_bcc_specific(sub_path)
        elif s_ns in item_lw and s_bcc not in item_lw and s_mel not in item_lw: # NS only
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_nsk_map.update(m)
        elif s_bcc in item_lw and s_ns not in item_lw and s_mel not in item_lw: # BCC only
            basal_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            basal_patient2leaf_dir_map[sub_path] = m
            l2ic_bcc_map.update(m)
        elif s_mel in item_lw and s_ns not in item_lw and s_bcc not in item_lw: # Melanocytic only
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_mel_map.update(m)
        elif s_len in item_lw:                                                  # lentigo
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_ltg_map.update(m)
        elif 'seb k' in item_lw:                                                # "seb K"
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_seb_map.update(m)
        elif s_ns in item_lw and s_bcc in item_lw and s_mel not in item_lw:     # NS and BCC
            handle_ns_bcc(sub_path)
        elif s_bcc in item_lw and s_mel in item_lw and s_ns not in item_lw:     # BCC and Melanocytic
            handle_bcc_mel(sub_path)
        elif 'normalskin' in item_lw:                                           # "normalskin"
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_nsk_map.update(m)
        elif 'normal' in item_lw:                                               # "normal"
            other_patient_dirs.append(sub_path)
            m = get_leaf_dir_map(sub_path)
            other_patient2leaf_dir_map[sub_path] = m
            l2ic_nsk_map.update(m)
        elif re.match(r"^\d{4} NI\d+ \d+ lesions$", item):                      # "1519 NI038 3 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} \d+ lesions$", item):                            # "1602 2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} {2}\d+ lesions$", item):                         # "1495  2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} NI\d+\w \d+ lesions$", item):                    # "1504 NI025a 5 lesions"
            handle_lesions(sub_path)
        else:
            unknown_patient_dirs.append(sub_path)
    # for
    basal_patient_dirs.sort()
    other_patient_dirs.sort()
    unknown_patient_dirs.sort()
    log_info(f"  basal_patient_dirs  : {len(basal_patient_dirs)}")
    log_info(f"  other_patient_dirs  : {len(other_patient_dirs)}")
    log_info(f"  unknown_patient_dirs: {len(unknown_patient_dirs)}")
    log_info(f"classify_dirs_by_patient({data_dir})...Done")

def merge_leaf_dirs(patient_dirs: [], patient_map: {}):
    new_map = {}
    for pd in patient_dirs:
        leaf_map = patient_map[pd]
        new_map.update(leaf_map)
    # for
    return new_map

def save_dirs_by_patient(data_dir):
    log_info(f"save_dirs_by_patient({data_dir})...")
    log_info(f"  basal_patient_dirs  : {len(basal_patient_dirs)}")
    log_info(f"  other_patient_dirs  : {len(other_patient_dirs)}")
    log_info(f"  unknown_patient_dirs: {len(unknown_patient_dirs)}")
    log_info(f"  basal_patient2leaf_dir_map: {len(basal_patient2leaf_dir_map)}")
    log_info(f"  other_patient2leaf_dir_map: {len(other_patient2leaf_dir_map)}")

    f_path_arr = []
    prefix_len = len(data_dir)
    f_path = os.path.join(data_dir, f"load_by_patient_bcc_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in basal_patient_dirs]
    log_info(f"saved: {f_path}")
    f_path_arr.append(f_path)

    f_path = os.path.join(data_dir, f"load_by_patient_other_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in other_patient_dirs]
    log_info(f"saved: {f_path}")
    f_path_arr.append(f_path)

    f_path = os.path.join(data_dir, f"load_by_patient_unknown_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in unknown_patient_dirs]
    log_info(f"saved: {f_path}")
    f_path_arr.append(f_path)

    f_path = os.path.join(data_dir, f"load_by_seq_bcc_dirs.txt")
    leaf_map = merge_leaf_dirs(basal_patient_dirs, basal_patient2leaf_dir_map)
    with open(f_path, 'w') as fptr:
        keys = list(leaf_map.keys())
        keys.sort()
        [fptr.write(f"{key}\t{leaf_map[key]}\r\n") for key in keys]
    log_info(f"saved: {f_path}")
    f_path_arr.append(f_path)

    f_path = os.path.join(data_dir, f"load_by_seq_other_dirs.txt")
    leaf_map = merge_leaf_dirs(other_patient_dirs, other_patient2leaf_dir_map)
    with open(f_path, 'w') as fptr:
        keys = list(leaf_map.keys())
        keys.sort()
        [fptr.write(f"{key}\t{leaf_map[key]}\r\n") for key in keys]
    log_info(f"saved: {f_path}")
    f_path_arr.append(f_path)

    return f_path_arr

def shuffle_and_split():
    log_info(f"shuffle_and_split()...")
    basal_cnt, other_cnt = len(basal_patient_dirs), len(other_patient_dirs)
    random.shuffle(basal_patient_dirs)
    random.shuffle(other_patient_dirs)
    basal_test_cnt, other_test_cnt = int(basal_cnt/6), int(other_cnt/6)
    bcc4test_list    = basal_patient_dirs[:basal_test_cnt]
    other4test_list  = other_patient_dirs[:other_test_cnt]
    bcc4train_list   = basal_patient_dirs[basal_test_cnt:]
    other4train_list = other_patient_dirs[other_test_cnt:]
    log_info(f"Patient dir count:")
    log_info(f"  bcc_cnt    : {basal_cnt:5d}")
    log_info(f"  bcc train  : {len(bcc4train_list):5d}")
    log_info(f"  bcc test   : {len(bcc4test_list):5d}")
    log_info(f"  other_cnt  : {other_cnt:5d}")
    log_info(f"  other train: {len(other4train_list):5d}")
    log_info(f"  other test : {len(other4test_list):5d}")
    return bcc4train_list, bcc4test_list, other4train_list, other4test_list

def find_and_save_leaf_dirs(root_dir, bcc4train_list, bcc4test_list, other4train_list, other4test_list):
    log_info(f"find_and_save_leaf_dirs()...")
    bcc4train_map   = merge_leaf_dirs(bcc4train_list, basal_patient2leaf_dir_map)
    bcc4test_map    = merge_leaf_dirs(bcc4test_list, basal_patient2leaf_dir_map)
    other4train_map = merge_leaf_dirs(other4train_list, other_patient2leaf_dir_map)
    other4test_map  = merge_leaf_dirs(other4test_list, other_patient2leaf_dir_map)
    log_info(f"Leaf dir count:")
    log_info(f"  bcc train  : {len(bcc4train_map):5d}")
    log_info(f"  bcc test   : {len(bcc4test_map):5d}")
    log_info(f"  other train: {len(other4train_map):5d}")
    log_info(f"  other test : {len(other4test_map):5d}")

    prefix_len = len(root_dir)
    file_path_arr = []
    keys = list(bcc4train_map.keys())
    keys.sort()
    f_path = os.path.join(root_dir, f"dir_train_bcc.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f[prefix_len:]}\t{bcc4train_map[f]}\r\n") for f in keys]
    log_info(f"saved: {f_path}")
    file_path_arr.append(f_path)

    keys = list(other4train_map.keys())
    keys.sort()
    f_path = os.path.join(root_dir, f"dir_train_other.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f[prefix_len:]}\t{other4train_map[f]}\r\n") for f in keys]
    log_info(f"saved: {f_path}")
    file_path_arr.append(f_path)

    keys = list(bcc4test_map.keys())
    keys.sort()
    f_path = os.path.join(root_dir, f"dir_val_bcc.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f[prefix_len:]}\t{bcc4test_map[f]}\r\n") for f in keys]
    log_info(f"saved: {f_path}")
    file_path_arr.append(f_path)

    keys = list(other4test_map.keys())
    keys.sort()
    f_path = os.path.join(root_dir, f"dir_val_other.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f[prefix_len:]}\t{other4test_map[f]}\r\n") for f in keys]
    log_info(f"saved: {f_path}")
    file_path_arr.append(f_path)
    return file_path_arr

def check_file_lines_duplication(file_arr):
    log_info(f"check_dir_files()...")
    main_map = {} # file-name to line-set mapping
    for f_name in file_arr:
        lines = read_lines(f_name)
        log_info(f"  Read {len(lines):4d} lines for: {f_name}")
        main_map[f_name] = set(lines)
    # for
    dup_cnt = 0
    for f_name in file_arr:
        line_set = main_map[f_name]
        for fn, ls in main_map.items(): # each file-name and line-set
            if f_name == fn: continue
            log_info(f"  checking {f_name} vs {fn}")
            for line in line_set:
                if line in ls:
                    log_info(f"[Dup] line: \"{line}\". Found in both:")
                    log_info(f"  {f_name}")
                    log_info(f"  {fn}")
                    dup_cnt += 1
            # for
        # for
    # for
    log_info(f"check_dir_files()...Done. dup_cnt:{dup_cnt}.")
    return dup_cnt

def check_file_line_prefix_duplication(file_arr):
    log_info(f"check_file_line_prefix_duplication()...")
    main_map = {} # file-name to line-set mapping
    for f_name in file_arr:
        lines = read_lines(f_name)
        log_info(f"  Read {len(lines):4d} lines for: {f_name}")
        new_set = set()
        for line in lines:
            arr = line.split('/')
            if arr[1] == 'BCC_23 July 2020':
                s = "/".join(arr[:3])   # "dataset/BCC_23 July 2020/1010/L cheek/Confocal Images"
            else:
                s = "/".join(arr[:2])   # "dataset/1644 Scalp BCC/1644/scalp/VivaStack #5"
            new_set.add(s)
        main_map[f_name] = new_set
    # for
    dup_cnt = 0
    for f_name, line_set in main_map.items():
        for fn, ls in main_map.items(): # each file-name and line-set
            if f_name == fn: continue
            log_info(f"  checking {f_name} vs {fn}")
            for line in line_set:
                if line in ls:
                    log_info(f"[Dup] line: \"{line}\". Found in both:")
                    log_info(f"  {f_name}")
                    log_info(f"  {fn}")
                    dup_cnt += 1
            # for
        # for
    # for
    log_info(f"check_file_line_prefix_duplication()...Done. dup_cnt:{dup_cnt}")
    return dup_cnt

def save_statistic(l2ic_map, f_path):
    log_info(f"save_statistic({f_path})")
    keys = list(l2ic_map.keys())
    keys.sort()
    with open(f_path, 'w') as fptr:
        for key in keys:
            fptr.write(f"{key}\t{l2ic_map[key]}\r\n")
    # with

def main():
    hd_dir = './dataset/harddisk/'
    # classify dirs by patient
    root_dir = os.path.join(hd_dir, 'dataset')
    classify_dirs_by_patient(root_dir)

    save_dirs_by_patient(hd_dir)

    # shuffle patient dirs, and split for train & test
    # get the dir list for 4 types: train-bcc, train-other, test-bcc, test-other
    patient_dir_arr_arr = shuffle_and_split()

    # find and save leaf dirs. A leaf dir contains and only contains an image sequence.
    f_arr = find_and_save_leaf_dirs(hd_dir, *patient_dir_arr_arr)

    save_statistic(l2ic_bcc_map, os.path.join(hd_dir, f"load_stat_bcc_dirs.txt"))
    save_statistic(l2ic_nsk_map, os.path.join(hd_dir, f"load_stat_ns_dirs.txt"))
    save_statistic(l2ic_mel_map, os.path.join(hd_dir, f"load_stat_melanocytic_dirs.txt"))
    save_statistic(l2ic_ltg_map, os.path.join(hd_dir, f"load_stat_lentigo_dirs.txt"))
    save_statistic(l2ic_seb_map, os.path.join(hd_dir, f"load_stat_sebK_dirs.txt"))

    # check duplication
    bcc4train, other4train, bcc4test, other4test = f_arr
    dup_cnt_a1 = check_file_lines_duplication([bcc4train, bcc4test])
    dup_cnt_a2 = check_file_lines_duplication([other4train, other4test])

    dup_cnt_b1 = check_file_line_prefix_duplication([bcc4train, bcc4test])
    dup_cnt_b2 = check_file_line_prefix_duplication([other4train, other4test])

    log_info(f"check_file_lines_duplication() Final dup_cnt:", dup_cnt_a1, dup_cnt_a2)
    log_info(f"check_file_line_prefix_duplication() Final dup_cnt:", dup_cnt_b1, dup_cnt_b2)

if __name__ == '__main__':
    main()
