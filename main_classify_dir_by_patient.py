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

bcc_patient_dirs = []   # BCC
ns_patient_dirs  = []   # normal skin
mel_patient_dirs = []   # melanocytic
ltg_patient_dirs = []   # lentigo
seb_patient_dirs = []   # seb K
unknown_dirs     = []   # unknown

# ignore it because duplication.
#   "1419 normal skin central back/1419/central back"
#   "1419 R elbow and central back normal/1419/central back"
# the latter has other folders, so we ignore the former.
ign_patient_dir1 = "1419 normal skin central back"

def classify_dirs_by_patient(data_dir):
    def handle_bcc_specific(root_path):
        """Handle the folder 'BCC_23 July 2020' """
        for x in os.listdir(root_path):
            if x == '57 -  no images': continue
            full_path = os.path.join(root_path, x)
            if not os.path.isdir(full_path): continue
            bcc_patient_dirs.append(full_path)

    def handle_ns_bcc(root_path):
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir:
                ns_patient_dirs.append(dir_path)
            elif s_bcc in base_dir and s_ns not in base_dir:
                bcc_patient_dirs.append(dir_path)
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
                bcc_patient_dirs.append(dir_path)
            elif s_mel in base_dir and s_bcc not in base_dir:
                mel_patient_dirs.append(dir_path)
        # for

    def handle_lesions(root_path):
        for dir_path, subdir_list, _ in os.walk(root_path):
            base_dir = os.path.basename(dir_path)
            base_dir = base_dir.lower()
            if s_ns in base_dir and s_bcc not in base_dir and s_mel not in base_dir:
                ns_patient_dirs.append(dir_path)
            elif s_bcc in base_dir and s_ns not in base_dir and s_mel not in base_dir:
                bcc_patient_dirs.append(dir_path)
            elif s_mel in base_dir and s_ns not in base_dir and s_bcc not in base_dir:
                mel_patient_dirs.append(dir_path)
        # for

    log_info(f"classify_dirs_by_patient({data_dir})...")
    item_list = os.listdir(data_dir)
    item_list.sort()
    for i, item in enumerate(item_list):
        # log_fn(f"{i: 3d}: {item}")
        if item == ign_patient_dir1:
            log_info(f"Ignore patient dir: {ign_patient_dir1}")
            continue
        sub_path = os.path.join(data_dir, item)
        if not os.path.isdir(sub_path): continue
        sub_lw = item.lower()
        if sub_lw == 'BCC_23 July 2020'.lower():
            handle_bcc_specific(sub_path)
        elif s_ns in sub_lw and s_bcc not in sub_lw and s_mel not in sub_lw:    # NS only
            ns_patient_dirs.append(sub_path)
        elif s_bcc in sub_lw and s_ns not in sub_lw and s_mel not in sub_lw:    # BCC only
            bcc_patient_dirs.append(sub_path)
        elif s_mel in sub_lw and s_ns not in sub_lw and s_bcc not in sub_lw:    # Melanocytic only
            mel_patient_dirs.append(sub_path)
        elif s_len in sub_lw:                                                   # lentigo
            ltg_patient_dirs.append(sub_path)
        elif 'seb k' in sub_lw:                                                 # "seb K"
            seb_patient_dirs.append(sub_path)
        elif s_ns in sub_lw and s_bcc in sub_lw and s_mel not in sub_lw:        # NS and BCC
            handle_ns_bcc(sub_path)
        elif s_bcc in sub_lw and s_mel in sub_lw and s_ns not in sub_lw:        # BCC and Melanocytic
            handle_bcc_mel(sub_path)
        elif 'normalskin' in sub_lw:                                            # "normalskin"
            ns_patient_dirs.append(sub_path)
        elif 'normal' in sub_lw:                                                # "normal"
            ns_patient_dirs.append(sub_path)
        elif re.match(r"^\d{4} NI\d+ \d+ lesions$", item):                      # "1519 NI038 3 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} \d+ lesions$", item):                            # "1602 2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} {2}\d+ lesions$", item):                         # "1495  2 lesions"
            handle_lesions(sub_path)
        elif re.match(r"^\d{4} NI\d+\w \d+ lesions$", item):                    # "1504 NI025a 5 lesions"
            handle_lesions(sub_path)
        else:
            unknown_dirs.append(sub_path)

def save_dirs_by_patient(data_dir):
    log_info(f"save_dirs_by_patient({data_dir})...")
    log_info(f"  bcc_patient_dirs   : {len(bcc_patient_dirs)}")
    log_info(f"  ns_patient_dirs    : {len(ns_patient_dirs)}")
    log_info(f"  mel_patient_dirs   : {len(mel_patient_dirs)}")
    log_info(f"  ltg_patient_dirs   : {len(ltg_patient_dirs)}")
    log_info(f"  seb_patient_dirs   : {len(seb_patient_dirs)}")
    log_info(f"  unknown_dirs       : {len(unknown_dirs)}")

    prefix_len = len(data_dir)
    f_path = os.path.join(data_dir, f"load_by_patient_bcc_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in bcc_patient_dirs]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_by_patient_ns_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in ns_patient_dirs]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_by_patient_melanocytic_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in mel_patient_dirs]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_by_patient_lentigo_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in ltg_patient_dirs]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_by_patient_sebK_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in seb_patient_dirs]
    log_info(f"saved: {f_path}")

    f_path = os.path.join(data_dir, f"load_by_patient_unknown_dirs.txt")
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{fp[prefix_len:]}\r\n") for fp in unknown_dirs]
    log_info(f"saved: {f_path}")

def shuffle_and_split(bcc_dirs, other_dirs):
    log_info(f"shuffle_and_split()...")
    bcc_cnt, other_cnt = len(bcc_dirs), len(other_dirs)
    random.shuffle(bcc_dirs)
    random.shuffle(other_dirs)
    bcc_test_cnt, other_test_cnt = int(bcc_cnt/6), int(other_cnt/6)
    bcc4test_list, bcc4train_list = bcc_dirs[:bcc_test_cnt], bcc_dirs[bcc_test_cnt:]
    other4test_list, other4train_list = other_dirs[:other_test_cnt], other_dirs[other_test_cnt:]
    log_info(f"Patient dir count:")
    log_info(f"  bcc_cnt    : {bcc_cnt:5d}")
    log_info(f"  bcc train  : {len(bcc4train_list):5d}")
    log_info(f"  bcc test   : {len(bcc4test_list):5d}")
    log_info(f"  other_cnt  : {other_cnt:5d}")
    log_info(f"  other train: {len(other4train_list):5d}")
    log_info(f"  other test : {len(other4test_list):5d}")
    return bcc4train_list, bcc4test_list, other4train_list, other4test_list

def get_leaf_dir_map_from_dir_list(dir_list):
    def append_all_leaf_dir(dir_root: str):
        for dir_path, subdir_list, f_list in os.walk(dir_root):
            if len(subdir_list) > 0: continue  # has subdir, then dir_path is not leaf dir.
            if s_ignore1 in dir_path: continue
            if s_ignore2 in dir_path: continue
            if s_ignore31 in dir_path and s_ignore32 in dir_path: continue
            tif_arr = [f for f in f_list if f.endswith('tif')]
            if len(tif_arr) == 0: continue
            leaf_dir_map[dir_path] = len(tif_arr)  # this is leaf dir.
        # for

    leaf_dir_map = {}
    for d in dir_list:
        append_all_leaf_dir(d)
    # for
    return leaf_dir_map

def find_leaf_dirs(root_dir, bcc4train_list, bcc4test_list, other4train_list, other4test_list):
    log_info(f"find_leaf_dirs()...")
    bcc4train_map   = get_leaf_dir_map_from_dir_list(bcc4train_list)
    bcc4test_map    = get_leaf_dir_map_from_dir_list(bcc4test_list)
    other4train_map = get_leaf_dir_map_from_dir_list(other4train_list)
    other4test_map  = get_leaf_dir_map_from_dir_list(other4test_list)
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

def load_by_sequence(patient_dirs, root_dir, out_file_name):
    log_info(f"load_by_sequence({out_file_name})...")
    leaf_dir_map = get_leaf_dir_map_from_dir_list(patient_dirs)
    log_info(f"  leaf_dir count: {len(leaf_dir_map)}")

    prefix_len = len(root_dir)
    keys = list(leaf_dir_map.keys())
    keys.sort()
    f_path = os.path.join(root_dir, out_file_name)
    with open(f_path, 'w') as fptr:
        [fptr.write(f"{f[prefix_len:]}\t{leaf_dir_map[f]}\r\n") for f in keys]
    log_info(f"saved: {f_path}")

def main():
    hd_dir = './dataset/harddisk/'
    # classify dirs by patient
    root_dir = os.path.join(hd_dir, 'dataset')
    classify_dirs_by_patient(root_dir)

    # save dirs by patient
    save_dirs_by_patient(hd_dir)

    # shuffle and split dirs
    other_patient_dirs = ns_patient_dirs + mel_patient_dirs + ltg_patient_dirs + seb_patient_dirs
    bcc4train, bcc4test, other4train, other4test = shuffle_and_split(bcc_patient_dirs, other_patient_dirs)

    # find leaf dirs
    file_arr = find_leaf_dirs(hd_dir, bcc4train, bcc4test, other4train, other4test)

    load_by_sequence(bcc_patient_dirs, hd_dir, 'load_by_seq_bcc_dirs.txt')
    load_by_sequence(ns_patient_dirs, hd_dir, 'load_by_seq_ns_dirs.txt')
    load_by_sequence(mel_patient_dirs, hd_dir, 'load_by_seq_melanocytic_dirs.txt')
    load_by_sequence(ltg_patient_dirs, hd_dir, 'load_by_seq_lentigo_dirs.txt')
    load_by_sequence(seb_patient_dirs, hd_dir, 'load_by_seq_sebK_dirs.txt')

    # check duplication for leaf dirs
    check_file_lines_duplication(file_arr)

if __name__ == '__main__':
    main()
