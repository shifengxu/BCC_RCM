"""
Evaluate sequence of hard-disk images.
Here, sequence means all the images in a specific folder, such as "BCC" folder. In the BCC folder,
there are RCM images of BCC patients. And we suppose some or all of the images are BCC-images.
BCC: Basal Cell Carcinoma
RCM: Reflectance Confocal Microscopy
"""
import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as tu_data
from torch.backends import cudnn

import models
from dataset_harddisk import DatasetHarddisk
from utils import log_info, count_parameters, read_lines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[3])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--image_size", nargs="+", type=int, default=[224, 224])
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--ckpt_load_path", type=str, default="./checkpoint/model_Epoch080.ckpt")
    parser.add_argument("--data_dir", type=str, default="./dataset/harddisk")
    _args = parser.parse_args()

    # add device
    gpu_ids = _args.gpu_ids
    log_info(f"gpu_ids : {gpu_ids}")
    _args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    cudnn.benchmark = True
    return _args

def load_model():
    ckpt_path = args.ckpt_load_path
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Invalid file path: {ckpt_path}")
    log_info(f"load model from: {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=args.device)
    m_type = state_dict['resnet']
    log_info(f"get m_type from state_dict: {m_type}")
    if m_type in ['resnet101', '101']:
        model = models.resnet101(weights=None, num_classes=2)
    elif m_type in ['resnet152', '152']:
        model = models.resnet152(weights=None, num_classes=2)
    else:
        raise ValueError(f"Invalid resnet from state_dict: {m_type}")
    m, ep, lr, ta = state_dict['model'], state_dict['epoch'], state_dict['lr'], state_dict['train_accu']
    model.load_state_dict(m)
    log_info(f"load model from: {ckpt_path}...Done. epoch:{ep}, lr:{lr:.8f}, train_accu:{ta:.4f}")

    counter, str_size = count_parameters(model)
    log_info(f"model size: counter={counter} => {str_size}")
    model.to(args.device)
    log_info(f"model.to({args.device})")
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        log_info(f"model = torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
    return model

def get_data_loader_info(index_file):
    # get dir list from file
    dir_list = read_lines(index_file)
    dir_list = [d.split('\t')[0] for d in dir_list] # remove suffix (\t image_count)
    file_list = []
    info_map = {}
    i_start, i_ended = 0, 0
    for subdir in dir_list: # get file list from each dir
        i_start = i_ended
        full_dir = os.path.join(args.data_dir, subdir)
        f_names = os.listdir(full_dir)
        f_names = [f for f in f_names if str(f).endswith(".tif")]
        [file_list.append(os.path.join(full_dir, f)) for f in f_names]
        i_ended += len(f_names)
        info_map[full_dir] = (i_start, i_ended)
    # for
    tf = transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()])
    tf_l = transforms.Compose([ # rotate to left
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation((-10, -5)),
        transforms.ToTensor()
    ])
    tf_r = transforms.Compose([ # rotate to right
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation((5, 10)),
        transforms.ToTensor()
    ])
    if 'bcc' in index_file:
        res_dataset = DatasetHarddisk(file_list, [], image_transform=tf)
        res_dataset_l = DatasetHarddisk(file_list, [], image_transform=tf_l)
        res_dataset_r = DatasetHarddisk(file_list, [], image_transform=tf_r)
    else:
        res_dataset = DatasetHarddisk([], file_list, image_transform=tf)
        res_dataset_l = DatasetHarddisk([], file_list, image_transform=tf_l)
        res_dataset_r = DatasetHarddisk([], file_list, image_transform=tf_r)

    res_loader = tu_data.DataLoader(
        res_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    res_loader_l = tu_data.DataLoader(
        res_dataset_l,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    res_loader_r = tu_data.DataLoader(
        res_dataset_r,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    log_info(f"get_data_loader_info() ==================")
    log_info(f"  index      : {index_file}")
    log_info(f"  dir cnt    : {len(dir_list)}")
    log_info(f"  file cnt   : {len(file_list)}")
    log_info(f"  batch_count: {len(res_loader)}")
    log_info(f"  batch_size : {args.batch_size}")
    log_info(f"  shuffle    : False")
    log_info(f"  num_workers: {args.num_workers}")
    return res_loader, res_loader_l, res_loader_r, info_map

def get_predict_result(model, data_loader):
    res_arr = []
    b_cnt = len(data_loader)
    log_info(f"get_predict_result()...")
    for b_idx, (input, target) in enumerate(data_loader):
        if b_idx % args.log_interval == 0 or b_idx+1 == b_cnt:
            log_info(f"B{b_idx:03d}/{b_cnt}")
        input, target = input.to(args.device), target.to(args.device)
        output = model(input)
        res_arr.append(output)
    # for
    res = torch.concat(res_arr, dim=0)
    return res

def predict_by_avg_score(model, test_bcc_loader, test_bcc_info, test_other_loader, test_other_info):
    f_path_b = f"./ground_truth_value_1_0.res"  # BCC
    f_path_o = f"./ground_truth_value_0_0.res"  # other
    if os.path.exists(f_path_b):
        res_test_bcc = torch.load(f_path_b)
    else:
        with torch.no_grad(): res_test_bcc = get_predict_result(model, test_bcc_loader)
        torch.save(res_test_bcc, f_path_b)
    if os.path.exists(f_path_o):
        res_test_other = torch.load(f_path_o)
    else:
        with torch.no_grad(): res_test_other = get_predict_result(model, test_other_loader)
        torch.save(res_test_other, f_path_o)

    # =====================================================================
    # 2023.09.02 (Sep 2nd). accuracy on: model_resnet101_500x500.ckpt
    # test_bcc  : accu:0.970588 =  99 / 102
    # test_other: accu:0.935294 = 157 / 170
    # summary   : accu:0.941176 = 256 / 272
    # ---------------------------------------------------------------------
    numerator, denominator = 0, 0
    nu, de = 0, len(test_bcc_info)
    for full_dir, (i_start, i_ended) in test_bcc_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = res_test_bcc[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        if pred == 1: nu += 1
    # for
    accu = float(nu) / de
    log_info(f"test_bcc  : accu:{accu:.6f}={nu:3d}/{de:3d}")
    numerator += nu
    denominator += de

    nu, de = 0, len(test_other_info)
    for full_dir, (i_start, i_ended) in test_other_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = res_test_other[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        if pred == 0: nu += 1
    # for
    accu = float(nu) / de
    log_info(f"test_other: accu:{accu:.6f}={nu:3d}/{de:3d}")
    numerator += nu
    denominator += de

    accu = float(numerator) / denominator
    log_info(f"summary   : accu:{accu:.6f}={numerator}/{denominator}")
    log_info(f"checkpoint: {args.ckpt_load_path}")
    log_info(f"image_size: {args.image_size}")

def predict_by_image(model, loader0, loader1, loader2, data_info, ground_truth_value):
    # this is the cache for the prediction result.
    f_path0 = f"./ground_truth_value_{ground_truth_value}_0.res"
    f_path1 = f"./ground_truth_value_{ground_truth_value}_1.res"
    f_path2 = f"./ground_truth_value_{ground_truth_value}_2.res"
    if os.path.exists(f_path0):
        res_score_arr0 = torch.load(f_path0)
    else:
        with torch.no_grad(): res_score_arr0 = get_predict_result(model, loader0)
        torch.save(res_score_arr0, f_path0)
    if os.path.exists(f_path1):
        res_score_arr1 = torch.load(f_path1)
    else:
        with torch.no_grad(): res_score_arr1 = get_predict_result(model, loader1)
        torch.save(res_score_arr1, f_path1)
    if os.path.exists(f_path2):
        res_score_arr2 = torch.load(f_path2)
    else:
        with torch.no_grad(): res_score_arr2 = get_predict_result(model, loader2)
        torch.save(res_score_arr2, f_path2)

    numerator = 0
    denominator = len(data_info)
    for full_dir, (i_start, i_ended) in data_info.items():
        idx_list = list(range(i_start, i_ended))
        seq_score_arr0 = res_score_arr0[idx_list]
        seq_score_arr1 = res_score_arr1[idx_list]
        seq_score_arr2 = res_score_arr2[idx_list]

        # =====================================================================
        # 2023.09.02 (Sep 2nd). accuracy on: model_resnet101_500x500.ckpt
        # test_bcc  : accu:0.970588 =  99 / 102
        # test_other: accu:0.935294 = 159 / 170
        # summary   : accu:0.948529 = 258 / 272
        # ---------------------------------------------------------------------
        # seq_score_arr0 += seq_score_arr1
        # seq_score_arr0 += seq_score_arr2
        # avg_score = torch.mean(seq_score_arr0, dim=0)
        # pred = torch.argmax(avg_score)
        # if pred == ground_truth_value:
        #     numerator += 1

        # =====================================================================
        # 2023.09.02 (Sep 2nd). accuracy on: model_resnet101_500x500.ckpt
        # test_bcc  : accu:0.960784 =  98 / 102
        # test_other: accu:0.947059 = 161 / 170
        # summary   : accu:0.952206 = 259 / 272
        # ---------------------------------------------------------------------
        # get prediction result: 0 or 1.
        seq_pred_arr0 = torch.argmax(seq_score_arr0, dim=1)
        seq_pred_arr1 = torch.argmax(seq_score_arr1, dim=1)
        seq_pred_arr2 = torch.argmax(seq_score_arr2, dim=1)
        # check if correct: the prediction result match ground_truth
        seq_res_arr0 = torch.eq(seq_pred_arr0, ground_truth_value).long()
        seq_res_arr1 = torch.eq(seq_pred_arr1, ground_truth_value).long()
        seq_res_arr2 = torch.eq(seq_pred_arr2, ground_truth_value).long()
        # sum up the correct count
        seq_res_arr0 += seq_res_arr1
        seq_res_arr0 += seq_res_arr2
        nu = torch.ge(seq_res_arr0, 2).sum() # of 3 prediction, 2 or more are correct
        de = seq_pred_arr0.shape[0]
        accu = float(nu) / de
        # assume threshold is 0.6.
        # Of the images in a sequence, if BCC images cover more than 60%, then we say sequence is BCC.
        # And if less than 60%, then we say sequence is "other".
        threshold = 0.6
        if ground_truth_value == 1: # handle BCC case
            if accu > threshold: numerator += 1
        else: # handle "other" case
            if accu >= (1-threshold): numerator += 1
    # for
    return numerator, denominator

def main():
    model = load_model()
    model.eval()
    # file_train_bcc   = os.path.join(args.data_dir, "dir_train_bcc.txt")
    # file_train_other = os.path.join(args.data_dir, "dir_train_other.txt")
    # train_bcc_loader, train_bcc_info     = get_data_loader_info(file_train_bcc)
    # train_other_loader, train_other_info = get_data_loader_info(file_train_other)
    file_test_bcc   = os.path.join(args.data_dir, "dir_val_bcc.txt")
    file_test_other = os.path.join(args.data_dir, "dir_val_other.txt")
    basal_loader, basal_loader1, basal_loader2, basal_info = get_data_loader_info(file_test_bcc)
    other_loader, other_loader1, other_loader2, other_info = get_data_loader_info(file_test_other)

    # calculate accuracy by average score.
    # predict_by_avg_score(model, basal_loader, basal_info, other_loader, other_info)

    # Calculate by single image classification result. For example:
    # Of a sequence, if more than half is marked "BCC", then assume the sequence is BCC.
    log_info(f"predict_by_image()...")
    numerator, denominator = 0, 0
    nu, de = predict_by_image(model, basal_loader, basal_loader1, basal_loader2, basal_info, 1)
    accu = float(nu) / de
    log_info(f"test_bcc  : accu:{accu:.6f}={nu:3d}/{de:3d}")
    numerator += nu
    denominator += de
    nu, de = predict_by_image(model, other_loader, other_loader1, other_loader2, other_info, 0)
    accu = float(nu) / de
    log_info(f"test_other: accu:{accu:.6f}={nu:3d}/{de:3d}")
    numerator += nu
    denominator += de

    accu = float(numerator) / denominator
    log_info(f"summary   : accu:{accu:.6f}={numerator}/{denominator}")
    log_info(f"checkpoint: {args.ckpt_load_path}")
    log_info(f"image_size: {args.image_size}")


if __name__ == '__main__':
    args = parse_args()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")
    main()
