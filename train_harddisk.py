"""
training and testing on images from hard-disk of AK
"""

import argparse
import os
import time
import torch
import numpy as np
from torch.backends import cudnn
import torch.utils.data as tu_data
import torchvision.transforms as transforms

import models
from dataset_harddisk import DatasetHarddisk
from utils import log_info, get_lr, get_time_ttl_and_eta, count_parameters, read_lines, str2bool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7, 6, 5, 4])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--image_size", nargs="+", type=int, default=[224, 224])
    parser.add_argument("--resnet", type=str, default="resnet101")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--ckpt_save_interval", type=int, default=10)
    parser.add_argument("--ckpt_save_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_load_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./dataset/harddisk")
    parser.add_argument("--ifile_train_bcc", type=str, default="dir_train_bcc.txt", help="index file")
    parser.add_argument("--ifile_train_other", type=str, default="dir_train_other.txt")
    parser.add_argument("--ifile_test_bcc", type=str, default="dir_val_bcc.txt")
    parser.add_argument("--ifile_test_other", type=str, default="dir_val_other.txt")
    parser.add_argument("--output_dir_list", type=str2bool, default=False)
    _args = parser.parse_args()

    # add device
    gpu_ids = _args.gpu_ids
    log_info(f"gpu_ids : {gpu_ids}")
    _args.device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"

    # set random seed
    seed = _args.seed  # if seed is 0. then ignore it.
    log_info(f"args.seed : {seed}")
    if seed:
        log_info(f"  torch.manual_seed({seed})")
        log_info(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")
    cudnn.benchmark = True
    return _args

def get_test_data_loader_info(index_file, target):
    # get dir list from file
    dir_list = read_lines(index_file)
    file_list = []
    info_map = {}
    i_start, i_ended = 0, 0
    log_info(f"get_test_data_loader_info() ==================")
    log_info(f"  index      : {index_file}")
    log_info(f"  dir count  : {len(dir_list)}")
    log_info(f"  batch_size : {args.batch_size}")
    log_info(f"  shuffle    : False")
    log_info(f"  num_workers: {args.num_workers}")
    d_cnt = len(dir_list)
    for d_idx, subdir in enumerate(dir_list): # get file list from each dir
        subdir = subdir.split('\t')[0]
        i_start = i_ended
        full_dir = os.path.join(args.data_dir, subdir)
        f_names = os.listdir(full_dir)
        f_names = [f for f in f_names if str(f).endswith(".tif")]
        [file_list.append(os.path.join(full_dir, f)) for f in f_names]
        i_ended += len(f_names)
        info_map[full_dir] = (i_start, i_ended)
        log_info(f"  {d_idx:4d}/{d_cnt}: {subdir}\t{len(f_names)}") if args.output_dir_list else None
    # for
    tf = transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()])
    if target == 1:
        res_dataset = DatasetHarddisk(file_list, [], image_transform=tf)
    else:
        res_dataset = DatasetHarddisk([], file_list, image_transform=tf)

    res_loader = tu_data.DataLoader(
        res_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    log_info(f"  file_count : {len(file_list)}")
    log_info(f"  batch_count: {len(res_loader)}")
    return res_loader, info_map

def get_train_data_loader():
    train_tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((-15, 15)),
        # transforms.RandomEqualize(p=0.2),
        transforms.ToTensor(),
    ])
    log_info(f"Train data augment: RandomHorizontalFlip(p=0.5)")
    log_info(f"Train data augment: RandomRotation((-15, 15))")

    # from dir list to file list. Each dir is a leaf dir, containing *.tif images
    def dirs2files(dir_list):
        f_list = []
        d_cnt = len(dir_list)
        for d_idx, d in enumerate(dir_list):
            d = d.split('\t')[0]
            dd = os.path.join(args.data_dir, d)
            f_names = os.listdir(dd)
            f_names = [f for f in f_names if str(f).endswith(".tif")]
            [f_list.append(os.path.join(dd, f)) for f in f_names]
            log_info(f"  {d_idx:4d}/{d_cnt}: {d}\t{len(f_names)}") if args.output_dir_list else None
        return f_list

    # get file paths. The file contains dir.
    file_train_bcc   = os.path.join(args.data_dir, args.ifile_train_bcc)
    file_train_other = os.path.join(args.data_dir, args.ifile_train_other)

    # get dir list from file
    train_bcc_dir_list   = read_lines(file_train_bcc)
    train_other_dir_list = read_lines(file_train_other)

    # get file list from dir list
    log_info(f"train BCC dirs: ---------------------") if args.output_dir_list else None
    train_bcc_list = dirs2files(train_bcc_dir_list)
    log_info(f"train other dirs: ---------------------") if args.output_dir_list else None
    train_other_list = dirs2files(train_other_dir_list)
    train_dataset = DatasetHarddisk(train_bcc_list, train_other_list, image_transform=train_tf)

    train_loader = tu_data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    log_info(f"  shuffle        : True")
    log_info(f"  num_workers    : {args.num_workers}")

    return train_loader

def save_model(model, epoch, lr, train_accu):
    m = model
    if isinstance(m, torch.nn.DataParallel):
        m = m.module
    state_dict = {
        "model": m.state_dict(),
        "epoch": epoch,
        "lr"   : lr,
        "train_accu": train_accu,
        "resnet"    : args.resnet,
    }
    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
        log_info(f"os.makedirs({args.ckpt_save_dir})")
    ckpt_path = os.path.join(args.ckpt_save_dir, f"model_Epoch{epoch:03d}.ckpt")
    log_info(f"save model to: {ckpt_path}... epoch:{epoch}, lr:{lr:.8f}, train_accu:{train_accu:.4f}")
    torch.save(state_dict, ckpt_path)
    log_info(f"save model to: {ckpt_path}... Done")

def load_model(ckpt_path, model):
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Invalid file path: {ckpt_path}")
    log_info(f"load model from: {ckpt_path}...")
    state_dict = torch.load(ckpt_path)
    m = state_dict['model']
    epoch = state_dict['epoch']
    lr = state_dict['lr']
    train_accu = state_dict['train_accu']
    model.load_state_dict(m)
    log_info(f"load model from: {ckpt_path}...Done. epoch:{epoch}, lr:{lr:.8f}, train_accu:{train_accu:.4f}")
    return model, epoch, lr

def get_or_load_model():
    if args.resnet in ['resnet101', '101']:
        model = models.resnet101(weights=None, num_classes=2)
    elif args.resnet in ['resnet152', '152']:
        model = models.resnet152(weights=None, num_classes=2)
    else:
        raise ValueError(f"Invalid resnet option: {args.resnet}")
    if args.ckpt_load_path:
        load_model(args.ckpt_load_path, model)
    counter, str_size = count_parameters(model)
    log_info(f"model size: counter={counter} => {str_size}")
    model.to(args.device)
    log_info(f"model.to({args.device})")
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        log_info(f"model = torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
    return model

def calc_accuracy(model, bcc_loader, other_loader, bcc_info, other_info):
    model.eval() # switch to evaluation mode
    numerator_img, denominator_img = 0, 0
    numerator_seq, denominator_seq = 0, 0
    output_bcc_arr = []
    # handle BCC data, image accuracy ---------------------
    nu, de = 0, 0
    with torch.no_grad():
        for input, target in bcc_loader:
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            output_bcc_arr.append(output)
            pred = torch.argmax(output, dim=1)
            de += input.shape[0]
            nu += torch.eq(target, pred).sum()
        # for
    # with
    accu_bcc_img = float(nu) / de
    log_info(f"accu_bcc_img  : {accu_bcc_img:.6f} = {nu:4d} / {de:4d}")
    numerator_img += nu
    denominator_img += de

    # handle BCC data, image-sequence accuracy ------------
    output_bcc = torch.concat(output_bcc_arr, dim=0)
    nu, de = 0, 0
    for full_dir, (i_start, i_ended) in bcc_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = output_bcc[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        de += 1
        if pred == 1: nu += 1
    # for
    accu_bcc_seq = float(nu) / de
    log_info(f"accu_bcc_seq  : {accu_bcc_seq:.6f} = {nu:4d} / {de:4d}")
    numerator_seq += nu
    denominator_seq += de

    output_other_arr = []
    # handle other data, image accuracy -------------------
    nu, de = 0, 0
    with torch.no_grad():
        for input, target in other_loader:
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            output_other_arr.append(output)
            pred = torch.argmax(output, dim=1)
            de += input.shape[0]
            nu += torch.eq(target, pred).sum()
        # for
    # with
    accu_other_img = float(nu) / de
    log_info(f"accu_other_img: {accu_other_img:.6f} = {nu:4d} / {de:4d}")
    numerator_img += nu
    denominator_img += de

    # handle other data, image-sequence accuracy ----------
    output_other = torch.concat(output_other_arr, dim=0)
    nu, de = 0, 0
    for full_dir, (i_start, i_ended) in other_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = output_other[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        de += 1
        if pred == 0: nu += 1
    # for
    accu_other_seq = float(nu) / de
    log_info(f"accu_other_seq: {accu_other_seq:.6f} = {nu:4d} / {de:4d}")
    numerator_seq += nu
    denominator_seq += de

    # summarize
    accu_img = float(numerator_img) / denominator_img
    accu_seq = float(numerator_seq) / denominator_seq
    log_info(f"accu_img      : {accu_img:.6f} = {numerator_img:4d} / {denominator_img:4d}")
    log_info(f"accu_seq      : {accu_seq:.6f} = {numerator_seq:4d} / {denominator_seq:4d}")
    return accu_img, accu_bcc_img, accu_other_img, accu_seq, accu_bcc_seq, accu_other_seq

def main():
    train_loader = get_train_data_loader()
    file_test_bcc   = os.path.join(args.data_dir, args.ifile_test_bcc)
    file_test_other = os.path.join(args.data_dir, args.ifile_test_other)
    tb_loader, tb_info = get_test_data_loader_info(file_test_bcc, 1)    # test bcc
    to_loader, to_info = get_test_data_loader_info(file_test_other, 0)  # test other

    model = get_or_load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr/1000)
    loss_fn = torch.nn.CrossEntropyLoss()
    b_cnt = len(train_loader)   # batch count
    e_cnt = args.epoch          # epoch count
    be_total = b_cnt * e_cnt
    log_interval = args.log_interval
    start_time = time.time()
    log_info(f"batch_cnt : {b_cnt}")
    log_info(f"batch_size: {args.batch_size}")
    log_info(f"image_size: {args.image_size}")
    log_info(f"resnet    : {args.resnet}")
    log_info(f"epoch_cnt : {e_cnt}")
    for e_idx in range(args.epoch):
        lr = get_lr(optimizer)
        log_info(f"E{e_idx:03d}/{e_cnt} ---------------------- lr:{lr:.8f}")
        numerator, denominator = 0, 0 # to calculate accuracy
        model.train() # switch to training mode
        for b_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(input)
            pred = torch.argmax(output, dim=1)
            denominator += input.shape[0]
            numerator += torch.eq(target, pred).sum()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if b_idx % log_interval == 0 or b_idx == b_cnt - 1:
                elp, eta = get_time_ttl_and_eta(start_time, e_idx*b_cnt+b_idx, be_total)
                accu = float(numerator) / denominator
                s = f"B{b_idx:04d}/{b_cnt}. loss:{loss:.8f}, accu:{accu:.6f}. elp:{elp}, eta:{eta}."
                log_info(s)
        # for batch
        scheduler.step()
        calc_accuracy(model, tb_loader, to_loader, tb_info, to_info)
        e_tmp = e_idx + 1
        if e_tmp % args.ckpt_save_interval == 0 or e_tmp == e_cnt:
            accu = float(numerator) / denominator
            save_model(model, e_tmp, lr, accu)
    # for epoch

if __name__ == "__main__":
    args = parse_args()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")
    main()
