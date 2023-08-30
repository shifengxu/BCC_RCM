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
    if 'bcc' in index_file:
        res_dataset = DatasetHarddisk(file_list, [], image_transform=tf)
    else:
        res_dataset = DatasetHarddisk([], file_list, image_transform=tf)

    res_loader = tu_data.DataLoader(
        res_dataset,
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
    return res_loader, info_map

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

def main():
    model = load_model()
    model.eval()
    # file_train_bcc   = os.path.join(args.data_dir, "dir_train_bcc.txt")
    # file_train_other = os.path.join(args.data_dir, "dir_train_other.txt")
    # train_bcc_loader, train_bcc_info     = get_data_loader_info(file_train_bcc)
    # train_other_loader, train_other_info = get_data_loader_info(file_train_other)
    file_test_bcc   = os.path.join(args.data_dir, "dir_val_bcc.txt")
    file_test_other = os.path.join(args.data_dir, "dir_val_other.txt")
    test_bcc_loader, test_bcc_info     = get_data_loader_info(file_test_bcc)
    test_other_loader, test_other_info = get_data_loader_info(file_test_other)
    with torch.no_grad():
        res_test_bcc   = get_predict_result(model, test_bcc_loader)
        res_test_other = get_predict_result(model, test_other_loader)
    # with
    numerator, denominator = 0, 0
    nu, de = 0, 0
    for full_dir, (i_start, i_ended) in test_bcc_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = res_test_bcc[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        de += 1
        if pred == 1: nu += 1
    # for
    accu = float(nu) / de
    log_info(f"test_bcc  : accu:{accu:.6f}={nu}/{de}")
    numerator += nu
    denominator += de

    nu, de = 0, 0
    for full_dir, (i_start, i_ended) in test_other_info.items():
        idx_list = list(range(i_start, i_ended))
        res_arr = res_test_other[idx_list]
        res_avg = torch.mean(res_arr, dim=0)
        pred = torch.argmax(res_avg)
        de += 1
        if pred == 0: nu += 1
    # for
    accu = float(nu) / de
    log_info(f"test_other: accu:{accu:.6f}={nu}/{de}")
    numerator += nu
    denominator += de

    accu = float(numerator) / denominator
    log_info(f"summary   : accu:{accu:.6f}={numerator}/{denominator}")
    log_info(f"checkpoint: {args.ckpt_load_path}")
    log_info(f"image_size: {args.image_size}")

    # todo: the above is to calculate accuracy by average score.
    # How about by single image classification result. For example:
    # Of a single image, "BCC" type mark 1 and "other"s" mark 0.
    # If average mark reaches 0.4, then assume the sequence is BCC.

if __name__ == '__main__':
    args = parse_args()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")
    main()
