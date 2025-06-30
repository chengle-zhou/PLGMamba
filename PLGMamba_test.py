import os
import torch
import argparse
from tqdm import tqdm
import time
import numpy as np
import scipy.io as sio
import torch.nn as nn
from torch.autograd import Variable
from utils import gpu_configuration, setup_seed_cudnn, ssrImgMerge, subImgMerge, ssr_numpy_metric
from dataLoader import TestHSRDataset
from model.PLGMamba import PLGMamba
from torch.utils.data import DataLoader
from collections import OrderedDict

# Global variable
GPUINFO = ["0", 1]  # GPU编号与数量, e.g., 2 GPUs: GPUINFO = ["0, 1", 2] | 1 GPU: GPUINFO = ["0", 1]

# Default configuration
parser = argparse.ArgumentParser(description='HSI super-resolution using MSDformer')
parser.add_argument('--method', type=str, default='PLGMamba', help='the method for SSR')
parser.add_argument('--scale', type=int, default=4, help='SSR scale, i.e., x2, x4, x8')
parser.add_argument('--cropsize', type=int, default=256, help='crop size of subimage, i.e., 256 or 128')
parser.add_argument('--overlapixel', type=int, default=128, help='overlapping pixels among subimages')
parser.add_argument('--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--trained_model', type=str, default=r'./model_zoo/PLGMamba_x4/best_model_x4.pth',  # 设置高分辨率图片路径参数
                    help='path to HSI dir')
parser.add_argument('--dataroot', type=str, default='/home/zcl/Dataset/semantic_seg_dataset/Chikusei.mat',  # 设置高分辨率图片路径参数
                    help='path to HSI dir')
parser.add_argument('--ssr_img_dir', type=str, default=r'./results/PLGMamba_x4',  # 设置低分辨率路径参数
                    help='path to desired output dir for downsampled images')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device, num_w, pin_m = gpu_configuration(gpu_id=GPUINFO[0], gpu_num=GPUINFO[1])  # setting the number of GPUs
setup_seed_cudnn(seed=2025, cudnn_flag=True)  # setting the random seed and cudnn configuration
args = parser.parse_args()


if __name__ == '__main__':
    args.hr_img_dir = args.dataroot
    args.ssr_img_dir = args.ssr_img_dir
    args.trained_model = args.trained_model
    if not os.path.exists(args.ssr_img_dir):
        os.makedirs(args.ssr_img_dir)
    testdata = TestHSRDataset(dataroot=args.hr_img_dir, cropsize=args.cropsize, testarea=(1511, 1767, 'row'),
                              overlapixel=args.overlapixel, upscale=args.scale, arg=False)
    print(f"Iteration per epoch: {len(testdata)}")
    testloader = DataLoader(dataset=testdata, batch_size=args.batchsize, shuffle=False,
                            num_workers=num_w, pin_memory=pin_m, drop_last=False)

    # load model
    model = PLGMamba(in_channels=testdata.c, g_num=10, scale=args.scale, fea_dim=64, res_scale=0.4,
                     mlp_ratio=1, block_num_g=2, block_num_a=3)
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(device, dtype=torch.float)

    saved_pth_file = torch.load(args.trained_model, map_location=device)
    model_weights = saved_pth_file['state_dict']
    model.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in model_weights.items()}))

    model.eval()
    testbar = tqdm(testloader)
    lr_hsis_list, hr_hsis_list, ssr_hsis_list, time_exec_list = [], [], [], []
    for iteration, batch in enumerate(testbar, 1):
        start_time = time.time()
        lr_hsi, hr_hsi = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
        breakponit_time = time.time()
        with torch.no_grad():
            ssr_hsi = model(lr_hsi)
        lr_hsis_list.append(lr_hsi)
        ssr_hsis_list.append(ssr_hsi)
        hr_hsis_list.append(hr_hsi)
        once_run_time = breakponit_time - start_time  # time
        time_exec_list.append(once_run_time)
        # print("| ****** {}: {:0>3} | {} | {:.3f}s ****** |"
        #       .format(args.method, iteration + 1, len(testbar), once_run_time))
    ssr_hsis = torch.cat(ssr_hsis_list, dim=0).permute(0, 3, 2, 1).cpu().numpy()
    hr_hsis = torch.cat(hr_hsis_list, dim=0).permute(0, 3, 2, 1).cpu().numpy()
    lr_hsis = torch.cat(lr_hsis_list, dim=0).permute(0, 3, 2, 1).cpu().numpy()

    # image block merge
    [ssr_hsi_split, ssr_hsi_merge, h_pad, w_pad] = ssrImgMerge(ssr_hsis, testdata)
    [hr_hsi_split, hr_hsi_merge, _, _] = ssrImgMerge(hr_hsis, testdata)
    [lr_hsi_split, lr_hsi_merge, n_h, n_w] = subImgMerge(lr_hsis, testdata, scale=args.scale)

    # evaluate performance
    cc_b, sam_b, psnr_b, ssim_b, rmse_b, ergas_b = ssr_numpy_metric(hr_hsi_split, ssr_hsi_split)
    metric_block = np.array([cc_b, sam_b, psnr_b, ssim_b, rmse_b, ergas_b], dtype=object)
    cc_o, sam_o, psnr_o, ssim_o, rmse_o, ergas_o = ssr_numpy_metric(hr_hsi_merge[None, :, :, :],
                                                                    ssr_hsi_merge[None, :, :, :])
    metric_overall = np.array([cc_o, sam_o, psnr_o, ssim_o, rmse_o, ergas_o], dtype=object)

    # save testing result as '.mat'
    import_name = args.method + '_' + 'x' + str(args.scale) + '.mat'
    sio.savemat(os.path.join(args.ssr_img_dir, import_name),
                {'ssr_hsi_split': ssr_hsi_split, 'ssr_hsi_merge': ssr_hsi_merge, 'h_pad': h_pad, 'w_pad': w_pad,
                 'tar_hsi_split': hr_hsi_split, 'tar_hsi_merge': hr_hsi_merge,
                 'lr_hsi_split': lr_hsi_split, 'lr_hsi_merge': lr_hsi_merge, 'n_h': n_h, 'n_w': n_w,
                 'metric_block': metric_block, 'metric_overall': metric_overall,
                 'test_time': np.array(time_exec_list)
                 })
    print("It takes {:.4f}s for processing".format(np.sum(np.array(time_exec_list))))
