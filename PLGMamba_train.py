import argparse
import torch
import torch.nn as nn
import os
import time
import torch.optim as optim
import scipy.io as sio
from loss_func import HLoss
from dataLoader import TrainHSRDataset
from utils import train, save_checkpoint, gpu_configuration, setup_seed_cudnn
from model.PLGMamba import PLGMamba
from torch.utils.data import DataLoader

# Global variable
GPUINFO = ["0, 1", 2]  # GPU编号与数量, e.g., 2 GPUs: GPUINFO = ["0, 1", 2] | 1 GPU: GPUINFO = ["0", 1]

# Default configuration
parser = argparse.ArgumentParser(
    description='HSI super-resolution using MSDformer interpolation')
parser.add_argument('--method', type=str, default='PLGMamba', help='the method for SSR')
parser.add_argument('--scale', type=int, default=4, help='SSR scale, i.e., x2, x4, x8')
parser.add_argument('--cropsize', type=int, default=256, help='crop size of subimage, i.e., 256 or 128')
parser.add_argument('--overlapixel', type=int, default=128, help='overlapping pixels among subimages')
parser.add_argument('--sel_num', type=int, default=1024, help='randomly select quantity')
parser.add_argument('--batchsize', type=int, default=12, help='batch size')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")  # default=1e-4
parser.add_argument('--pretrained_model_path',  type=str, default=None)
parser.add_argument('--dataroot', type=str, default='/home/zcl/Dataset/semantic_seg_dataset/Chikusei.mat',
                    help='path to HSI dir')

parser.add_argument("--step", type=int, default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--out_path", type=str, default='./model_zoo/PLGMamba_x4', help='path log files')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device, num_w, pin_m = gpu_configuration(gpu_id=GPUINFO[0], gpu_num=GPUINFO[1])  # setting the number of GPUs
setup_seed_cudnn(seed=2025, cudnn_flag=True)  # setting the random seed and cudnn configuration
args = parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    args.hr_hsi_dir = args.dataroot
    args.model_zoo_dir = args.out_path
    if not os.path.exists(args.model_zoo_dir):
        os.makedirs(args.model_zoo_dir)

    # load dataset
    print("\nloading dataset ...")
    traindata = TrainHSRDataset(dataroot=args.hr_hsi_dir, cropsize=args.cropsize, testarea=(1511, 1767, 'row'),
                                overlapixel=args.overlapixel, rdm_sel_num=args.sel_num, upscale=args.scale, arg=True)
    print(f"Iteration per epoch: {len(traindata)}")
    trainloader = DataLoader(dataset=traindata, batch_size=args.batchsize, shuffle=True, num_workers=num_w,
                             pin_memory=pin_m, drop_last=True)
    check_train_sample = next(iter(trainloader))

    # import model
    model = PLGMamba(in_channels=traindata.c, g_num=10, scale=args.scale, fea_dim=64, res_scale=0.4,
                     mlp_ratio=1, block_num_g=2, block_num_a=3)
    # model = model_generator(args.method, in_channels=channels, upscale=args.scale)
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(device, dtype=torch.float)
    param_numbers = sum(param.numel() for param in model.parameters()) / 1e3
    print('Parameters number is {:.3f}'.format(param_numbers))

    # loss function
    metric = nn.L1Loss(reduction='mean').to(device)
    criterion = HLoss(la1=0.6, la2=0.2, sam=False, gra=False)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # 加载训练完毕后的模型参数。其中，pth文件中包括模型参数、优化器、epoch以及学习率
    if args.pretrained_model_path is not None:
        saved_pth_file = torch.load(args.pretrained_model_path, map_location=device)
        model.load_state_dict(saved_pth_file['state_dict'])
        optimizer.load_state_dict(saved_pth_file['optimizer'])

    loss_epoch = []
    performance_epoch = []
    for epoch in range(1, args.epochs + 1):
        loss_batch, performance_batch, lr = train(trainloader, optimizer, model, criterion, metric, epoch, args, device)
        loss_epoch.append(loss_batch)
        performance_epoch.append(performance_batch)
        sio.savemat(os.path.join(args.model_zoo_dir,
                                 'trainLoss_{}.mat'.format('x'+str(args.scale))), {'train_loss': loss_epoch,
                                                                                   'train_metric': performance_epoch})
        save_checkpoint(args.model_zoo_dir, epoch, model, optimizer, lr, args.scale)

    elapsed_time = time.time() - start_time
    sio.savemat(os.path.join(args.model_zoo_dir, 'trainExpdata_{}.mat'.format('x'+str(args.scale))),
                {'train_loss': loss_epoch, 'train_metric': performance_epoch, 'train_time': elapsed_time})
    print("It takes {:.4f}s for processing".format(elapsed_time))
