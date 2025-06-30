import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import datetime
import random


# current time
def current_time():
    """capture the current time"""
    now = str(datetime.date.today())
    curr_time = now.replace("-", "")
    return curr_time


# device configuration
def gpu_configuration(gpu_id="0,1", gpu_num=2):
    """ GPU configuration (Using nn.DataParallel for training) """
    cuda_flag = torch.cuda.is_available()
    gpu_ids = [int(i) for i in gpu_id.split(',')]
    if cuda_flag and len(gpu_ids) == gpu_num and gpu_num >= 2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_DEVICE_ORDER"] = gpu_id  # setting the number of GPUs
        device = torch.device('cuda')
        num_w, pin_m = 8, True
        print('GPUs are available for {} devices!'.format(gpu_num))
    elif cuda_flag and (len(gpu_ids) == 1 or gpu_num == 1):
        device = torch.device('cuda')
        num_w, pin_m = 0, False
        print('GPU are available for 1 device!')
    else:
        device = torch.device('cpu')
        num_w, pin_m = 0, False
        print('CPU have replaced GPU for tasks!')
    return device, num_w, pin_m


# setting random seed and speed configuration
def setup_seed_cudnn(seed=2025, cudnn_flag=False, seed_mode='torch'):
    if seed_mode == 'numpy':
        np.random.seed(seed)
        random.seed(seed)
    elif seed_mode == 'torch':
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # 并行gpu
    elif seed_mode == 'all':
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # 并行gpu
    # 模型结构固定且保持输入大小不变设置为(True)实现加速, 否则设置为(False)
    torch.backends.cudnn.benchmark = cudnn_flag
    # 设置为True，则每次返回的卷积算法将是确定的，即默认算法。
    torch.backends.cudnn.deterministic = cudnn_flag   # cpu/gpu结果一致


"""
       只操作y通道
       因为我们感兴趣的不是颜色变化(存储在 CbCr 通道中的信息)而只是其亮度(Y 通道);
       根本原因在于相较于色差，人类视觉对亮度变化更为敏感。
"""
def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))

"""
        RGB转YCBCR
        Y=0.257*R+0.564*G+0.098*B+16
        Cb=-0.148*R-0.291*G+0.439*B+128
        Cr=0.439*R-0.368*G-0.071*B+128
"""
def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))

"""
        YCBCR转RGB
        R=1.164*(Y-16)+1.596*(Cr-128)
        G=1.164*(Y-16)-0.392*(Cb-128)-0.813*(Cr-128)
        B=1.164*(Y-16)+2.017*(Cb-128)
"""
def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))

# PSNR 计算
# def calc_psnr(img1, img2):
#     return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
# def calc_psnr(img1, img2):
#     # 将输入转换为PyTorch张量
#     img1_tensor = torch.tensor(img1)
#     img2_tensor = torch.tensor(img2)
#
#     # 计算PSNR
#     mse = torch.mean((img1_tensor - img2_tensor) ** 2)
#     psnr = 10. * torch.log10(1. / mse)
#     return psnr
import torch
import numpy as np

def calc_psnr(img1, img2):
    # 将输入转换为PyTorch张量
    # img1_tensor = torch.tensor(img1)
    # img2_tensor = torch.tensor(img2)
    img1_tensor = img1.clone().detach()
    img2_tensor = img2.clone().detach()

    # 计算PSNR
    mse = torch.mean((img1_tensor - img2_tensor) ** 2)
    psnr = 10. * torch.log10(1. / mse)

    # 计算CC
    img1_flat = img1_tensor.view(-1)
    img2_flat = img2_tensor.view(-1)
    cc = torch.mean((img1_flat - torch.mean(img1_flat)) * (img2_flat - torch.mean(img2_flat))) / \
         (torch.std(img1_flat) * torch.std(img2_flat))

    # 计算SAM
    cos_sim = torch.nn.CosineSimilarity(dim=0)
    sam = torch.acos(cos_sim(img1_flat, img2_flat)) * (180.0 / np.pi)

    # 计算RMSE
    rmse = torch.sqrt(mse)

    return psnr, cc, sam, rmse
    # return psnr


# SSR evaluate indicators
def ssr_numpy_metric(target, ssr_hsi):
    cc_array, sam_array, psnr_array, = np.array([]), np.array([]), np.array([])
    ssim_array, rmse_array, ergas_array, = np.array([]), np.array([]), np.array([])
    [N, _, _, _] = target.shape
    for i in range(N):
        tar_img, ssr_img = target[i], ssr_hsi[i]
        cc_array = np.append(cc_array, numpy_CC(tar_img, ssr_img))
        sam_array = np.append(sam_array, numpy_SAM(tar_img, ssr_img))
        psnr_array = np.append(psnr_array, numpy_PSNR(tar_img, ssr_img))
        ssim_array = np.append(ssim_array, numpy_SSIM(tar_img, ssr_img))
        rmse_array = np.append(rmse_array, numpy_RMSE(tar_img, ssr_img))
        ergas_array = np.append(ergas_array, numpy_ERGAS(tar_img, ssr_img))
    return cc_array, sam_array, psnr_array, ssim_array, rmse_array, ergas_array




# Correlation coefficient (CC)
def numpy_CC(im_true, im_resc):
    """
    calculate correlation coefficient of bands bwtween ref. HSI and rec. HSI
    Author: chengle zhou, SYSU, 2024-03-29
    Args:
        im_true: H x W x C
        im_resc: H x W x C
    Returns: 1 x C
    """
    h, w, c = im_true.shape
    data_sum = np.sum((im_true * im_resc).reshape(h*w, -1), axis=0)
    im_true_sum = np.sum((im_true.reshape(h*w, -1))**2, axis=0)
    im_resc_sum = np.sum((im_resc.reshape(h*w, -1))**2, axis=0)
    im_true_mean = np.mean(im_true.reshape(h*w, -1), axis=0)
    im_resc_mean = np.mean(im_resc.reshape(h*w, -1), axis=0)
    data_mean = h*w*im_true_mean*im_resc_mean
    c1 = data_sum - data_mean
    c2 = im_resc_sum - h*w*im_resc_mean**2
    c3 = im_true_sum - h*w*im_true_mean**2
    cc = c1 / np.sqrt(c2*c3)
    return cc


# Spectral angle mapper (SAM)  -- numpy version
def numpy_SAM(im_true, im_resc):
    h, w, c = im_true.shape
    sum1 = np.sum(im_true * im_resc, axis=-1)
    sum2 = np.sum(im_true * im_true, axis=-1)
    sum3 = np.sum(im_resc * im_resc, axis=-1)
    t = np.sqrt(sum2*sum3)
    t[t == 0] = 1.
    sum1[sum1 == 0] = 1.
    num = np.sum(t.reshape(h*w, -1) > 0)
    t = sum1 / t
    t[t > 1] = 1.
    angle = np.arccos(t)
    sumangle = np.sum(angle.reshape(1, -1))
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    sam = averangle * 180 / 3.14159256
    return sam

# Peak Signal-to-Noise Ratio (PSNR)
def numpy_PSNR(im_true, im_resc, maxgray=255):
    h, w, c = im_true.shape
    MSE = np.mean(((im_true - im_resc).reshape(h * w, -1)) ** 2, axis=0)
    val_rate = (maxgray ** 2) / MSE
    PSNR = 10 * np.log10(val_rate)
    return PSNR

# Relative dimensionless global error synthesis (ERGAS)
def numpy_ERGAS(im_true, im_resc, ratio=1.0):
    h, w, c = im_true.shape
    mes = np.mean(((im_true - im_resc) ** 2).reshape(h*w, -1), axis=0)
    mean_square = np.mean(im_true.reshape(h*w, -1), axis=0) ** 2
    diff_rate_sum = np.sum(mes / mean_square)
    ergas = 100 * (1 / ratio) * ((diff_rate_sum / c) ** 0.5)
    return ergas


# Root-Mean-Square Error (RMSE)
def numpy_RMSE(im_true, im_resc):
    h, w, c = im_true.shape
    mse = np.mean(((im_true - im_resc) ** 2).reshape(h*w, -1), axis=0)
    rmse = np.sqrt(mse)
    return rmse


# Structural Similarity Index (SSIM)
def numpy_SSIM(im_true, im_resc):
    ssims = []
    for i in range(im_true.shape[2]):
        ssims.append(_ssim(im_true[..., i], im_resc[..., i]))
    return np.array(ssims)



def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssrImgMerge(ssr_imgs_numpy, info):
    # 获取图像块的分割信息
    cropsize = info.cropsize  # 图像分块后, 正方形子图大小
    overlapixel = info.overlapixel  # 图像分块后, 每个分块图像重叠数
    n_h = info.n_h  # 图像分块行数
    n_w = info.n_w  # 图像分块列数
    h_pad = info.h_te_pad  # 原图像行方向padding数
    w_pad = info.w_te_pad  # 原图像列方向padding数
    h_ori = info.h_te_bef  # 原图像行数
    w_ori = info.w_te_bef  # 原图像列数
    h_te = info.h_te  # 原图像padding后行数
    w_te = info.w_te  # 原图像padding后列数
    # 图像块拼接
    idx = 0
    ssr_imgs_row = []
    for r in range(n_h):  # 遍历行
        ssr_imgs_col = []
        for c in range(n_w):  # 遍历列
            ssr_imgs_col.append(ssr_imgs_numpy[idx])
            idx = idx + 1
        # SSR图像块列方向拼接, 并放入list
        ssr_imgs_row.append(np.concatenate(ssr_imgs_col, axis=1))
    # SSR图像块列方向拼接后的结果行方向拼接
    ssr_imgs = np.concatenate(ssr_imgs_row,  axis=0)
    # 以及图像块重叠尺度(overlapixel)还原测试图像
    # 列删除
    if cropsize < cropsize * n_w:  # 判断列方向是否有多个(>2)图像块
        if overlapixel % 2 == 0:  # 获取列方向图像块重叠
            col_del_left = col_del_right = int(overlapixel / 2)
        else:
            col_del_left = int(np.ceil(overlapixel / 2.))
            col_del_right = int(np.floor(overlapixel / 2.))
        col_del_nodes = np.linspace(cropsize, cropsize*n_w, n_w)[:-1]  # 获取列方向图像块重叠像元, 待删除
        col_del_pixels = []
        for i in range(len(col_del_nodes)):
            col_del_pixels.append(np.arange(col_del_nodes[i] - col_del_left, col_del_nodes[i] + col_del_right))
        col_del_pixels = np.concatenate(col_del_pixels, axis=0).astype(np.int16)
        ssr_imgs = np.delete(ssr_imgs, col_del_pixels, axis=1)  # 删除重叠列
    # 行删除
    if cropsize < cropsize * n_h:  # 判断行方向是否有多个(>2)图像块
        if overlapixel % 2 == 0:  # 获取行方向图像块重叠
            row_del_up = row_del_down = int(overlapixel / 2)
        else:
            row_del_up = int(np.ceil(overlapixel / 2.))
            row_del_down = int(np.floor(overlapixel / 2.))
        row_del_nodes = np.linspace(cropsize, cropsize*n_h, n_h)[:-1]  # 获取行方向图像块重叠像元, 待删除
        row_del_pixels = []
        for i in range(len(row_del_nodes)):
            row_del_pixels.append(np.arange(row_del_nodes[i] - row_del_up, row_del_nodes[i] + row_del_down))
        row_del_pixels = np.concatenate(row_del_pixels, axis=0).astype(np.int16)
        ssr_imgs = np.delete(ssr_imgs, row_del_pixels, axis=0)  # 删除重叠列
    assert ssr_imgs.shape[0] == h_te and ssr_imgs.shape[1] == w_te, "拼接图与原图padding尺度不相等"
    # 删除padding
    if h_pad > 0 and w_pad > 0:
        ssr_imgs = ssr_imgs[:-h_pad, :-w_pad, :]
    elif h_pad > 0 and w_pad <= 0:
        ssr_imgs = ssr_imgs[:-h_pad, :, :]
    elif h_pad <= 0 and w_pad > 0:
        ssr_imgs = ssr_imgs[:, :-w_pad, :]
    else:
        ssr_imgs = ssr_imgs
    assert ssr_imgs.shape[0] == h_ori and ssr_imgs.shape[1] == w_ori, "拼接图与原图尺度不相等"
    ssr_imgs_merge = ssr_imgs  # 合并图像
    ssr_imgs_split = ssr_imgs_numpy  # 分块图像

    return ssr_imgs_split, ssr_imgs_merge, h_pad, w_pad


def subImgMerge(sub_imgs_numpy, info, scale=2):
    # 获取图像块的分割信息
    cropsize = round(info.cropsize / scale)  # 图像分块后, 正方形子图大小, round四舍五入取整, int向下取整
    overlapixel = round(info.overlapixel / scale)  # 图像分块后, 每个分块图像重叠数
    n_h = info.n_h  # 图像分块行数
    n_w = info.n_w  # 图像分块列数
    h_pad = round(info.h_te_pad / scale)  # 原图像行方向padding数
    w_pad = round(info.w_te_pad / scale)  # 原图像列方向padding数
    h_ori = round(info.h_te_bef / scale)  # 原图像行数
    w_ori = round(info.w_te_bef / scale)  # 原图像列数
    h_te = round(info.h_te / scale)  # 原图像padding后行数
    w_te = round(info.w_te / scale)  # 原图像padding后列数
    # 图像块拼接
    idx = 0
    sub_imgs_row = []
    for r in range(n_h):  # 遍历行
        sub_imgs_col = []
        for c in range(n_w):  # 遍历列
            sub_imgs_col.append(sub_imgs_numpy[idx])
            idx = idx + 1
        # SSR图像块列方向拼接, 并放入list
        sub_imgs_row.append(np.concatenate(sub_imgs_col, axis=1))
    # SSR图像块列方向拼接后的结果行方向拼接
    sub_imgs = np.concatenate(sub_imgs_row,  axis=0)
    # 以及图像块重叠尺度(overlapixel)还原测试图像
    # 列删除
    if cropsize < cropsize * n_w:  # 判断列方向是否有多个(>2)图像块
        if overlapixel % 2 == 0:  # 获取列方向图像块重叠
            col_del_left = col_del_right = int(overlapixel / 2)
        else:
            col_del_left = int(np.ceil(overlapixel / 2.))
            col_del_right = int(np.floor(overlapixel / 2.))
        col_del_nodes = np.linspace(cropsize, cropsize * n_w, n_w)[:-1]  # 获取列方向图像块重叠像元, 待删除
        col_del_pixels = []
        for i in range(len(col_del_nodes)):
            col_del_pixels.append(np.arange(col_del_nodes[i] - col_del_left, col_del_nodes[i] + col_del_right))
        col_del_pixels = np.concatenate(col_del_pixels, axis=0).astype(np.int16)
        sub_imgs = np.delete(sub_imgs, col_del_pixels, axis=1)  # 删除重叠列
    # 行删除
    if cropsize < cropsize * n_h:  # 判断行方向是否有多个(>2)图像块
        if overlapixel % 2 == 0:  # 获取行方向图像块重叠
            row_del_up = row_del_down = int(overlapixel / 2)
        else:
            row_del_up = int(np.ceil(overlapixel / 2.))
            row_del_down = int(np.floor(overlapixel / 2.))
        row_del_nodes = np.linspace(cropsize, cropsize * n_h, n_h)[:-1]  # 获取行方向图像块重叠像元, 待删除
        row_del_pixels = []
        for i in range(len(row_del_nodes)):
            row_del_pixels.append(np.arange(row_del_nodes[i] - row_del_up, row_del_nodes[i] + row_del_down))
        row_del_pixels = np.concatenate(row_del_pixels, axis=0).astype(np.int16)
        sub_imgs = np.delete(sub_imgs, row_del_pixels, axis=0)  # 删除重叠列
    assert sub_imgs.shape[0] == h_te and sub_imgs.shape[1] == w_te, "拼接图与原图padding尺度不相等"
    # 删除padding
    if h_pad > 0 and w_pad > 0:
        sub_imgs = sub_imgs[:-h_pad, :-w_pad, :]
    elif h_pad > 0 and w_pad <= 0:
        sub_imgs = sub_imgs[:-h_pad, :, :]
    elif h_pad <= 0 and w_pad > 0:
        sub_imgs = sub_imgs[:, :-w_pad, :]
    else:
        sub_imgs = sub_imgs
    assert sub_imgs.shape[0] == h_ori and sub_imgs.shape[1] == w_ori, "拼接图与原图尺度不相等"
    sub_imgs_merge = sub_imgs  # 合并图像
    sub_imgs_split = sub_imgs_numpy  # 分块图像
    return sub_imgs_split, sub_imgs_merge, n_h, n_w



# 计算 平均数，求和，长度
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# save model
def save_checkpoint(model_path, epoch, model, optimizer, lr, scale):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learn_rate': lr,
    }
    torch.save(state, os.path.join(model_path, 'best_model_{}.pth'.format('x'+str(scale))))

# learning rate
def adjust_learning_rate(epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    # lr = opt.lr * np.power(0.93, int(epoch / 10))

    return lr


# model training
def train(train_loader, optimizer, model, criterion, metric, epoch, opt, device):
    lr = adjust_learning_rate(epoch - 1, opt)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    model.train()
    loss_batch = []
    performance_batch = []
    train_bar = tqdm(train_loader)
    for iteration, batch in enumerate(train_bar, 1):
        input, target = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
        output = model(input)
        loss = criterion(output, target)
        performance = metric(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bar.set_description(
            desc='[epoch: %d] train_len: %d/%d loss:%.4f lr:%.2e' % (
                epoch, iteration, len(train_loader), loss.data, lr
            ))
        loss_batch.append(loss.data)
        performance_batch.append(performance.data)
    return torch.mean(torch.as_tensor(loss_batch)), torch.mean(torch.as_tensor(performance_batch)), lr
