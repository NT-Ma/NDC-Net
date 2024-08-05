import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim, MS_SSIM, ms_ssim, SSIM

import logging

# 导入自定义的模块和网络结构
import dataloader
import NDCNet as net

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_plot_loss(config):
    """训练模型并绘制损失图，同时支持检查点的保存和加载"""
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.apply(weights_init)

    # 数据加载
    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, 'train')
    test_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, 'test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False,
                                              num_workers=config.num_workers, pin_memory=True)

    # 损失函数和优化器
    mse_loss = nn.MSELoss().cuda()
    ssim_loss = SSIMLoss().cuda()

    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 初始化指标
    train_losses, train_ssim, train_psnr = [], [], []
    test_losses, test_ssim, test_psnr = [], [], []
    best_test_loss = float('inf')
    best_epoch = 0

    # 尝试加载最新的检查点
    checkpoint_path = os.path.join(config.snapshots_folder, 'latest_checkpoint.pth')
    dehaze_net, optimizer, start_epoch, best_test_loss = load_checkpoint(checkpoint_path, dehaze_net, optimizer)

    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        dehaze_net.train()
        train_loss, ssim_score, psnr_score = 0.0, 0.0, 0.0

        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig, img_haze = img_orig.cuda(), img_haze.cuda()
            clean_image = dehaze_net(img_haze)
            mse = mse_loss(clean_image, img_orig)
            ssim = ssim_loss(clean_image, img_orig)
            loss = 0.8 * mse + 0.2 * ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 计算SSIM和PSNR
            batch_ssim, batch_psnr = 0.0, 0.0
            for i in range(img_orig.size(0)):
                single_orig = img_orig[i].cpu().detach().squeeze().numpy()
                single_clean = clean_image[i].cpu().detach().squeeze().numpy()
                batch_ssim +=  ssim_skimage(single_orig, single_clean, win_size=3, data_range=1, multichannel=False)
                batch_psnr += psnr(single_orig, single_clean)
            ssim_score += batch_ssim / img_orig.size(0)
            psnr_score += batch_psnr / img_orig.size(0)

            # 记录到日志
            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch + 1}/{config.num_epochs}], Iteration [{iteration + 1}], Loss: {loss.item():.4f}")

        train_losses.append(train_loss / len(train_loader))
        train_ssim.append(ssim_score / len(train_loader))
        train_psnr.append(psnr_score / len(train_loader))

        # 评估模式
        dehaze_net.eval()
        test_loss, ssim_test, psnr_test = 0.0, 0.0, 0.0
        with torch.no_grad():
            for img_orig, img_haze in test_loader:
                img_orig, img_haze = img_orig.cuda(), img_haze.cuda()
                clean_image = dehaze_net(img_haze)
                mse = mse_loss(clean_image, img_orig)

                ssim = ssim_loss(clean_image, img_orig)
                loss = 0.8 * mse + 0.2 * ssim

                test_loss += loss.item()
                batch_ssim, batch_psnr = 0.0, 0.0
                for i in range(img_orig.size(0)):
                    single_orig = img_orig[i].cpu().detach().squeeze().numpy()
                    single_clean = clean_image[i].cpu().detach().squeeze().numpy()
                    batch_ssim += ssim_skimage(single_orig, single_clean, win_size=3, data_range=1, multichannel=False)
                    batch_psnr += psnr(single_orig, single_clean)
                ssim_test += batch_ssim / img_orig.size(0)
                psnr_test += batch_psnr / img_orig.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_ssim.append(ssim_test / len(test_loader))
        test_psnr.append(psnr_test / len(test_loader))

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save({
                'state_dict': dehaze_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_loss': best_test_loss,
                'epoch': epoch
            }, checkpoint_path)

        torch.save(dehaze_net.state_dict(), os.path.join(config.snapshots_folder, f"dehaze_net_epoch_{epoch + 1}.pth"))

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Train SSIM: {train_ssim[-1]:.4f}, Train PSNR: {train_psnr[-1]:.4f}")
        print(f"Epoch {epoch + 1}: Test Loss: {test_losses[-1]:.4f}, Test SSIM: {test_ssim[-1]:.4f}, Test PSNR: {test_psnr[-1]:.4f}")
        print(f"Epoch duration: {(epoch_end_time - epoch_start_time) / 60:.2f} minutes")

        # 日志记录
        logging.info(
            f"Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Train SSIM: {train_ssim[-1]:.4f}, Train PSNR: {train_psnr[-1]:.4f}")
        logging.info(
            f"Epoch {epoch + 1}: Test Loss: {test_losses[-1]:.4f}, Test SSIM: {test_ssim[-1]:.4f}, Test PSNR: {test_psnr[-1]:.4f}")
        logging.info(f"Epoch duration: {(time.time() - epoch_start_time) / 60:.2f} minutes")

    total_duration = (time.time() - start_time) / 60
    print(f"Total training time: {total_duration:.2f} minutes")
    print(f"Best model was from epoch {best_epoch} with a test loss of {best_test_loss:.4f}")
    print("Train_loss", train_losses)
    print("Test_loss", test_losses)
    
if __name__ == "__main__":

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train_and_plot_loss(config)
