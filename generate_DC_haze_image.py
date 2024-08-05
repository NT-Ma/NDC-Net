import numpy as np
import cv2
import os

def dark_channel_prior(img, window_size):
    min_channel = np.min(img, axis=2)  # 选择三个通道中的最小值
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size), dtype=np.uint8))  # 使用腐蚀操作模拟最小滤波
    return dark_channel

def estimate_atmospheric_light(img, dark_channel, percent):
    num_pixels = dark_channel.size
    top_pixels = int(percent * num_pixels)  # 计算顶部百分比的像素数

    # 找到暗通道中最亮的像素位置
    indices = np.argsort(dark_channel, axis=None)[-top_pixels:]

    # 创建一个全黑的图像作为掩模
    mask = np.zeros_like(img, dtype=np.uint8)

    # 在掩模上标记选中的最亮位置的像素
    flat_image = img.reshape((-1, 3))
    brightest_pixels = flat_image[indices]
    mask.reshape((-1, 3))[indices] = brightest_pixels  # 把原图中最亮的像素按位置填回新图

    # 对掩模图像进行腐蚀操作，消除小区域的白色
    eroded_mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))

    # 检查腐蚀后的掩模是否全为0
    if np.all(eroded_mask == 0):
        eroded_mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
        use_mask = eroded_mask
        if np.all(eroded_mask == 0):
            use_mask = mask
    else:
        use_mask = eroded_mask

    # 找出使用掩模中最大像素值及其位置
    max_val = np.max(use_mask)
    max_position = np.unravel_index(np.argmax(use_mask, axis=None), use_mask.shape)
    max_color = use_mask[max_position[0], max_position[1], :]

    return max_color

# 测试代码
if __name__ == "__main__":
    # input_folder = 'hazyImage_folder'
    # output_folder = 'DC_image_folder'
    input_folder = 'test_image'
    output_folder = 'DC_image'
    filenames = os.listdir(input_folder)
    for filename in filenames:
        clear_file_path = os.path.join(input_folder, filename)
        clear = cv2.imread(clear_file_path)

        dark_channel = dark_channel_prior(clear, 15)
        atmospheric_light = estimate_atmospheric_light(clear, dark_channel, 0.001)  # 选用最亮的1%的像素

        image_normalized = clear / 255.0
        atmospheric_light_normalized = atmospheric_light / 255.0
        dark_channel_light = dark_channel_prior(image_normalized / atmospheric_light_normalized, 15)

        # 确保dark_channel_light中的值不会超过1
        dark_channel_light = np.clip(dark_channel_light, 0, 1)

        dark_channel_light_scaled = (dark_channel_light * 255).astype(np.uint8)

        output_path = os.path.join(output_folder, filename)
        # 保存结果
        cv2.imwrite(output_path, dark_channel_light_scaled)

