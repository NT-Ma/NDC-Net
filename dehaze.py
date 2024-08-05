import numpy as np
import cv2
import os

def dark_channel_prior(img, window_size):
    height, width = img.shape[0:2]
    # 三通道最小
    min_channel = np.min(img, axis=2)

    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size), dtype=np.uint8))

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
        print("All zero post-erosion, using original mask.")
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

    print("Max pixel position:", max_position)
    print("Max pixel value:", max_color)

    return max_color


def dehaze(img, atmospheric_light, clear_image, t0=0.1):
    image_normalized = img.astype(np.float32) / 255.0
    atmospheric_light_normalized = atmospheric_light / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    transmission_DCP = (1 - dark_channel_prior(img / atmospheric_light, window_size=15))

    # 计算透射率
    transmission =  (1 - dark_channel_prior(img / atmospheric_light, window_size=15)) / (1 - clear_image / 255 + 1e-6)

    # 处理除以零或无效值的情况
    transmission[np.isnan(transmission)] = t0
    transmission[np.isinf(transmission)] = t0
    transmission_DCP[np.isnan(transmission_DCP)] = t0
    transmission_DCP[np.isinf(transmission_DCP)] = t0

    # 限制透射率的范围
    transmission = np.maximum(transmission, t0)
    transmission_DCP = np.maximum(transmission_DCP, t0)

    transmission = np.clip(transmission, 0, 1)
    transmission_DCP = np.clip(transmission_DCP, 0, 1)

    # 融合的权重
    transmission_final = 0.7 * transmission + 0.3 * transmission_DCP
    # transmission_final = transmission

    transmission_final = np.minimum(transmission_final * 255, 255).astype(np.uint8)

    transmission_final = cv2.ximgproc.guidedFilter(gray, transmission_final, 30, 0.01)

    transmission = transmission_final.astype(np.float32) / 255.0
    # 恢复原始图像
    dehazed_image = np.zeros_like(image_normalized)
    for i in range(3):
        # 处理除以零或无效值的情况
        transmission[transmission == 0] = 1  # 避免除以零
        dehazed_image[:, :, i] = (image_normalized[:, :, i] - atmospheric_light_normalized[i]) / transmission + \
                                 atmospheric_light_normalized[i]

    # 处理无效值
    dehazed_image[np.isnan(dehazed_image)] = 0
    dehazed_image[np.isinf(dehazed_image)] = 255

    dehazed_image = np.clip(dehazed_image, 0, 1) * 255.0
    dehazed_image = dehazed_image.astype(np.uint8)

    return dehazed_image


def process_all_images(input_folder, output_folder, clear_img_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图像
    for file in os.listdir(input_folder):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 构建清晰图像的路径
            name, _ = os.path.splitext(file)
            if file.endswith(".png"):
                clear_img_path = os.path.join(clear_img_folder, name + ".png")
            if file.endswith('.jpeg'):
                clear_img_path = os.path.join(clear_img_folder, name + ".jpeg")
            clear_img = cv2.imread(clear_img_path, cv2.IMREAD_GRAYSCALE)

            if clear_img is None:
                continue

            print(f"Processing {file}...")

            # 应用暗通道先验
            dark_channel = dark_channel_prior(img, 15)
            light = estimate_atmospheric_light(img, dark_channel, 0.001)
            dehazed = dehaze(img, light, clear_img)

            # 保存去雾后的图像
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, dehazed)

def main():
    input_folder = "test_image"
    output_folder = "result"
    DC_clear_image_folder = "DC_image/DC_clear_image"

    process_all_images(input_folder, output_folder, DC_clear_image_folder)

if __name__ == "__main__":
    main()
