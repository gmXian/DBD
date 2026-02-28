import numpy as np
import cv2
import io
import torch

from dataset.augmix_ops import augmentations
from dataset.datautils import augmix
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance

##### spatial transform
def random_crop_resize_flip(img, size=224, scale=(0.5, 1.0), ratio=(3./4., 4./3.)):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(),
    ])
    return transform(img)

##### pixel/color transform
def bit_depth_reduction(img, bits=3, quantize='round'):
    assert bits >= 1 and bits <= 8
    interval = 2 ** (8 - bits)
    img_np = np.array(img).astype(np.float32) / interval
    if quantize == 'floor':
        img_quantized = (np.floor(img_np) * interval).astype(np.uint8)
    elif quantize == 'round':
        img_quantized = (np.round(img_np) * interval).astype(np.uint8)
    elif quantize == 'ceil':
        img_quantized = (np.ceil(img_np) * interval).astype(np.uint8)
    else:
        raise ValueError("quantize should be 'floor', 'round' or 'ceil'")

    img_quantized = Image.fromarray(img_quantized)
    return img_quantized

def jpeg_compression(img, quality=75):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    img_jpeg = Image.open(buffered)
    return img_jpeg


def add_random_noise(img, sigma=0.1):
    img_np = np.array(img).astype(np.float32) / 255.0
    noise = np.random.randn(*img_np.shape) * sigma
    img_noisy = np.clip(img_np + noise, 0, 1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    img_noisy = Image.fromarray(img_noisy)
    return img_noisy

def autocontrast(img):
  return ImageOps.autocontrast(img)

def equalize(img):
  return ImageOps.equalize(img)

def color_space_conversion(
    image,
    space='LAB',
    perturb=True,
    hsv_perturb=(0.1, 0.1),    # (saturation_scale, value_scale) —— 乘法扰动
    ycrcb_perturb=(10, 10),     # (Cr_shift, Cb_shift) —— 加法扰动
    lab_perturb=(5, 5)          # (A_shift, B_shift) —— 加法扰动
):
    """
    颜色空间防御：RGB → 目标空间 → （可选扰动）→ RGB
    用于削弱对抗攻击扰动

    参数：
        image: PIL.Image (RGB mode)
        space: str, in ['HSV', 'YCrCb', 'LAB']
        perturb: bool, 是否在目标空间加扰动
        hsv_perturb: (s_scale, v_scale), 如 (0.9, 1.1) 表示 ±10% 扰动
        ycrcb_perturb: (Cr_shift, Cb_shift), 如 (±10)
        lab_perturb: (A_shift, B_shift), 如 (±5)

    返回：
        defended_image: PIL.Image (RGB)
    """
    # 转为 numpy array (RGB)
    rgb = np.array(image)

    # Step 1: RGB → 目标空间
    if space == 'HSV':
        # OpenCV: RGB → HSV
        target = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Step 2: 扰动（乘法扰动 S 和 V）
        if perturb:
            s_scale = np.random.uniform(1 - hsv_perturb[0], 1 + hsv_perturb[0])
            v_scale = np.random.uniform(1 - hsv_perturb[1], 1 + hsv_perturb[1])
            target[..., 1] *= s_scale  # Saturation
            target[..., 2] *= v_scale  # Value
            target = np.clip(target, 0, 255)

        # Step 3: HSV → RGB
        rgb_out = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_HSV2RGB)

    elif space == 'YCrCb':
        # OpenCV: RGB → YCrCb（注意：是 YCrCb，不是 YCbCr，但通道顺序可调）
        target = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)

        # Step 2: 扰动 Cr, Cb（加法扰动）
        if perturb:
            cr_shift = np.random.uniform(-ycrcb_perturb[0], ycrcb_perturb[0])
            cb_shift = np.random.uniform(-ycrcb_perturb[1], ycrcb_perturb[1])
            target[..., 1] += cr_shift  # Cr
            target[..., 2] += cb_shift  # Cb
            target = np.clip(target, 0, 255)

        # Step 3: YCrCb → RGB
        rgb_out = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    elif space == 'LAB':
        # OpenCV: RGB → LAB
        target = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Step 2: 扰动 A, B 通道（加法扰动），保留 L 亮度
        if perturb:
            a_shift = np.random.uniform(-lab_perturb[0], lab_perturb[0])
            b_shift = np.random.uniform(-lab_perturb[1], lab_perturb[1])
            target[..., 1] += a_shift  # A
            target[..., 2] += b_shift  # B
            target = np.clip(target, 0, 255)

        # Step 3: LAB → RGB
        rgb_out = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_LAB2RGB)

    else:
        raise ValueError(f"Unsupported color space: {space}. Choose from ['HSV', 'YCrCb', 'LAB']")

    # 转回 PIL Image
    return Image.fromarray(rgb_out)


#### Frequency Domain Transformations
def median_filter(img, kernel_size=3):
    img_np = np.array(img)
    img_filtered = cv2.medianBlur(img_np, kernel_size)
    img_filtered = Image.fromarray(img_filtered)
    return img_filtered

def mean_filter(img, kernel_size=3):
    img_np = np.array(img)
    img_filtered = cv2.blur(img_np, (kernel_size, kernel_size))
    img_filtered = Image.fromarray(img_filtered)
    return img_filtered

def gaussian_filter(img, kernel_size=3, sigma=0):
    img_np = np.array(img)
    img_filtered = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
    img_filtered = Image.fromarray(img_filtered)
    return img_filtered

def _apply_fdr(channel, alpha, noise_type, perturb_phase, beta):
    # Step 1: 傅里叶变换 + 中心化
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    # Step 2: 分解幅度和相位
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    # Step 3: 随机扰动幅度谱
    if noise_type == 'uniform':
        noise = np.random.uniform(-1, 1, size=magnitude.shape)
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, 1, size=magnitude.shape)
    else:
        raise ValueError("noise_type must be 'uniform' or 'gaussian'")

    magnitude_perturbed = magnitude * (1 + alpha * noise)

    # Step 4: （可选）扰动相位谱
    if perturb_phase:
        phase_noise = np.random.uniform(-np.pi, np.pi, size=phase.shape)
        phase_perturbed = phase + beta * phase_noise
    else:
        phase_perturbed = phase

    # Step 5: 重建复数频谱
    fshift_perturbed = magnitude_perturbed * np.exp(1j * phase_perturbed)

    # Step 6: 逆变换
    f_ishift = np.fft.ifftshift(fshift_perturbed)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    return img_back

def frequency_domain_randomization(
    image,
    alpha=0.1,          # 幅度扰动强度 [0.05, 0.2]
    noise_type='uniform', # 'uniform' or 'gaussian'
    perturb_phase=False,  # 是否扰动相位（通常不推荐）
    beta=0.1            # 相位扰动强度
):
    img = np.array(image).astype(np.float32)
    is_rgb = len(img.shape) == 3

    if is_rgb:
        channels = []
        for i in range(3):
            channel = img[:, :, i]
            randomized = _apply_fdr(channel, alpha, noise_type, perturb_phase, beta)
            channels.append(randomized)
        result = np.stack(channels, axis=2)
    else:
        result = _apply_fdr(img, alpha, noise_type, perturb_phase, beta)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class Augmenter(object):
    def __init__(self, base_transform, preprocess):
        self.base_transform = base_transform
        self.preprocess = preprocess

        
    def __call__(self, img_pil):
        # img_origin = self.preprocess(self.base_transform(img_pil))  # 3,224,224, [0,1]
        
        imgs_spatial_transformed = [
            random_crop_resize_flip(img_pil, size=224, scale=(0.5, 1.0), ratio=(3./4., 4./3.))
            for _ in range(16)]
        
        imgs_pixel_transformed = [
            bit_depth_reduction(img_pil, bits=3, quantize=q) for q in ['floor', 'round', 'ceil']
        ] + [
            jpeg_compression(img_pil, quality=q) for q in [50, 60, 75]
        ] + [
            add_random_noise(img_pil, sigma=0.1) for _ in range(6)
        ]

        imgs_frequency_transformed = [
            median_filter(img_pil, kernel_size=5),
            mean_filter(img_pil, kernel_size=5),
            gaussian_filter(img_pil, kernel_size=5),
        ]

        imgs = [img_pil] + imgs_spatial_transformed + imgs_pixel_transformed + imgs_frequency_transformed
        imgs = [self.preprocess(self.base_transform(img)) for img in imgs]
        imgs = torch.stack(imgs)
        return imgs


