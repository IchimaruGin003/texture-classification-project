import pytest
import os
import tempfile
import numpy as np
from PIL import Image
from src.processing.data_processor import TextureDataProcessor, add_salt_pepper_noise


def test_add_salt_pepper_noise():
    # 创建一个全灰的图像
    image = np.full((10, 10), 128, dtype=np.uint8)
    noisy_image = add_salt_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1)

    # 检查是否有白点和黑点
    assert np.any(noisy_image == 255)  # 盐噪声
    assert np.any(noisy_image == 0)  # 椒噪声
    assert np.any(noisy_image == 128)  # 保留部分原像素


def test_texture_data_processor_init():
    processor = TextureDataProcessor(base_dir="test_data")
    assert processor.base_dir == "test_data"
    assert processor.raw_data_dir == os.path.join("test_data", "raw_data")
    assert processor.processed_dir == os.path.join("test_data", "processed")
