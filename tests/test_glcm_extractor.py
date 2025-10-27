import pytest
import numpy as np
from src.features.glcm_extractor import (
    manual_glcm,
    calculate_contrast,
    calculate_energy,
)


def test_manual_glcm():
    # 创建一个简单的测试图像
    image = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8
    )

    glcm = manual_glcm(image, distance=1, angle=0)

    # 检查GLCM的基本属性
    assert glcm.shape == (256, 256)
    assert np.allclose(np.sum(glcm), 1.0)  # 应该归一化


def test_glcm_metrics():
    # 创建一个简单的GLCM
    glcm = np.zeros((256, 256))
    glcm[0, 0] = 0.5
    glcm[1, 1] = 0.5

    contrast = calculate_contrast(glcm)
    energy = calculate_energy(glcm)

    assert contrast == 0.0  # 相同像素对比度为0
    assert energy == 0.5  # 0.5^2 + 0.5^2 = 0.5
