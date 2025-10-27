import pytest
import numpy as np
from src.features.glcm_extractor import manual_glcm, calculate_contrast

class TestEdgeCases:
    def test_single_color_image(self):
        """测试单色图像的GLCM计算"""
        # 全黑图像
        black_image = np.zeros((10, 10), dtype=np.uint8)
        glcm = manual_glcm(black_image)
        
        # 单色图像的对比度应该为0
        contrast = calculate_contrast(glcm)
        assert contrast == 0.0, "单色图像对比度应该为0"
        
        # 全白图像
        white_image = np.full((10, 10), 255, dtype=np.uint8)
        glcm_white = manual_glcm(white_image)
        contrast_white = calculate_contrast(glcm_white)
        assert contrast_white == 0.0, "单色图像对比度应该为0"

    def test_small_image_handling(self):
        """测试小图像处理"""
        # 1x1 图像
        tiny_image = np.array([[128]], dtype=np.uint8)
        glcm = manual_glcm(tiny_image)
        
        # 应该能够处理但特征可能有限
        assert glcm.shape == (256, 256), "GLCM矩阵尺寸不正确"

    def test_extreme_noise_images(self):
        """测试极端噪声图像"""
        from src.processing.data_processor import add_salt_pepper_noise
        
        # 高噪声图像
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        noisy_image = add_salt_pepper_noise(image, salt_prob=0.5, pepper_prob=0.5)
        
        # 检查噪声是否添加成功
        salt_pixels = np.sum(noisy_image == 255)
        pepper_pixels = np.sum(noisy_image == 0)
        assert salt_pixels > 0 or pepper_pixels > 0, "噪声添加失败"

    def test_invalid_parameters(self):
        """测试无效参数处理"""
        from src.features.glcm_extractor import manual_glcm
        
        test_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        
        # 测试无效角度
        with pytest.raises(ValueError):
            manual_glcm(test_image, angle=180)