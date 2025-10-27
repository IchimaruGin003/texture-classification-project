"""
GLCM特征提取模块
"""
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def manual_glcm(image, distance=1, angle=0):
    """
    手动计算灰度共生矩阵
    """
    # 确保图像是uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # 角度转换为偏移量
    if angle == 0:    dx, dy = distance, 0
    elif angle == 45: dx, dy = distance, distance
    elif angle == 90: dx, dy = 0, distance
    elif angle == 135: dx, dy = -distance, distance
    else: raise ValueError("角度必须是0, 45, 90, 135度")
    
    glcm = np.zeros((256, 256), dtype=np.float64)
    height, width = image.shape
    
    for i in range(height):
        for j in range(width):
            i2, j2 = i + dy, j + dx
            if 0 <= i2 < height and 0 <= j2 < width:
                gray1, gray2 = image[i, j], image[i2, j2]
                glcm[gray1, gray2] += 1
    
    if np.sum(glcm) > 0:
        glcm /= np.sum(glcm)
    
    return glcm

def calculate_contrast(glcm):
    return np.sum([glcm[i, j] * (i - j) ** 2 for i in range(256) for j in range(256)])

def calculate_energy(glcm):
    return np.sum(glcm ** 2)

def calculate_correlation(glcm):
    i_idx, j_idx = np.indices(glcm.shape)
    mean_i, mean_j = np.sum(i_idx * glcm), np.sum(j_idx * glcm)
    std_i = np.sqrt(np.sum((i_idx - mean_i) ** 2 * glcm))
    std_j = np.sqrt(np.sum((j_idx - mean_j) ** 2 * glcm))
    return np.sum((i_idx - mean_i) * (j_idx - mean_j) * glcm) / (std_i * std_j) if std_i * std_j != 0 else 0

def calculate_entropy(glcm):
    return -np.sum([glcm[i, j] * np.log2(glcm[i, j]) for i in range(256) for j in range(256) if glcm[i, j] > 0])

def extract_glcm_features(image):
    """提取GLCM特征 - 4个方向平均值"""
    features = {}
    angles = [0, 45, 90, 135]
    
    metrics = {'energy': [], 'contrast': [], 'correlation': [], 'entropy': []}
    
    for angle in angles:
        glcm = manual_glcm(image, distance=1, angle=angle)
        metrics['energy'].append(calculate_energy(glcm))
        metrics['contrast'].append(calculate_contrast(glcm))
        metrics['correlation'].append(calculate_correlation(glcm))
        metrics['entropy'].append(calculate_entropy(glcm))
    
    for key in metrics:
        features[key] = np.mean(metrics[key])
    
    return features

class GLCMFeatureExtractor:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
    
    def load_image(self, image_path):
        """加载图像"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.array(Image.open(image_path).convert('L'))
            return image
        except Exception as e:
            print(f"无法读取图像 {image_path}: {e}")
            return None

    def process_images(self):
        """处理所有图像并提取特征"""
        features_list = []
        
        for class_num in range(1, 7):
            class_name = f"D{class_num}"
            
            # 处理训练集和测试集
            for set_type in ['train', 'test']:
                img_dir = os.path.join(self.processed_dir, f"{class_name}_{set_type}")
                
                if os.path.exists(img_dir):
                    for filename in os.listdir(img_dir):
                        if filename.endswith('.png'):
                            image_path = os.path.join(img_dir, filename)
                            image = self.load_image(image_path)
                            
                            if image is not None:
                                try:
                                    features = extract_glcm_features(image)
                                    features['filename'] = filename
                                    features['class'] = class_name
                                    features['set_type'] = set_type
                                    features_list.append(features)
                                    print(f"处理: {class_name}_{set_type}/{filename}")
                                except Exception as e:
                                    print(f"处理 {filename} 时出错: {e}")
        
        return pd.DataFrame(features_list)

    def extract_and_save_features(self, output_path="glcm_features.csv"):
        """提取特征并保存到CSV"""
        print("开始GLCM特征提取...")
        features_df = self.process_images()
        
        if len(features_df) == 0:
            print("没有找到任何图像文件，请检查路径设置")
            return None
        
        features_df.to_csv(output_path, index=False)
        print(f"特征已保存到: {output_path}")
        print(f"总共处理了 {len(features_df)} 张图像")
        return features_df