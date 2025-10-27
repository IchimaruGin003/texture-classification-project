import pytest
import os
import numpy as np
from PIL import Image
from src.processing.data_processor import TextureDataProcessor

class TestDataQuality:
    def test_raw_data_structure(self):
        """测试原始数据结构是否正确"""
        processor = TextureDataProcessor()
        
        # 检查原始数据目录是否存在
        assert os.path.exists(processor.raw_data_dir), "原始数据目录不存在"
        
        # 检查每个类别目录
        texture_classes = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        for class_name in texture_classes:
            class_dir = os.path.join(processor.raw_data_dir, class_name)
            assert os.path.exists(class_dir), f"类别目录 {class_name} 不存在"
            
            # 检查是否有TIF文件
            tif_files = [f for f in os.listdir(class_dir) if f.endswith('.tif')]
            assert len(tif_files) >= 16, f"类别 {class_name} 图像数量不足16张"

    def test_image_loading(self, tmp_path):
        """测试图像加载功能"""
        # 创建一个测试图像
        test_image_path = tmp_path / "test.png"
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        Image.fromarray(test_image).save(test_image_path)
        
        from src.features.glcm_extractor import GLCMFeatureExtractor
        extractor = GLCMFeatureExtractor()
        
        # 测试图像加载
        loaded_image = extractor.load_image(str(test_image_path))
        assert loaded_image is not None, "图像加载失败"
        assert loaded_image.shape == (100, 100), "图像尺寸不正确"

    def test_glcm_feature_extraction(self):
        """测试GLCM特征提取"""
        from src.features.glcm_extractor import extract_glcm_features
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # 提取特征
        features = extract_glcm_features(test_image)
        
        # 检查特征是否存在
        expected_features = ['energy', 'contrast', 'correlation', 'entropy']
        for feature in expected_features:
            assert feature in features, f"特征 {feature} 缺失"
            assert isinstance(features[feature], (int, float)), f"特征 {feature} 类型不正确"

    def test_empty_image_handling(self):
        """测试空图像处理"""
        from src.features.glcm_extractor import extract_glcm_features
        
        # 创建空图像
        empty_image = np.array([])
        
        # 应该能够优雅地处理空图像
        with pytest.raises(Exception):
            extract_glcm_features(empty_image)

    def test_model_training_smoke_test(self):
        """模型训练冒烟测试"""
        from src.models.knn_trainer import KNNTrainer
        
        # 创建模拟数据
        X_train = np.random.rand(10, 4)
        y_train = ['D1'] * 5 + ['D2'] * 5
        X_test = np.random.rand(5, 4)
        
        trainer = KNNTrainer(n_neighbors=3)
        trainer.train(X_train, y_train)
        
        # 测试预测
        predictions = trainer.model.predict(X_test)
        assert len(predictions) == 5, "预测结果数量不正确"