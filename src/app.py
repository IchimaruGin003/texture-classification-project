"""
纹理分类主应用 - 更新版本支持多环境配置
"""
import os
import argparse
from src.config import get_config
from src.processing.data_processor import TextureDataProcessor
from src.features.glcm_extractor import GLCMFeatureExtractor
from src.models.knn_trainer import KNNTrainer

def main():
    parser = argparse.ArgumentParser(description="纹理分类应用")
    parser.add_argument("--process-data", action="store_true", help="处理原始数据")
    parser.add_argument("--extract-features", action="store_true", help="提取GLCM特征")
    parser.add_argument("--train-model", action="store_true", help="训练KNN模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--all", action="store_true", help="运行完整流程")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="development", help="运行环境")
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.environment)
    config.validate()
    
    print(f"=== 运行环境: {config.ENVIRONMENT} ===")
    print(f"DEBUG: {config.DEBUG}, LOG_LEVEL: {config.LOG_LEVEL}")
    
    if args.process_data or args.all:
        print("=== 处理数据 ===")
        processor = TextureDataProcessor(base_dir=config.DATA_BASE_DIR)
        processor.process_texture_images()
    
    if args.extract_features or args.all:
        print("=== 提取特征 ===")
        extractor = GLCMFeatureExtractor(base_dir=config.DATA_BASE_DIR)
        features_path = os.path.join(config.DATA_BASE_DIR, "glcm_features.csv")
        extractor.extract_and_save_features(features_path)
    
    if args.train_model or args.all:
        print("=== 训练模型 ===")
        trainer = KNNTrainer(n_neighbors=config.KNN_NEIGHBORS)
        features_path = os.path.join(config.DATA_BASE_DIR, "glcm_features.csv")
        X_train, X_test, y_train, y_test, feature_cols = trainer.load_data(features_path)
        trainer.train(X_train, y_train)
        
        model_path = os.path.join("models", "knn_model.joblib")
        trainer.save_model(model_path)
        
        if args.evaluate or args.all:
            print("=== 评估模型 ===")
            accuracy, y_pred = trainer.evaluate(X_test, y_test)
            print(f"最终准确率: {accuracy:.4f}")
            
            # 特征重要性分析
            trainer.feature_importance(X_test, y_test, feature_cols)

if __name__ == "__main__":
    main()