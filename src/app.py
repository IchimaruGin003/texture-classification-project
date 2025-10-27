"""
纹理分类主应用
"""
import os
import argparse
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
    
    args = parser.parse_args()
    
    if args.process_data or args.all:
        print("=== 处理数据 ===")
        processor = TextureDataProcessor()
        processor.process_texture_images()
    
    if args.extract_features or args.all:
        print("=== 提取特征 ===")
        extractor = GLCMFeatureExtractor()
        extractor.extract_and_save_features("data/glcm_features.csv")
    
    if args.train_model or args.all:
        print("=== 训练模型 ===")
        trainer = KNNTrainer(n_neighbors=3)
        X_train, X_test, y_train, y_test, feature_cols = trainer.load_data("data/glcm_features.csv")
        trainer.train(X_train, y_train)
        trainer.save_model("models/knn_model.joblib")
        
        if args.evaluate or args.all:
            print("=== 评估模型 ===")
            accuracy, y_pred = trainer.evaluate(X_test, y_test)
            print(f"最终准确率: {accuracy:.4f}")
            
            # 特征重要性分析
            trainer.feature_importance(X_test, y_test, feature_cols)

if __name__ == "__main__":
    main()