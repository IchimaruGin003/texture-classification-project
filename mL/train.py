"""
MLflow集成训练脚本
"""
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.glcm_extractor import GLCMFeatureExtractor
from src.models.knn_trainer import KNNTrainer
from src.config import get_config

def train_baseline_model():
    """训练基线模型 - 使用v1数据"""
    print("=== 训练基线模型（v1数据）===")
    
    # 设置实验
    mlflow.set_experiment("Texture_Classification_Baseline")
    
    with mlflow.start_run(run_name="baseline_knn_v1"):
        # 记录代码版本
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            mlflow.log_param("git_commit", git_hash)
        except:
            print("无法获取git提交哈希")
        
        # 记录数据集版本
        mlflow.log_param("dataset_version", "v1")
        mlflow.log_param("data_path", "data/processed")
        
        # 模型参数
        n_neighbors = 3
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("model_type", "KNeighborsClassifier")
        
        print("提取v1数据特征...")
        # 提取v1数据特征
        extractor = GLCMFeatureExtractor()
        features_df = extractor.extract_and_save_features("data/glcm_features_baseline.csv")
        
        print("训练基线模型...")
        # 训练模型
        trainer = KNNTrainer(n_neighbors=n_neighbors)
        X_train, X_test, y_train, y_test, feature_cols = trainer.load_data("data/glcm_features_baseline.csv")
        trainer.train(X_train, y_train)
        
        # 评估模型
        accuracy, predictions = trainer.evaluate(X_test, y_test)
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"基线模型训练完成，准确率: {accuracy:.4f}")
        
        # 记录模型
        mlflow.sklearn.log_model(trainer.model, "baseline_knn_model")
        
        # 保存模型文件
        trainer.save_model("models/baseline_knn_model.joblib")
        mlflow.log_artifact("models/baseline_knn_model.joblib")
        
        return accuracy

def train_improved_model():
    """训练改进模型 - 使用v2数据"""
    print("=== 训练改进模型（v2数据）===")
    
    # 设置实验
    mlflow.set_experiment("Texture_Classification_Improved")
    
    with mlflow.start_run(run_name="improved_knn_v2"):
        # 记录代码版本
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            mlflow.log_param("git_commit", git_hash)
        except:
            print("无法获取git提交哈希")
        
        # 记录数据集版本
        mlflow.log_param("dataset_version", "v2")
        mlflow.log_param("data_path", "data/processed_v2")
        
        # 改进的模型参数
        n_neighbors = 5  # 调整邻居数
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("model_type", "KNeighborsClassifier")
        mlflow.log_param("improvement_note", "Increased n_neighbors for better generalization")
        
        print("提取v2数据特征...")
        # 使用v2数据提取特征
        extractor = GLCMFeatureExtractor(base_dir="data")
        # 修改GLCM提取器使用v2数据
        extractor.processed_dir = "data/processed_v2"
        features_df = extractor.extract_and_save_features("data/glcm_features_improved.csv")
        
        print("训练改进模型...")
        # 训练改进模型
        trainer = KNNTrainer(n_neighbors=n_neighbors)
        X_train, X_test, y_train, y_test, feature_cols = trainer.load_data("data/glcm_features_improved.csv")
        trainer.train(X_train, y_train)
        
        # 评估模型
        accuracy, predictions = trainer.evaluate(X_test, y_test)
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"改进模型训练完成，准确率: {accuracy:.4f}")
        
        # 记录模型
        mlflow.sklearn.log_model(trainer.model, "improved_knn_model")
        
        # 保存模型文件
        trainer.save_model("models/improved_knn_model.joblib")
        mlflow.log_artifact("models/improved_knn_model.joblib")
        
        return accuracy

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("开始MLflow实验...")
    
    # 运行两个实验
    baseline_acc = train_baseline_model()
    improved_acc = train_improved_model()
    
    print(f"\n=== 实验结果总结 ===")
    print(f"基线模型准确率: {baseline_acc:.4f}")
    print(f"改进模型准确率: {improved_acc:.4f}")
    print(f"改进幅度: {improved_acc - baseline_acc:.4f}")
    
    # 启动MLflow UI查看结果
    print("\n启动MLflow UI查看详细结果:")
    print("mlflow ui --port 5000")
    print("然后在浏览器中访问 http://localhost:5000")