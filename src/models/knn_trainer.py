"""
KNN模型训练和评估模块
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib


class KNNTrainer:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False

    def load_data(self, features_csv_path):
        """加载特征数据"""
        df = pd.read_csv(features_csv_path)

        # 提取特征列和标签列
        feature_cols = ["energy", "contrast", "correlation", "entropy"]
        X = df[feature_cols].values
        y = df["class"].values
        set_type = df["set_type"].values

        # 分离训练集和测试集
        X_train = X[set_type == "train"]
        y_train = y[set_type == "train"]
        X_test = X[set_type == "test"]
        y_test = y[set_type == "test"]

        return X_train, X_test, y_train, y_test, feature_cols

    def train(self, X_train, y_train):
        """训练模型"""
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练KNN分类器
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        print("模型训练完成！")

    def evaluate(self, X_test, y_test):
        """评估模型"""
        if not self.is_trained:
            raise Exception("模型尚未训练，请先调用train方法")

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"分类准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))

        return accuracy, y_pred

    def feature_importance(self, X_test, y_test, feature_cols):
        """计算特征重要性"""
        X_test_scaled = self.scaler.transform(X_test)

        perm_importance = permutation_importance(
            self.model, X_test_scaled, y_test, n_repeats=10, random_state=42
        )

        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": perm_importance.importances_mean}
        ).sort_values(by="importance", ascending=False)

        print("特征重要性（置换重要性）：")
        print(importance_df)

        # 可视化
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("置换重要性得分（准确率下降值）")
        plt.ylabel("特征")
        plt.title("GLCM特征对KNN分类结果的贡献")
        plt.gca().invert_yaxis()
        plt.show()

        return importance_df

    def save_model(self, model_path="models/knn_model.joblib"):
        """保存模型"""
        if not self.is_trained:
            raise Exception("模型尚未训练，无法保存")

        joblib.dump({"model": self.model, "scaler": self.scaler}, model_path)
        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path="models/knn_model.joblib"):
        """加载模型"""
        loaded = joblib.load(model_path)
        self.model = loaded["model"]
        self.scaler = loaded["scaler"]
        self.is_trained = True
        print(f"模型已从 {model_path} 加载")
