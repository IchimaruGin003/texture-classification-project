import pytest
import pandas as pd
import numpy as np
from src.models.knn_trainer import KNNTrainer


class TestIntegration:
    def test_end_to_end_workflow(self, tmp_path):
        """测试端到端工作流程"""
        # 创建模拟特征数据
        features_data = {
            "energy": [0.5, 0.6, 0.4, 0.7, 0.3],
            "contrast": [10.2, 8.5, 12.1, 9.8, 11.3],
            "correlation": [0.8, 0.9, 0.7, 0.85, 0.75],
            "entropy": [2.1, 1.8, 2.3, 1.9, 2.2],
            "class": ["D1", "D1", "D2", "D2", "D1"],
            "set_type": ["train", "train", "train", "test", "test"],
        }

        features_df = pd.DataFrame(features_data)
        csv_path = tmp_path / "test_features.csv"
        features_df.to_csv(csv_path, index=False)

        # 测试完整的工作流程
        trainer = KNNTrainer(n_neighbors=3)
        X_train, X_test, y_train, y_test, feature_cols = trainer.load_data(
            str(csv_path)
        )

        assert len(X_train) == 3, "训练集数量不正确"
        assert len(X_test) == 2, "测试集数量不正确"
        assert feature_cols == ["energy", "contrast", "correlation", "entropy"]

        # 训练模型
        trainer.train(X_train, y_train)
        assert trainer.is_trained, "模型训练失败"

        # 进行评估
        accuracy, predictions = trainer.evaluate(X_test, y_test)
        assert 0 <= accuracy <= 1, "准确率应该在0-1之间"

    def test_model_persistence(self, tmp_path):
        """测试模型保存和加载"""
        # 创建模拟数据
        X_train = np.random.rand(10, 4)
        y_train = ["D1"] * 5 + ["D2"] * 5

        # 训练并保存模型
        trainer = KNNTrainer(n_neighbors=3)
        trainer.train(X_train, y_train)

        model_path = tmp_path / "test_model.joblib"
        trainer.save_model(str(model_path))

        # 加载模型
        new_trainer = KNNTrainer()
        new_trainer.load_model(str(model_path))

        # 测试加载的模型
        assert new_trainer.is_trained, "加载的模型应该已经训练"
        X_test = np.random.rand(3, 4)
        predictions = new_trainer.model.predict(X_test)
        assert len(predictions) == 3, "加载的模型预测失败"
