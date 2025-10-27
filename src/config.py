"""
配置管理模块 - 支持多环境配置
"""
import os
from dotenv import load_dotenv

class Config:
    """基础配置类"""
    def __init__(self, env_file=None):
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            # 默认加载.env文件
            load_dotenv()
        
        # 应用配置
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        
        # 数据路径
        self.DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "data")
        self.RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw_data")
        self.PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
        
        # 模型配置
        self.KNN_NEIGHBORS = int(os.getenv("KNN_NEIGHBORS", "3"))
        self.TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
        self.RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
        
        # MLflow配置
        self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "texture_classification")
        
        # 性能配置
        self.MAX_PREDICTION_TIME = int(os.getenv("MAX_PREDICTION_TIME", "60"))
    
    def validate(self):
        """验证配置完整性"""
        required_vars = [
            'DATA_BASE_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR'
        ]
        
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")
        
        return True

# 创建全局配置实例
config = Config()

def get_config(environment=None):
    """获取指定环境的配置"""
    env_file = None
    if environment == "staging":
        env_file = ".env.staging"
    elif environment == "production":
        env_file = ".env.production"
    
    return Config(env_file)