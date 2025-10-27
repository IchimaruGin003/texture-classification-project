FROM efreidevopschina.azurecr.io/cache/library/python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码和配置文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/raw_data data/processed models

# 设置入口点（默认使用production配置）
CMD ["python", "src/app.py", "--environment", "production"]