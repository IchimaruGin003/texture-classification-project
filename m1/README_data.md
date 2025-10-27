markdown
# 数据版本管理文档

## 数据来源

原始数据来自Brodatz纹理数据集，包含6个纹理类别（D1-D6），每个类别16张TIF格式的纹理图像。

## 数据版本历史

### v1: 基础数据集
- **提交哈希**: [运行 `git log --oneline | head -1` 查看]
- **数据文件**: `data/raw_data/` (DVC跟踪)
- **处理方式**: 
  - 训练集: 原始图像，无噪声
  - 测试集: 添加轻微椒盐噪声 (salt_prob=0.002, pepper_prob=0.002)
- **特点**: 基础版本，适合建立基线模型

### v2: 增强鲁棒性数据集  
- **提交哈希**: [运行 `git log --oneline | head -1` 查看]
- **数据文件**: `data/processed_v2/` (DVC跟踪)
- **处理方式**:
  - 训练集: 添加轻微噪声 (salt_prob=0.001, pepper_prob=0.001)
  - 测试集: 添加较强噪声 (salt_prob=0.01, pepper_prob=0.01)
- **特点**: 增强模型对噪声的鲁棒性，模拟真实场景

## 数据与代码关联

每个数据版本都通过DVC与特定的代码提交关联，确保实验的可复现性。

## 数据目录结构
data/
├── raw_data/ # v1: 原始数据 (DVC跟踪)
│ ├── D1/
│ ├── D2/
│ └── ...
├── processed/ # v1: 处理后的数据
└── processed_v2/ # v2: 增强鲁棒性数据 (DVC跟踪)


要获取特定版本的数据：
```bash
# 获取最新数据
dvc pull

# 获取特定版本的数据
git checkout <commit-hash>
dvc pull

### **步骤6：提交数据版本控制**

```bash
# 创建m1目录
mkdir -p m1

# 添加数据文档
git add m1/README_data.md

# 提交数据版本控制完成
git commit -m "feat: complete DVC data versioning with two dataset versions

- Add v1: baseline dataset with minimal test noise
- Add v2: enhanced dataset with training and test noise for robustness
- Create comprehensive data documentation
- Push all data to DagsHub remote storage"

# 推送到DagsHub
git push origin feature/zcy