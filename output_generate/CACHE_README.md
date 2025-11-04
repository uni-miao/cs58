# 缓存系统使用说明

## 📁 文件说明

`cache_utils.py` - 统一的缓存管理模块

这个文件整合了缓存工具库和预计算功能，可以：
1. **作为模块使用** - 被各个页面导入
2. **作为脚本运行** - 预先生成缓存

## 🔧 使用方式

### 方式 1: 作为模块（自动缓存）

各个页面会自动使用缓存功能：

```python
from cache_utils import (
    load_cache, save_cache, is_cache_valid,
    clear_cache, load_data_from_source
)
```

**工作流程：**
- 页面加载时自动检查缓存
- 如果缓存有效，从缓存加载（快速）
- 如果缓存无效，从源文件加载并缓存（较慢，但下次就快了）

### 方式 2: 作为脚本（预计算缓存）

**预计算单个文件：**
```bash
python cache_utils.py datasets/abstract_retractionNotice/abstract_features.csv
```

**预计算整个目录：**
```bash
python cache_utils.py datasets/abstract_retractionNotice/
```

**预计算所有默认数据集（不提供参数）：**
```bash
python cache_utils.py
```

默认数据集包括：
- `enhanced_doi_analysis_v3_summary.csv`
- `paper_level_summary.csv`
- `paper_domain_pairs.csv`
- `style_features_data.csv`
- `abstract_retractionNotice/abstract_features.csv`
- `abstract_retractionNotice/retraction_features.csv`

## 📊 缓存位置

缓存文件存储在：
```
output_generate/
└── .cache/
    ├── doi_analysis__{signature}.parquet  # 缓存数据（Parquet格式）
    └── doi_analysis__{signature}.json     # 元数据（签名、时间等）
```

## 🎯 功能特性

### 核心缓存功能
- ✅ `get_file_signature()` - 生成文件签名
- ✅ `is_cache_valid()` - 检查缓存有效性
- ✅ `save_cache()` - 保存到缓存（Parquet格式）
- ✅ `load_cache()` - 从缓存加载
- ✅ `clear_cache()` - 清除缓存
- ✅ `load_data_from_source()` - 从源加载并自动缓存

### 预计算功能
- ✅ `precompute_cache()` - 预计算单个文件
- ✅ `precompute_directory()` - 预计算整个目录
- ✅ `precompute_all_default_datasets()` - 预计算所有默认数据集

## 🚀 使用场景

### 场景 1: 日常使用（推荐）
- **使用：** 自动缓存（页面自动处理）
- **优点：** 无需手动操作，首次访问自动生成缓存

### 场景 2: 首次部署/数据更新
- **使用：** 运行 `python cache_utils.py` 预先生成缓存
- **优点：** 确保所有用户首次访问都很快

### 场景 3: 批量处理
- **使用：** `python cache_utils.py datasets/`
- **优点：** 一次性处理所有CSV文件

## 💡 注意事项

1. **缓存自动失效** - 当源文件被修改时，缓存会自动失效
2. **缓存格式** - 优先使用Parquet格式（更快），如果pyarrow不可用则回退到pickle
3. **手动清除** - 可以在仪表板侧边栏的"Cache Management"中清除缓存

## 📝 示例输出

运行预计算脚本时的输出示例：

```
============================================================
Precomputing cache for: datasets/abstract_features.csv
============================================================

📂 Loading data from source CSV...
✓ Loaded 50,902 records
💾 Saving to cache...
✓ Cache saved successfully!
  Cache directory: .cache
  Cache file: .cache/doi_analysis__abc123def456.parquet
  Metadata file: .cache/doi_analysis__abc123def456.json

🎉 Precomputation complete! Dashboard will load instantly.
```

