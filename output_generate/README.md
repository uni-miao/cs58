# Interactive Dashboard 使用说明

## 快速运行
### 命令行
```bash
cd CrawlData_TextProcessing_Chengyi/output_generate
python -m streamlit run interactive_dashboard.py
```

### 安装依赖
```bash
pip install streamlit pandas plotly
```

### 数据文件
确保 `datasets/` 目录下包含所需数据文件：
- `abstract_retractionNotice/abstract_features.csv`
- `abstract_retractionNotice/retraction_features.csv`
- `abstract_retractionNotice/difference_statistics.json`
- `enhanced_doi_analysis_v3_summary.csv`
- `individual_image_results.csv`
- `paper_level_summary.csv`
- `style_features_data.csv`

## 使用说明

1. 运行后浏览器会自动打开 Dashboard（通常是 `http://localhost:8501`）
2. 在侧边栏选择分析模式：
   - 📊 Multisource Integrity Dashboard（主页）
   - 📷 Image Analysis（图像分析）
   - 📈 Altmetric Source Analysis（来源分析）
   - 📝 Altmetric News Analysis（新闻分析）
   - 📊 Abstract & Retraction Analysis（摘要与撤稿分析）

## 缓存系统

### 自动缓存
- 页面加载时自动检查并使用缓存
- 源文件修改后缓存自动失效

### 预计算缓存（可选）
首次部署或数据更新后，可预先生成缓存：
```bash
python cache_utils.py                    # 预计算所有默认数据集
python cache_utils.py datasets/folder/   # 预计算指定目录
```

缓存文件存储在 `.cache/` 目录（Parquet 格式）

## 常见问题

- **命令未找到**：使用 `python -m streamlit run interactive_dashboard.py`
- **端口占用**：Streamlit 会自动尝试其他端口（8502, 8503...）
- **停止运行**：终端按 `Ctrl + C`

