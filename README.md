# LLMImputation Dataset Loader

该项目演示如何使用 `datasets` 库连接 Hugging Face 上的 `toxigen/toxigen-data` 数据集，并默认读取 `annotated` 配置；同时提供缺失值统计、缺失注入与多种插补脚本、以及基于 TF-IDF + 逻辑回归的“两阶段筛选”评估工具。

## 快速开始

1. (可选) 创建虚拟环境：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. 安装依赖（示例命令以 `conda` 环境 `bertEnv` 为例，亦可使用虚拟环境）：

   ```bash
   conda activate bertEnv
   pip install -r requirements.txt
   ```

   依赖包括 `datasets`、`pandas` 与 `pyarrow`，分别用于数据加载、统计以及直接读取 Hugging Face 上的 parquet 文件。

3. 运行数据概览脚本：

   ```bash
   python src/main.py
   ```

   程序会自动下载数据集（若本地未缓存），输出每个数据切分的行数、列数以及缺失值统计信息（区分空值与空字符串）。

## 数据下载与缺失处理脚本

使用 `src/download_parquet.py` 直接通过 pandas 读取 Hugging Face 上的 parquet 文件并保存到本地，默认包含 `train` 与 `test` 两个切分；脚本会在下载完成后打印每个切分的行数、列信息与缺失统计：

- 下载默认切分并查看统计：

  ```bash
  python src/download_parquet.py data/parquet
  ```

  结果会输出到 `data/parquet/train.parquet` 与 `data/parquet/test.parquet`，终端同时展示行数、列名以及缺失值概览。

- 如需转换为 CSV：

  ```bash
  python src/download_parquet.py data/csv --format csv
  ```

- 仅查看某个切分的前几行：

  ```bash
  python src/download_parquet.py data/ --split test --show --head 10
  ```

## 目录说明

- `README.md`：项目说明与使用指引。
- `requirements.txt`：Python 依赖列表。
- `src/main.py`：程序入口，负责加载数据集并打印形状及缺失值统计。
- `src/download_parquet.py`：命令行工具，基于 pandas 下载 Hugging Face 上的 parquet 切分，并显示行数及缺失统计，可选择保存为 parquet 或 CSV。
- `src/data/__init__.py`：数据工具包的导出入口，统一暴露加载、预览与统计函数。
- `src/data/dataset_loader.py`：封装 Hugging Face `load_dataset` 调用，提供数据集下载接口。
- `src/data/parquet_downloader.py`：定义使用 pandas 下载 parquet 切分的实用方法，并在下载时返回统计信息。
- `src/data/preview.py`：提供获取数据集前若干行的预览工具函数。
- `src/data/stats.py`：提供数据集与 DataFrame 两种形态的缺失值统计工具函数。
- `scripts/run_imputers_tradition.py` / `scripts/run_imputers_llm.py`：生成多种传统/LLM 插补版本的数据集，输出至 `data/parquet/imputed/` 与 `data/parquet/imputed_llm/`。
- `src/evaluation/screening.py`：两阶段策略的筛选阶段实现，提供数据集发现与评估逻辑。
- `scripts/run_screening.py`：命令行入口，遍历插补数据集，运行 TF-IDF + 逻辑回归评估，输出宏 F1、ROC-AUC 与人口均等差异。
### 两阶段筛选评估

第一阶段使用轻量级模型快速比较不同插补策略，第二阶段只对表现最佳的方案做昂贵的 BERT 训练。本仓库已实现筛选阶段。

1. 准备数据：确保 `data/parquet/` 下包含原始 `train.parquet` / `test.parquet`、以及不同缺失机制与插补方式生成的 `train_text_*.parquet`。
2. 运行筛选脚本（下例只评估 KNN 与 LLM 插补，包含 05/15/30 三种缺失率）：

   ```bash
   conda activate bertEnv
   python scripts/run_screening.py \
     --data-root /Users/lilijie/Projects/LLMImputation/data/parquet \
     --output-dir outputs/screening \
     --include-imputers knn llm \
     --include-rates 05 15 30
   ```

3. 脚本默认输出 CSV 结果文件（如 `outputs/screening/screening_results_YYYYMMDD_HHMMSS.csv`），并在终端提示 Top 5 结果。
4. 指标定义：
   - `macro_f1`：宏平均 F1，反映多类别总体表现。
   - `roc_auc`：多分类 ROC-AUC（OVR）。
   - `dp_diff`：人口均等差异，衡量不同 `target_group` 的正例率差距。

可通过 `--include-mechanisms mar mcar` 等参数进一步筛选缺失机制，也可以加入 `--max-datasets` 做快速调试。
- `data/`：建议的数据输出目录，可存放下载后的 parquet/CSV 文件。
