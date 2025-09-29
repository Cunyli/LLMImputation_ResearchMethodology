# LLMImputation Dataset Loader

该项目演示如何使用 `datasets` 库连接 Hugging Face 上的 `toxigen/toxigen-data` 数据集，并默认读取 `annotated` 配置；同时提供缺失值统计工具与基于 pandas 的 parquet 下载脚本。

## 快速开始

1. (可选) 创建虚拟环境：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   依赖包括 `datasets`、`pandas` 与 `pyarrow`，分别用于数据加载、统计以及直接读取 Hugging Face 上的 parquet 文件。

3. 运行统计脚本：

   ```bash
   python src/main.py
   ```

   程序会自动下载数据集（若本地未缓存），输出每个数据切分的行数、列数以及缺失值统计信息（区分空值与空字符串）。

## 下载 parquet 数据

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
- `data/`：建议的数据输出目录，可存放下载后的 parquet/CSV 文件。
