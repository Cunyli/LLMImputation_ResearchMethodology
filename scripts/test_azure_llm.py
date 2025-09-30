"""使用 Azure OpenAI GPT-4o-mini 对缺失文本进行示例填充。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

TEXT_COLUMN = "text"
DEFAULT_DATA_PATH = Path("data/parquet/mcar/train_text_mcar_30.parquet")


def ensure_env() -> Dict[str, str]:
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    env: Dict[str, str] = {}
    for key in required:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"环境变量 {key} 未设置，请在 .env 中填写后重新运行")
        env[key] = value
    env["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    return env


def load_sample_row(path: Path, text_column: str = TEXT_COLUMN) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    df = pd.read_parquet(path)
    mask = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
    if not mask.any():
        raise ValueError("数据中没有缺失的文本样本，请选择其他文件")

    row = df[mask].iloc[0]
    return row.to_dict()


def build_prompt(row: Dict[str, Any], text_column: str = TEXT_COLUMN) -> str:
    context_items = [f"{k}: {v}" for k, v in row.items() if k != text_column]
    context = "\n".join(context_items)
    prompt = (
        "你是一名帮助标注员修复数据集缺失文本的助手。\n"
        "请根据下方元数据补写缺失的文本字段，要求内容自然、符合上下文，不要引入事实错误。\n"
        "尽量保持语言与上下文一致，可以输出1~2句话。\n\n"
        "[元数据]\n"
        f"{context}\n\n"
        "请输出最终的文本。"
    )
    return prompt


def call_azure_llm(prompt: str, env: Dict[str, str]) -> str:
    client = AzureOpenAI(
        api_key=env["AZURE_OPENAI_API_KEY"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
    )
    deployment = env["AZURE_OPENAI_DEPLOYMENT"]
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "你是一个专业的数据填补助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    if response.choices:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            if texts:
                return "\n".join(texts).strip()
        # 回退到字典访问
        try:
            # pydantic模型同时支持 dict 接口
            content = message["content"]
            if isinstance(content, str):
                return content.strip()
        except Exception:
            pass
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 Azure GPT-4o-mini 进行缺失文本填充示例")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="包含缺失文本的数据文件（parquet）",
    )
    args = parser.parse_args()

    load_dotenv()
    env = ensure_env()

    row = load_sample_row(args.data)
    prompt = build_prompt(row)
    print("=== 发送给模型的提示 ===")
    print(prompt)
    print("==========================\n")

    completion = call_azure_llm(prompt, env)
    print("=== 模型回复 ===")
    print(completion)


if __name__ == "__main__":
    main()
