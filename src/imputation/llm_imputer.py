"""基于 Azure OpenAI GPT-4o-mini 的文本缺失填充器。"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

EXCLUDED_COLUMNS = {
    "text",
    "text_original",
    "text_is_missing",
    "text_missing_rate",
}


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return "未知"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass
class LLMImputerConfig:
    text_column: str = "text"
    system_prompt: str = "You are an expert data imputation assistant."
    temperature: float = 0.7
    max_retries: int = 3
    request_pause: float = 0.0
    example_min: int = 5
    example_max: int = 10
    group_column: str = "target_group"
    random_state: int = 42


class LLMTextImputer:
    """使用 Azure OpenAI GPT-4o-mini 对 DataFrame 中的缺失文本进行填补。"""

    def __init__(self, config: Optional[LLMImputerConfig] = None) -> None:
        load_dotenv()
        self.config = config or LLMImputerConfig()
        self.env = self._load_env()
        self.client = AzureOpenAI(
            api_key=self.env["AZURE_OPENAI_API_KEY"],
            api_version=self.env["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=self.env["AZURE_OPENAI_ENDPOINT"],
        )
        self.deployment = self.env["AZURE_OPENAI_DEPLOYMENT"]

    @staticmethod
    def _load_env() -> Dict[str, str]:
        keys = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
        ]
        env: Dict[str, str] = {}
        for key in keys:
            value = os.getenv(key)
            if not value:
                raise RuntimeError(f"环境变量 {key} 未设置，请在 .env 中配置")
            env[key] = value
        env["AZURE_OPENAI_API_VERSION"] = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        )
        return env

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回填充后的 DataFrame，不会就地修改原数据。"""

        text_column = self.config.text_column
        if text_column not in df.columns:
            raise KeyError(f"DataFrame 中不存在列 '{text_column}'")

        result = df.copy()
        mask = result[text_column].isna() | (result[text_column].astype(str).str.strip() == "")
        if not mask.any():
            return result

        examples = self._select_reference_examples(result, mask)
        missing_indices = result.index[mask]
        progress = tqdm(missing_indices, desc="LLM填充", unit="条")

        for idx in progress:
            row = result.loc[idx]
            prompt = self._build_prompt(row, examples)
            completion = self._call_model(prompt)
            if completion:
                result.at[idx, text_column] = completion
            if self.config.request_pause:
                time.sleep(self.config.request_pause)
        return result

    def _select_reference_examples(self, df: pd.DataFrame, missing_mask: pd.Series) -> List[pd.Series]:
        text_col = self.config.text_column
        group_col = self.config.group_column

        complete = df.loc[~missing_mask].copy()
        complete = complete[complete[text_col].astype(str).str.strip() != ""]
        if complete.empty:
            return []

        rng = random.Random(self.config.random_state)

        if group_col and group_col in complete.columns:
            grouped = {
                grp: grouped_df.sample(frac=1.0, random_state=rng.randint(0, 10_000)).to_dict("records")
                for grp, grouped_df in complete.groupby(group_col)
            }
            group_order = list(grouped.keys())
            rng.shuffle(group_order)

            max_needed = min(self.config.example_max, len(complete))
            min_needed = min(self.config.example_min, max_needed)

            examples: List[pd.Series] = []
            while len(examples) < max_needed and group_order:
                for grp in list(group_order):
                    records = grouped[grp]
                    if not records:
                        group_order.remove(grp)
                        continue
                    record = records.pop()
                    examples.append(pd.Series(record))
                    if len(examples) >= max_needed:
                        break
                if not group_order:
                    break

            if len(examples) < min_needed:
                needed = min(max_needed - len(examples), len(complete))
                if needed > 0:
                    fallback = complete.sample(n=needed, random_state=self.config.random_state)
                    for rec in fallback.to_dict("records"):
                        examples.append(pd.Series(rec))
                        if len(examples) >= max_needed:
                            break

            return examples[:max_needed]

        fallback = complete.sample(
            n=min(len(complete), self.config.example_max),
            random_state=self.config.random_state,
        )
        return [pd.Series(rec) for rec in fallback.to_dict("records")]

    def _format_row(self, row: pd.Series, *, missing_text: bool) -> str:
        lines: List[str] = []
        for column, value in row.items():
            if column in EXCLUDED_COLUMNS:
                continue
            if column == self.config.text_column and missing_text:
                display = "``"
            else:
                display = _normalize_value(value)
            lines.append(f"- {column}: {display}")
        return "\n".join(lines)

    def _build_prompt(self, row: pd.Series, examples: List[pd.Series]) -> str:
        intro = (
            "You are a data imputation assistant. Below are complete examples that show the relationship between table columns.\n"
            "Use these references to fill the missing `text` field in the final sample, where missing values are denoted by ``."
        )

        example_blocks: List[str] = []
        for idx, example in enumerate(examples[: self.config.example_max], start=1):
            example_blocks.append(
                f"[Example {idx}]\n{self._format_row(example, missing_text=False)}"
            )
        if not example_blocks:
            example_blocks.append("[Example]\n(No available examples)")

        target_block = (
            "[Sample to fill]\n"
            f"{self._format_row(row, missing_text=True)}"
        )

        closing = "Respond only with the completed `text` content. Do not add explanations."

        prompt = "\n\n".join([intro, "\n\n".join(example_blocks), target_block, closing])
        return prompt

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _call_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
        )
        if response.choices:
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                return "\n".join(filter(None, texts)).strip()
            try:
                content = message["content"]  # type: ignore[index]
                if isinstance(content, str):
                    return content.strip()
            except Exception:
                pass
        return ""
