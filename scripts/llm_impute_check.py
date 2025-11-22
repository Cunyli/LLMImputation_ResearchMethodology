from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.imputation.llm_imputer import LLMTextImputer, LLMImputerConfig  # noqa: E402

DATA_PATH = Path("data/parquet/mar/train_text_mar_30.parquet")
LOG_PATH = Path("llm_debug.jsonl")

if LOG_PATH.exists():
    LOG_PATH.unlink()

original = pd.read_parquet(DATA_PATH).head(10)
config = LLMImputerConfig(log_path=LOG_PATH)
imputer = LLMTextImputer(config)
filled = imputer.impute(original.copy())

text_col = config.text_column
changed_mask = filled[text_col].astype(str) != original[text_col].astype(str)

print("=== Rows with newly generated text ===")
print(filled.loc[changed_mask, [text_col]])

print("\n=== All 10 rows (after imputation) ===")
print(filled[[text_col]])

if LOG_PATH.exists():
    print("\n=== LLM prompts & completions (first few) ===")
    with LOG_PATH.open(encoding="utf-8") as log_file:
        for idx, line in enumerate(log_file):
            if idx >= 5:
                break
            entry = json.loads(line)
            print(f"\nRow {entry.get('row_index')} attempt {entry.get('attempt')}:")
            print("Prompt:")
            print(entry.get("prompt", ""))
            print("Completion:")
            print(entry.get("completion", ""))
