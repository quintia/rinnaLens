"""
rinna tokenizer を static/model/tokenizer/ に書き出す。
transformers.js の AutoTokenizer.from_pretrained() がローカル URL から読めるようにする。

使い方:
  uv run python scripts/export_tokenizer.py
"""

import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

MODEL_ID = "rinna/japanese-gpt2-small"
OUT_DIR = Path(__file__).parent.parent / "static" / "model" / "tokenizer"

def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # transformers の save_pretrained でファイル一式を出力
    tok.save_pretrained(OUT_DIR)

    # config.json も書き出す（transformers.js がモデル種別判定に使う）
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.save_pretrained(OUT_DIR)

    # spiece.model を明示的にダウンロード
    # transformers.js は Precompiled normalizer の処理に spiece.model を必要とする
    sp_model_path = hf_hub_download(repo_id=MODEL_ID, filename="spiece.model")
    shutil.copy(sp_model_path, OUT_DIR / "spiece.model")

    print(f"Saved tokenizer files to: {OUT_DIR}")

    # 何が出力されたか列挙
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s}  {size_kb:8.1f} KB")

    # vocab の先頭を確認
    vocab = tok.get_vocab()
    print(f"\nvocab size: {len(vocab)}")
    sample = dict(list(vocab.items())[:10])
    print(f"sample vocab: {json.dumps(sample, ensure_ascii=False, indent=2)}")

    # 末尾スペース問題の確認
    texts = [
        "吾輩は猫である",
        "吾輩は猫である ",
    ]
    for t in texts:
        ids = tok.encode(t)
        tokens = [tok.decode([i]) for i in ids]
        print(f"\nencode({repr(t)})")
        print(f"  ids   : {ids}")
        print(f"  tokens: {tokens}")

if __name__ == "__main__":
    main()
