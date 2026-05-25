"""
rinna/japanese-gpt2-small を ONNX にエクスポート（中間出力全込み）

GPT-2 のブロックを手動ループして hidden_states / attentions を収集する。
HuggingFace の output_capturing 機構（新 transformers で副作用ベース）を
回避するための実装。

出力先: static/model/gpt2/
  model.onnx          — 本体
  model.onnx.partXX   — 50MB チャンク分割版（ブラウザ向け）
  lm_head.npy         — logit lens 用重み [vocab, hidden]
  metadata.json       — レイヤー数・ヘッド数等

使い方:
  uv run python scripts/export_onnx.py
"""

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_ID = "rinna/japanese-gpt2-small"
OUT_DIR = Path(__file__).parent.parent / "static" / "model" / "gpt2"
CHUNK_MB = 50


class GPT2ManualForward(nn.Module):
    """GPT-2 の各ブロックを手動で呼び出し中間値を収集するラッパー。

    HuggingFace の output_capturing 機構（torch.export と非互換）を使わず、
    各ブロックを直接呼び出して hidden_states / attentions を収集する。

    Returns:
        logits        [batch, seq, vocab]
        hidden_states [n_layer+1, batch, seq, hidden]  ← embedding + 各ブロック後
        attentions    [n_layer, batch, n_head, seq, seq]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        transformer = model.transformer
        self.wte = transformer.wte          # token embedding
        self.wpe = transformer.wpe          # position embedding
        self.drop = transformer.drop
        self.blocks = transformer.h         # ModuleList of GPT2Block
        self.ln_f = transformer.ln_f        # final layer norm
        self.lm_head = model.lm_head

    def forward(self, input_ids: torch.Tensor):
        batch, seq = input_ids.shape
        pos = torch.arange(seq, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        hidden = self.drop(self.wte(input_ids) + self.wpe(pos))

        # 新しい transformers では GPT2Attention が外部から渡された causal mask に依存する。
        # HuggingFace の GPT2Model と同じ因果マスクを作成して各層に渡す。
        # shape: [1, 1, seq, seq] — 上三角が -inf、下三角（対角含む）が 0
        min_val = torch.finfo(hidden.dtype).min
        causal_mask = torch.full((seq, seq), min_val, dtype=hidden.dtype, device=input_ids.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        all_hidden = [hidden]   # list[tensor [batch, seq, hidden]]
        all_attn   = []         # list[tensor [batch, heads, seq, seq]]

        for block in self.blocks:
            # GPT2Block を層ごとに直接呼び出す（新 transformers では
            # block.forward が output_attentions を返さないため）
            # ln_1 → attn → residual → ln_2 → mlp → residual
            residual = hidden
            hidden = block.ln_1(hidden)
            attn_out = block.attn(hidden, attention_mask=causal_mask, output_attentions=True)
            hidden       = residual + attn_out[0]   # [batch, seq, hidden]
            attn_weights = attn_out[1]              # [batch, heads, seq, seq]

            residual = hidden
            hidden = block.ln_2(hidden)
            hidden = residual + block.mlp(hidden)

            all_hidden.append(hidden)
            all_attn.append(attn_weights)

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        hidden_states = torch.stack(all_hidden, dim=0)  # [n_layer+1, batch, seq, hidden]
        attentions    = torch.stack(all_attn,   dim=0)  # [n_layer, batch, heads, seq, seq]

        return logits, hidden_states, attentions


def split_into_chunks(src: Path, chunk_mb: int) -> list[str]:
    chunk_size = chunk_mb * 1024 * 1024
    data = src.read_bytes()
    names = []
    for i, offset in enumerate(range(0, len(data), chunk_size)):
        name = f"{src.name}.part{i:02d}"
        (src.parent / name).write_bytes(data[offset : offset + chunk_size])
        names.append(name)
    return names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── モデルロード ──────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    )
    model.eval()
    cfg = model.config
    print(
        f"  {cfg.n_layer} layers / {cfg.n_head} heads / {cfg.n_embd} dim"
        f" / {cfg.vocab_size} vocab  ({time.time()-t0:.1f}s)"
    )

    # ── lm_head 重みを別途保存 ───────────────────────────────────
    lm_head_w = model.lm_head.weight.detach().float().numpy()  # [vocab, hidden]
    np.save(OUT_DIR / "lm_head.npy", lm_head_w)
    print(f"  lm_head.npy: {lm_head_w.shape}  ({lm_head_w.nbytes/1e6:.1f} MB)")

    # ── ln_f 重みを別途保存 ──────────────────────────────────────
    # logit lens では全層に ln_f を適用してから lm_head を通す
    ln_f_weight = model.transformer.ln_f.weight.detach().float().numpy()  # [hidden]
    ln_f_bias   = model.transformer.ln_f.bias.detach().float().numpy()    # [hidden]
    np.save(OUT_DIR / "ln_f_weight.npy", ln_f_weight)
    np.save(OUT_DIR / "ln_f_bias.npy",   ln_f_bias)
    print(f"  ln_f_weight.npy: {ln_f_weight.shape}  ln_f_bias.npy: {ln_f_bias.shape}")

    # ── 動作確認 ─────────────────────────────────────────────────
    wrapper = GPT2ManualForward(model)
    wrapper.eval()

    dummy = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits, hidden_states, attentions = wrapper(dummy)
    print(f"  Manual forward OK:")
    print(f"    logits        : {tuple(logits.shape)}")
    print(f"    hidden_states : {tuple(hidden_states.shape)}")
    print(f"    attentions    : {tuple(attentions.shape)}")

    # ── ONNX エクスポート ────────────────────────────────────────
    onnx_path = OUT_DIR / "model.onnx"
    print(f"\nExporting ONNX → {onnx_path} ...")
    t0 = time.time()

    seq_dim = torch.export.Dim("seq_len", min=2, max=2048)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits", "hidden_states", "attentions"],
            dynamic_shapes={"input_ids": {0: torch.export.Dim.STATIC, 1: seq_dim}},
            dynamo=True,
        )

    # 外部データファイルがあれば単一ファイルに統合
    onnx_data = onnx_path.parent / "model.onnx.data"
    if onnx_data.exists():
        print("  Merging external data ...")
        merged = onnx.load(str(onnx_path), load_external_data=True)
        onnx_path.unlink()
        onnx_data.unlink()
        onnx.save(merged, str(onnx_path), save_as_external_data=False)

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  ONNX size: {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")

    # ── onnxruntime で推論確認 ───────────────────────────────────
    print("Verifying with onnxruntime ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_ids": dummy.numpy()})
    lo, hs, at = ort_out
    print(f"  ✓ logits={lo.shape} hidden={hs.shape} attn={at.shape}")

    # ── チャンク分割 ─────────────────────────────────────────────
    print(f"Splitting into {CHUNK_MB}MB chunks ...")
    parts = split_into_chunks(onnx_path, CHUNK_MB)
    print(f"  {len(parts)} chunks: {parts[0]} .. {parts[-1]}")

    # ── メタデータ ───────────────────────────────────────────────
    meta = {
        "model_id": MODEL_ID,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
        "onnx_size_mb": round(size_mb, 1),
        "onnx_chunks": parts,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\nDone. Output files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:45s}  {f.stat().st_size/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
