/**
 * InferenceEngine — ONNX Runtime Web を使ったブラウザ内推論エンジン
 *
 * 担当:
 *   - モデルチャンクの fetch & 結合 → ORT InferenceSession 作成
 *   - テキスト → トークン → 推論 → logit lens → グラフデータ
 *   - ノードクリック時のデータ提供（attention / logits / tokens）
 */

import { loadTokenizer }    from './tokenizer.js';
import { LogitLensEngine }  from './logit_lens.js';
import { buildGraphData }   from './graph_builder.js';

const MODEL_BASE     = new URL('../model/gpt2',      import.meta.url).href;
const TOKENIZER_BASE = new URL('../model/tokenizer', import.meta.url).href;

export class InferenceEngine {
  constructor() {
    this.ort        = null;  // onnxruntime-web モジュール
    this.session    = null;
    this.tokenizer  = null;
    this.lensEngine = null;
    this.meta       = null;
    this._cache     = null;  // 直近の推論結果キャッシュ
  }

  /**
   * モデル全体をロードする。
   * @param {function} onProgress ({ phase, value }) コールバック
   */
  async load(onProgress = () => {}) {
    // ORT を動的 import（呼び出し元が await engine.load() を終えるまで使わない）
    const ortModule = await import(new URL('../vendor/ort/ort.min.mjs', import.meta.url).href);
    this.ort = ortModule;
    this.ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;

    onProgress({ phase: 'meta', value: 0 });

    // メタデータ + トークナイザを並行取得
    const [meta, tokenizer] = await Promise.all([
      fetch(`${MODEL_BASE}/metadata.json`).then(r => r.json()),
      loadTokenizer(TOKENIZER_BASE),
    ]);
    this.meta      = meta;
    this.tokenizer = tokenizer;

    onProgress({ phase: 'lmhead', value: 0.1 });

    // lm_head と モデルチャンクを並行取得
    this.lensEngine = new LogitLensEngine(MODEL_BASE, meta);

    const [modelBytes] = await Promise.all([
      this._fetchChunks(meta, (v) => onProgress({ phase: 'chunks', value: 0.1 + v * 0.8 })),
      this.lensEngine.load(),
    ]);

    onProgress({ phase: 'session', value: 0.9 });

    this.session = await this.ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
    });

    onProgress({ phase: 'ready', value: 1.0 });
  }

  async _fetchChunks(meta, onChunkProgress) {
    const chunks = meta.onnx_chunks;
    const buffers = await Promise.all(
      chunks.map(async (name, i) => {
        const res = await fetch(`${MODEL_BASE}/${name}`);
        if (!res.ok) throw new Error(`Failed to fetch chunk: ${name}`);
        const buf = await res.arrayBuffer();
        onChunkProgress((i + 1) / chunks.length);
        return buf;
      })
    );
    const total  = buffers.reduce((s, b) => s + b.byteLength, 0);
    const merged = new Uint8Array(total);
    let offset   = 0;
    for (const buf of buffers) { merged.set(new Uint8Array(buf), offset); offset += buf.byteLength; }
    return merged;
  }

  /**
   * テキストを解析し、グラフデータと次トークン予測を返す。
   * @param {string} text
   * @returns {{ prompt, output, graph_data }}
   */
  async run(text) {
    if (!this.session) throw new Error('Engine not loaded');

    // Python の add_special_tokens=False と同じ挙動：BOS・EOS を付けない
    const encoded   = this.tokenizer.encode(text, { addSpecialTokens: false });
    const ids       = encoded.input_ids;
    const tokensArr = encoded.tokens;
    const seqLen  = ids.length;

    // int64 テンソル
    const inputTensor = new this.ort.Tensor(
      'int64',
      BigInt64Array.from(ids.map(BigInt)),
      [1, seqLen],
    );

    const output = await this.session.run({ input_ids: inputTensor });
    const logits  = output['logits'];      // [1, seq, vocab]
    const hidden  = output['hidden_states'];  // [n_layer+1, 1, seq, hidden]
    const attn    = output['attentions'];  // [n_layer, 1, n_head, seq, seq]

    // 最終トークン位置で全層の logit lens を計算
    const lensResults = this.lensEngine.computePosition(hidden.data, seqLen, -1, 10);

    // MLP ノードの top 予測（layer n の MLP = lensResults[n+1]）
    const topPredictions = {};
    for (let l = 0; l < this.meta.n_layer; l++) {
      const { topTokens } = lensResults[l + 1];
      topPredictions[`MLP${l}`] = topTokens.map(({ id, prob }) => ({
        token: this.tokenizer.vocab[id]?.[0]?.replace(/▁/g, ' ').trim() || '?',
        prob:  parseFloat((prob * 100).toFixed(1)),
      }));
    }

    // 最終出力トークン（最後の層の top-1）
    const lastLayerTop = lensResults[lensResults.length - 1].topTokens[0];
    const outputToken  = this.tokenizer.vocab[lastLayerTop.id]?.[0]
      ?.replace(/▁/g, ' ').trim() ?? '';

    // 推論結果をキャッシュ
    this._cache = {
      tokens:   tokensArr,
      inputIds: ids,
      logits,
      hidden,
      attn,
      lensResults,
      seqLen,
    };

    const graphData = buildGraphData(this.meta, topPredictions);

    return {
      prompt:     text,
      output:     outputToken,
      graph_data: graphData,
    };
  }

  /**
   * ノードクリック時のデータを返す（app.js の openNode から呼ばれる）。
   * Python の /api/node/{id} と /api/node/{id}/{resource} を代替する。
   *
   * @param {string} nodeId
   * @returns {object[]}  payload の配列（type: "tokens" | "attention" | "logits"）
   */
  getNodeData(nodeId) {
    if (!this._cache) throw new Error('No inference result cached');
    const { tokens, inputIds, attn, lensResults, seqLen } = this._cache;
    const { n_head, n_embd } = this.meta;

    // SentencePiece standalone word-boundary marker — Python では非表示にする
    const blankIdx = new Set(
      tokens.map((t, i) => t === '▁' ? i : -1).filter(i => i >= 0)
    );

    if (nodeId === 'Input') {
      return [{
        type:  'tokens',
        items: inputIds
          .map((id, i) => ({
            index:    i,
            token_id: id,
            token:    tokens[i]?.replace(/▁/g, ' ') ?? '?',
          }))
          .filter((_, i) => !blankIdx.has(i)),
      }];
    }

    if (nodeId.startsWith('A')) {
      // A{layer}.H{head}
      const [layerStr, headStr] = nodeId.slice(1).split('.H');
      const layer = parseInt(layerStr, 10);
      const head  = parseInt(headStr,  10);

      // attn.data: flat [n_layer, 1, n_head, seq, seq]
      const stride = n_head * seqLen * seqLen;

      // ▁ トークンの行/列を除外し、残りの行を正規化
      const visibleIdx = [...Array(seqLen).keys()].filter(i => !blankIdx.has(i));
      const matrix = visibleIdx.map(row => {
        const rowData = visibleIdx.map(col => {
          const idx = layer * stride + head * seqLen * seqLen + row * seqLen + col;
          return attn.data[idx];
        });
        const sum = rowData.reduce((s, v) => s + v, 0);
        return sum > 0
          ? rowData.map(v => parseFloat((v / sum).toFixed(6)))
          : rowData.map(v => parseFloat(v.toFixed(6)));
      });

      return [{
        type:   'attention',
        title:  `${nodeId} — Attention Pattern`,
        layer,
        head,
        tokens: visibleIdx.map(i => tokens[i]?.replace(/▁/g, ' ') ?? '?'),
        matrix,
      }];
    }

    if (nodeId.startsWith('MLP') || nodeId === 'Output') {
      let lensIdx;
      if (nodeId === 'Output')          lensIdx = lensResults.length - 1;
      else                              lensIdx = parseInt(nodeId.slice(3), 10) + 1;

      const { topTokens } = this.lensEngine.computePosition(
        this._cache.hidden.data, seqLen, -1, 10
      )[lensIdx];

      return [{
        type:  'logits',
        title: `${nodeId} — Logit Lens (last token)`,
        items: topTokens.map(({ id, prob, logit }) => ({
          token_id: id,
          token:    this.tokenizer.vocab[id]?.[0]?.replace(/▁/g, ' ').trim() ?? '?',
          logit:    parseFloat(logit.toFixed(6)),
          prob:     parseFloat(prob.toFixed(6)),
        })),
      }];
    }

    return [];
  }
}
