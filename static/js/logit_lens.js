/**
 * Logit Lens — 各層の hidden_state に lm_head を適用してトークン予測を得る
 *
 * NOTE: 厳密な logit lens は ln_f (final layer norm) を各 hidden に適用してから
 * lm_head を通す。現状は ln_f を省略した近似実装。
 */

// ── .npy パーサ ────────────────────────────────────────────────────

function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);

  // magic: 0x93 'N' 'U' 'M' 'P' 'Y'
  if (bytes[0] !== 0x93 || bytes[1] !== 78) {
    throw new Error('Not a .npy file');
  }

  const major = bytes[6];
  let headerLen, dataOffset;
  if (major === 1) {
    headerLen  = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLen;
  } else if (major === 2) {
    headerLen  = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    dataOffset = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy version: ${major}`);
  }

  const headerBytes = bytes.slice(major === 1 ? 10 : 12, dataOffset);
  const header = String.fromCharCode(...headerBytes);

  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error('Cannot parse shape from .npy header');
  const shape = shapeMatch[1]
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n));

  const dtypeMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  if (!dtypeMatch) throw new Error('Cannot parse dtype from .npy header');
  const dtype = dtypeMatch[1]; // e.g. '<f4'

  const dataBuffer = buffer.slice(dataOffset);

  let data;
  if (dtype === '<f4' || dtype === '=f4' || dtype === '|f4') {
    data = new Float32Array(dataBuffer);
  } else if (dtype === '<f2') {
    data = float16ArrayToFloat32(new Uint16Array(dataBuffer));
  } else {
    throw new Error(`Unsupported .npy dtype: ${dtype}`);
  }

  return { shape, data };
}

// float16 → float32 変換（必要時）
function float16ArrayToFloat32(u16arr) {
  const out = new Float32Array(u16arr.length);
  for (let i = 0; i < u16arr.length; i++) {
    const h = u16arr[i];
    const sign  = (h >> 15) & 1;
    const exp   = (h >> 10) & 0x1f;
    const frac  = h & 0x3ff;
    let val;
    if (exp === 0)       val = frac * (1 / 16777216);      // subnormal
    else if (exp === 31) val = frac ? NaN : Infinity;       // inf / nan
    else                 val = (1 + frac / 1024) * (2 ** (exp - 15));
    out[i] = sign ? -val : val;
  }
  return out;
}

// ── softmax ────────────────────────────────────────────────────────

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

// ── top-k ──────────────────────────────────────────────────────────

function topK(probs, k) {
  // partial sort で上位 k 件だけ取る（全 sort より速い）
  const results = [];
  for (let i = 0; i < probs.length; i++) results.push([probs[i], i]);
  results.sort((a, b) => b[0] - a[0]);
  return results.slice(0, k);
}

// ── LogitLensEngine ────────────────────────────────────────────────

// ── LayerNorm ──────────────────────────────────────────────────────

function layerNorm(x, weight, bias, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; variance += d * d; }
  variance /= n;
  const invStd = 1 / Math.sqrt(variance + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
  }
  return out;
}

export class LogitLensEngine {
  /**
   * @param {string} baseUrl  モデルファイルのベース URL（lm_head.npy, ln_f_*.npy を含むディレクトリ）
   * @param {object} meta     metadata.json の内容（n_embd, vocab_size, n_layer）
   */
  constructor(baseUrl, meta) {
    this.baseUrl = baseUrl;
    this.meta    = meta;
    this.lmHead  = null;  // Float32Array [vocab_size * n_embd]
    this.lnWeight = null; // Float32Array [n_embd]
    this.lnBias   = null; // Float32Array [n_embd]
    this.lnEps    = 1e-5;
  }

  async load() {
    const t0 = performance.now();
    const fetchNpy = async (name) => {
      const res = await fetch(`${this.baseUrl}/${name}`);
      if (!res.ok) throw new Error(`Failed to fetch ${name}: ${res.status}`);
      return parseNpy(await res.arrayBuffer()).data;
    };

    [this.lmHead, this.lnWeight, this.lnBias] = await Promise.all([
      fetchNpy('lm_head.npy'),
      fetchNpy('ln_f_weight.npy'),
      fetchNpy('ln_f_bias.npy'),
    ]);
    return (performance.now() - t0) | 0;
  }

  /**
   * 指定トークン位置での各層 top-k 予測を計算する。
   *
   * @param {Float32Array} hiddenStates  ORT 出力 hidden_states の flat data
   *   shape: [n_layer+1, batch=1, seq, n_embd]
   * @param {number}       seqLen        トークン数
   * @param {number}       tokenPos      計算するトークン位置（-1 = 最終トークン）
   * @param {number}       k             top-k
   * @returns {Array}  長さ n_layer+1 の配列
   *   各要素: { layer, topTokens: [{id, prob, logit}] }
   */
  computePosition(hiddenStates, seqLen, tokenPos = -1, k = 10) {
    if (!this.lmHead) throw new Error('lm_head not loaded');

    const { n_embd, vocab_size, n_layer } = this.meta;
    const nLayers = n_layer + 1;  // embedding + 12 layers
    const pos = tokenPos < 0 ? seqLen + tokenPos : tokenPos;

    const results = [];

    for (let layer = 0; layer < nLayers; layer++) {
      // hidden_states の flat index: [layer, 0, pos, :]
      const hidOffset = (layer * seqLen + pos) * n_embd;
      const hiddenRaw = hiddenStates.subarray(hidOffset, hidOffset + n_embd);

      // logit lens: ln_f を適用してから lm_head を掛ける（Python と同じ）
      const hidden = layerNorm(hiddenRaw, this.lnWeight, this.lnBias, this.lnEps);

      // matmul: logits[v] = dot(lm_head[v,:], hidden[:])
      const logits = new Float32Array(vocab_size);
      for (let v = 0; v < vocab_size; v++) {
        const rowOffset = v * n_embd;
        let sum = 0;
        for (let h = 0; h < n_embd; h++) {
          sum += this.lmHead[rowOffset + h] * hidden[h];
        }
        logits[v] = sum;
      }

      const probs = softmax(logits);
      const top = topK(probs, k);
      results.push({
        layer,
        topTokens: top.map(([prob, id]) => ({ id, prob, logit: logits[id] })),
      });
    }

    return results;
  }

  /**
   * 全トークン位置 × 全層を計算する（ノードクリック時のオンデマンド用）。
   * seqLen が大きいと重いので注意。
   */
  computeAll(hiddenStates, seqLen, k = 10) {
    const results = [];
    for (let pos = 0; pos < seqLen; pos++) {
      results.push(this.computePosition(hiddenStates, seqLen, pos, k));
    }
    return results;  // [seqLen][nLayers] の 2D 配列
  }
}
