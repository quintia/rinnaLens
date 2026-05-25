/**
 * rinna/japanese-gpt2-small 向け Unigram SentencePiece トークナイザ
 * static/model/tokenizer/tokenizer.json から vocab/scores を読み込む
 */

export class UnigramTokenizer {
  constructor(tokenizerJson) {
    const model = tokenizerJson.model; // { type: "Unigram", vocab: [[token, score], ...] }

    // vocab[i] = [token_string, log_score]
    this.vocab = model.vocab;
    this.unkId = model.unk_id ?? 0;

    // token文字列 → {id, score} の Map（Viterbi 用）
    this.tokenMap = new Map();
    for (let i = 0; i < this.vocab.length; i++) {
      const [token, score] = this.vocab[i];
      this.tokenMap.set(token, { id: i, score });
    }

    // special tokens
    this.bosId = this._findId('<s>') ?? 1;
    this.eosId = this._findId('</s>') ?? 2;
    this.unkToken = this.vocab[this.unkId]?.[0] ?? '<unk>';

    // added_tokens (special tokens の追加分)
    this.addedTokens = new Map();
    for (const t of (tokenizerJson.added_tokens ?? [])) {
      this.addedTokens.set(t.content, t.id);
    }

    // post_processor から add_bos / add_eos を読む
    const pp = tokenizerJson.post_processor;
    this.addBos = pp?.type === 'TemplateProcessing'
      ? pp.single?.some(s => s.SpecialToken?.id === '<s>') ?? false
      : false;
    this.addEos = pp?.type === 'TemplateProcessing'
      ? pp.single?.some(s => s.SpecialToken?.id === '</s>') ?? false
      : false;
  }

  _findId(token) {
    const entry = this.tokenMap.get(token);
    return entry ? entry.id : null;
  }

  /**
   * SentencePiece の前処理: 先頭に ▁ を付ける（単語境界マーク）
   * rinna は空白を ▁ に変換し、文頭にも付与する
   */
  _preprocess(text) {
    // 空白 → ▁ 変換、文頭に ▁ を付加
    return '▁' + text.replace(/ /g, '▁');
  }

  /**
   * Viterbi アルゴリズムで最適トークン列を求める
   */
  _viterbi(text) {
    const n = text.length;
    // best[i] = { score, tokenEnd, tokenStr } — 位置 i まで来たときの最善
    const best = Array(n + 1).fill(null).map(() => ({ score: -Infinity, end: -1, token: null }));
    best[0].score = 0;

    for (let i = 0; i < n; i++) {
      if (best[i].score === -Infinity) continue;
      // 位置 i から始まる全 substring を試す（最大長を制限）
      const maxLen = Math.min(n - i, 32);
      for (let len = 1; len <= maxLen; len++) {
        const sub = text.slice(i, i + len);
        const entry = this.tokenMap.get(sub);
        if (entry) {
          const newScore = best[i].score + entry.score;
          if (newScore > best[i + len].score) {
            best[i + len] = { score: newScore, end: i, token: sub };
          }
        }
      }
      // unknown 文字（1文字）のフォールバック
      if (best[i + 1].score === -Infinity) {
        best[i + 1] = { score: best[i].score - 10, end: i, token: text[i] };
      }
    }

    // バックトレース
    const tokens = [];
    let pos = n;
    while (pos > 0) {
      const { end, token } = best[pos];
      tokens.push(token);
      pos = end;
    }
    tokens.reverse();
    return tokens;
  }

  /**
   * テキストをトークン ID 列にエンコード
   * @param {string} text
   * @param {object} opts
   * @param {boolean} opts.addSpecialTokens - BOS/EOS を付けるか（デフォルト true）
   * @returns {{ input_ids: number[], tokens: string[] }}
   */
  encode(text, { addSpecialTokens = true } = {}) {
    const normalized = this._preprocess(text);
    const tokenStrs = this._viterbi(normalized);

    const ids = tokenStrs.map(t => {
      const entry = this.tokenMap.get(t);
      return entry ? entry.id : this.unkId;
    });

    const finalIds = addSpecialTokens
      ? [...(this.addBos ? [this.bosId] : []), ...ids, ...(this.addEos ? [this.eosId] : [])]
      : ids;
    const finalTokens = addSpecialTokens
      ? [...(this.addBos ? ['<s>'] : []), ...tokenStrs, ...(this.addEos ? ['</s>'] : [])]
      : tokenStrs;

    return { input_ids: finalIds, tokens: finalTokens };
  }

  decode(ids, { skipSpecialTokens = true } = {}) {
    return ids
      .filter(id => !skipSpecialTokens || (id !== this.bosId && id !== this.eosId))
      .map(id => this.vocab[id]?.[0] ?? this.unkToken)
      .join('')
      .replace(/▁/g, ' ')
      .trimStart();
  }

  get vocabSize() { return this.vocab.length; }
}

/**
 * ファクトリ: /static/model/tokenizer/tokenizer.json を fetch してインスタンスを返す
 */
export async function loadTokenizer(baseUrl = new URL('../model/tokenizer', import.meta.url).href) {
  const res = await fetch(`${baseUrl}/tokenizer.json`);
  if (!res.ok) throw new Error(`Failed to fetch tokenizer.json: ${res.status}`);
  const json = await res.json();
  return new UnigramTokenizer(json);
}
