import React, { useEffect, useRef } from "https://esm.sh/react@18";
import { createRoot } from "https://esm.sh/react-dom@18/client";
import {
  Background,
  Controls,
  Handle,

  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "https://esm.sh/@xyflow/react@12?deps=react@18,react-dom@18";
import { InferenceEngine } from "./engine.js";

const e = React.createElement;

const promptField = document.getElementById("prompt");
const submitBtn = document.getElementById("submit");
const exampleDropdown = document.getElementById("example-dropdown");
const exampleDropdownToggle = document.getElementById("example-dropdown-toggle");
const exampleDropdownMenu = document.getElementById("example-dropdown-menu");
const clearLink = document.getElementById("clear-link");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("result");
const graphEl = document.getElementById("graph");
const modal = document.getElementById("imageModal");
const modalBody = document.getElementById("modalBody");
const modalCloseBtn = document.getElementById("modal-close-btn");

const graphRoot = createRoot(graphEl);

let currentSource = null;
let currentGraphData = null;
let examplePrompts = [];

const getPromptValue = () => (promptField.innerText ?? "").replace(/\u00A0/g, " ");

const setPromptText = (text) => {
  promptField.textContent = text ?? "";
};

const enterEditMode = ({ focus = false, caretToEnd = false } = {}) => {
  if (promptField.dataset.mode === "edit") {
    if (focus) promptField.focus();
    return;
  }
  const text = getPromptValue();
  promptField.dataset.mode = "edit";
  promptField.setAttribute("contenteditable", "true");
  setPromptText(text);
  if (focus) {
    promptField.focus();
    if (caretToEnd) {
      const range = document.createRange();
      range.selectNodeContents(promptField);
      range.collapse(false);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }
};

const enterDisplayMode = (prompt, output) => {
  promptField.dataset.mode = "display";
  promptField.setAttribute("contenteditable", "false");
  promptField.innerHTML = "";
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt-text";
  promptSpan.textContent = prompt ?? "";
  const predictSpan = document.createElement("span");
  predictSpan.className = "predict-text";
  predictSpan.textContent = output ?? "";
  promptField.appendChild(promptSpan);
  promptField.appendChild(predictSpan);
};

const engine = new InferenceEngine();

const phaseLabels = {
  cache: "モデル内部データを取得中…",
  init_graph: "グラフを組み立てています…",
  heatmaps: "アテンション可視化を生成中…",
  logits: "ロジット可視化を生成中…",
  final_graph: "グラフを仕上げています…",
};

function LensNode({ data }) {
  const nodeStyle = data?.nodeStyle ?? {};

  return e(
    "div",
    {
      style: {
        ...nodeStyle,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
      },
    },
    e(Handle, { type: "target", position: Position.Left }),
    e("div", null, data.label),
    e(Handle, { type: "source", position: Position.Right }),
    e(Handle, { type: "source", id: "bottom", position: Position.Bottom, style: { left: "75%" } }),
  );
}

function PredictionNode({ data }) {
  const topPredictions = data?.topPredictions ?? [];

  return e(
    "div",
    {
      style: {
        position: "relative",
        width: "360px",
        padding: "4px 6px",
        borderRadius: "12px",
        background: "rgba(255, 255, 255, 0.94)",
        border: "1px solid rgba(132, 152, 177, 0.18)",
        boxShadow: "0 8px 20px rgba(66, 94, 130, 0.08)",
        display: "flex",
        flexDirection: "column",
        gap: "2px",
      },
    },
    e(Handle, { type: "target", id: "top", position: Position.Top }),
    ...topPredictions.map((item, idx) =>
      e(
        "div",
        {
          key: `${item.token}-${idx}`,
          style: {
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "3px 6px",
            borderRadius: "6px",
            background: "#f8fbff",
            border: "1px solid rgba(132, 152, 177, 0.12)",
            color: "#16202c",
            fontSize: "18px",
            fontWeight: 600,
            lineHeight: 1.2,
          },
        },
        e(
          "span",
          {
            style: {
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: "1 1 0",
              minWidth: 0,
              fontSize: "36px",
            },
          },
          item.token,
        ),
        e(
          "span",
          {
            style: {
              color: "#607286",
              fontVariantNumeric: "tabular-nums",
              flexShrink: 0,
              marginLeft: "6px",
            },
          },
          `${item.prob}%`,
        ),
      ),
    ),
  );
}

const nodeTypes = {
  lensNode: LensNode,
  predictionNode: PredictionNode,
};

function FlowCanvas({ graphData, onNodeOpen }) {
  const { fitView } = useReactFlow();
  const nodes = graphData?.nodes ?? [];
  const edges = graphData?.edges ?? [];
  const hasFitView = useRef(false);

  useEffect(() => {
    if (!nodes.length) {
      return;
    }
    if (hasFitView.current) {
      return;  // 2回目以降はビューを保持
    }
    hasFitView.current = true;
    const timer = window.setTimeout(() => {
      fitView({ padding: 0.16, duration: 500, maxZoom: 1.15 });
    }, 20);
    return () => window.clearTimeout(timer);
  }, [fitView, nodes]);

  return e(
    ReactFlow,
    {
      nodes,
      edges,
      fitView: true,
      minZoom: 0.25,
      maxZoom: 1.6,
      defaultEdgeOptions: { zIndex: 0 },
      nodesDraggable: false,
      nodesConnectable: false,
      elementsSelectable: true,
      zoomOnDoubleClick: false,
      panOnDrag: true,
      panOnScroll: false,
      selectionOnDrag: false,
      proOptions: { hideAttribution: true },
      nodeTypes,
      onNodeClick: (_event, node) => onNodeOpen(node.data?.parentId || node.id),
    },
    e(Background, {
      color: "rgba(157, 176, 196, 0.22)",
      gap: 22,
      size: 1.1,
      variant: "dots",
    }),

    e(Controls, {
      position: "bottom-right",
      showInteractive: false,
    }),
  );
}

function GraphApp({ graphData, onNodeOpen }) {
  if (!graphData?.nodes?.length) {
    return e("div", { className: "graph-empty" }, "解析を実行すると構造グラフがここに表示されます");
  }

  return e(
    ReactFlowProvider,
    null,
    e(FlowCanvas, { graphData, onNodeOpen }),
  );
}

function renderGraph(graphData) {
  currentGraphData = graphData;
  graphRoot.render(e(GraphApp, { graphData, onNodeOpen: openNode }));
}

const setStatus = (text) => {
  statusEl.textContent = text || "";
};

const setExampleDropdownVisible = (visible) => {
  if (exampleDropdown) {
    exampleDropdown.hidden = !visible;
  }
};

const populateExampleDropdown = () => {
  if (!exampleDropdownMenu) return;
  exampleDropdownMenu.innerHTML = "";
  examplePrompts.forEach((text) => {
    const item = document.createElement("li");
    item.setAttribute("role", "option");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      closeExampleDropdown();
      resetPredictionState();
      setPromptText(text);
      enterEditMode({ focus: true, caretToEnd: true });
    });
    item.appendChild(btn);
    exampleDropdownMenu.appendChild(item);
  });
};

const openExampleDropdown = () => {
  if (!exampleDropdown || !exampleDropdownMenu) return;
  exampleDropdown.dataset.open = "true";
  exampleDropdownMenu.hidden = false;
  exampleDropdownToggle?.setAttribute("aria-expanded", "true");
};

const closeExampleDropdown = () => {
  if (!exampleDropdown || !exampleDropdownMenu) return;
  exampleDropdown.dataset.open = "false";
  exampleDropdownMenu.hidden = true;
  exampleDropdownToggle?.setAttribute("aria-expanded", "false");
};

const toggleExampleDropdown = () => {
  if (exampleDropdown?.dataset.open === "true") {
    closeExampleDropdown();
  } else {
    openExampleDropdown();
  }
};

const parseExamplePrompts = (text) => {
  try {
    const payload = JSON.parse(text);
    if (Array.isArray(payload)) {
      return payload.filter((s) => typeof s === "string" && s !== "");
    }
  } catch (_) {
    // Plain text examples are handled below.
  }

  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter((s) => s !== "");
};

const setRunning = (running) => {
  submitBtn.disabled = running;
  if (running) {
    promptField.setAttribute("contenteditable", "false");
    promptField.setAttribute("aria-disabled", "true");
  } else {
    promptField.removeAttribute("aria-disabled");
    if (promptField.dataset.mode === "edit") {
      promptField.setAttribute("contenteditable", "true");
    }
  }
  if (exampleDropdownToggle) {
    exampleDropdownToggle.disabled = running;
  }
  if (running) {
    closeExampleDropdown();
  }
  if (clearLink) {
    clearLink.disabled = running;
  }
};

const closeModal = () => {
  modal.classList.remove("modal-active");
};

const createModalSection = (title) => {
  const section = document.createElement("section");
  section.className = "modal-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.appendChild(heading);
  return section;
};

const renderModalFeedback = (message) => {
  modalBody.innerHTML = "";
  const feedback = document.createElement("div");
  feedback.className = "modal-feedback";
  feedback.textContent = message;
  modalBody.appendChild(feedback);
};

const buildHeatmapColor = (value, maxValue = 1) => {
  const safeMax = maxValue > 0 ? maxValue : 1;
  const clamped = Math.max(0, Math.min(1, value / safeMax));
  const blue = Math.round(255 - clamped * 70);
  const red = Math.round(248 - clamped * 185);
  const green = Math.round(251 - clamped * 130);
  return `rgb(${red}, ${green}, ${blue})`;
};

const renderTokensView = (payload) => {
  const section = createModalSection(`トークン列（${payload.items.length} トークン）`);
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;";
  payload.items.forEach(({ token, token_id }) => {
    const chip = document.createElement("span");
    chip.style.cssText = "position:relative;display:inline-block;padding:4px 10px;background:#1e2d3d;color:#e8eef6;border-radius:8px;font-size:13px;font-weight:600;cursor:default;";
    chip.textContent = token || "　";
    const tip = document.createElement("span");
    tip.textContent = `ID: ${token_id}`;
    tip.style.cssText = "display:none;position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);background:#0d141c;color:#a6b3c2;font-size:11px;padding:3px 7px;border-radius:5px;white-space:nowrap;pointer-events:none;";
    chip.appendChild(tip);
    chip.addEventListener("mouseenter", () => { tip.style.display = "block"; });
    chip.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    wrap.appendChild(chip);
  });
  section.appendChild(wrap);
  modalBody.appendChild(section);
};

const renderAttentionView = (payload) => {
  const section = createModalSection(payload.title);
  const frame = document.createElement("div");
  frame.className = "attention-frame";

  const xTokens = document.createElement("div");
  xTokens.className = "attention-x-tokens";
  xTokens.style.gridTemplateColumns = `repeat(${payload.tokens.length}, minmax(28px, 1fr))`;
  payload.tokens.forEach((token) => {
    const cell = document.createElement("span");
    cell.className = "attention-token attention-token-x";
    cell.textContent = token;
    xTokens.appendChild(cell);
  });

  const yTokens = document.createElement("div");
  yTokens.className = "attention-y-tokens";
  payload.tokens.forEach((token) => {
    const cell = document.createElement("span");
    cell.className = "attention-token attention-token-y";
    cell.textContent = token;
    yTokens.appendChild(cell);
  });

  const matrix = document.createElement("div");
  matrix.className = "attention-grid";
  matrix.style.gridTemplateColumns = `repeat(${payload.tokens.length}, minmax(28px, 1fr))`;
  const maxValue = Math.max(
    0,
    ...payload.matrix.flat().map((value) => Number(value) || 0),
  );
  for (let row = 0; row < payload.matrix.length; row += 1) {
    for (let col = 0; col < payload.matrix[row].length; col += 1) {
      const value = payload.matrix[row][col];
      const cell = document.createElement("div");
      cell.className = "attention-cell";
      cell.style.background = buildHeatmapColor(value, maxValue);
      cell.title = `Q:${payload.tokens[row]} | K:${payload.tokens[col]} | ${value.toFixed(4)}`;
      if (maxValue > 0 && value / maxValue >= 0.72) {
        cell.style.color = "#ffffff";
      }
      cell.textContent = value >= Math.max(maxValue * 0.18, 0.02) ? value.toFixed(2) : "";
      matrix.appendChild(cell);
    }
  }

  const matrixShell = document.createElement("div");
  matrixShell.className = "attention-matrix-shell";
  matrixShell.appendChild(xTokens);
  matrixShell.appendChild(yTokens);
  matrixShell.appendChild(matrix);

  const axisLabels = document.createElement("div");
  axisLabels.className = "attention-axis-labels";
  axisLabels.innerHTML = "<span>Key / attended to</span><span>Query / attending from</span>";

  frame.appendChild(matrixShell);
  section.appendChild(frame);
  section.appendChild(axisLabels);
  modalBody.appendChild(section);
};

const renderLogitsView = (payload) => {
  const section = createModalSection(payload.title);
  const list = document.createElement("div");
  list.className = "logits-list";

  const maxLogit = Math.max(...payload.items.map((item) => item.logit), 1);
  payload.items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "logits-row";

    const rank = document.createElement("div");
    rank.className = "logits-rank";
    rank.textContent = `${index + 1}`;

    const token = document.createElement("div");
    token.className = "logits-token";
    token.textContent = item.token;

    const barShell = document.createElement("div");
    barShell.className = "logits-bar-shell";
    const bar = document.createElement("div");
    bar.className = "logits-bar";
    bar.style.width = `${Math.max((item.logit / maxLogit) * 100, 6)}%`;
    barShell.appendChild(bar);

    const meta = document.createElement("div");
    meta.className = "logits-meta";
    meta.textContent = `logit ${item.logit.toFixed(2)} / p ${(item.prob * 100).toFixed(2)}%`;

    row.appendChild(rank);
    row.appendChild(token);
    row.appendChild(barShell);
    row.appendChild(meta);
    list.appendChild(row);
  });

  section.appendChild(list);
  modalBody.appendChild(section);
};

async function openNode(nodeId) {
  if (!nodeId) return;

  modal.classList.add("modal-active");
  renderModalFeedback("データを生成しています...");

  try {
    const results = engine.getNodeData(nodeId);
    modalBody.innerHTML = "";

    for (const result of results) {
      if (result.type === "tokens")    renderTokensView(result);
      else if (result.type === "attention") renderAttentionView(result);
      else if (result.type === "logits")    renderLogitsView(result);
    }

    if (!modalBody.childElementCount) {
      renderModalFeedback("このノードの可視化を表示できませんでした。");
    }
  } catch (error) {
    renderModalFeedback(error.message || "データの取得に失敗しました。");
  }
}

if (modal) {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
}

if (modalCloseBtn) {
  modalCloseBtn.addEventListener("click", closeModal);
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
    closeExampleDropdown();
  }
});

document.addEventListener("click", (event) => {
  if (exampleDropdown?.dataset.open === "true" && !exampleDropdown.contains(event.target)) {
    closeExampleDropdown();
  }
});

const renderResult = (data) => {
  if (!data) {
    resultSection.hidden = true;
    return;
  }

  if (data.graph_data) {
    renderGraph(data.graph_data);
  }

  if (data.output != null && data.prompt !== undefined) {
    enterDisplayMode(data.prompt, data.output);
  }

  resultSection.hidden = false;
};

const clearResult = () => {
  if (!currentGraphData) {
    resultSection.hidden = true;
  }
};

const resetPredictionState = ({ clearInput = false } = {}) => {
  resetPreviousJob();
  if (clearInput) {
    setPromptText("");
  }
  enterEditMode();
  clearResult();
  setStatus("");
};

const resetPreviousJob = () => {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
};

const startJob = (jobId) => {
  resetPreviousJob();
  currentSource = new EventSource(`/events/${jobId}`);
  currentSource.onmessage = (event) => {
    const update = JSON.parse(event.data);
    if (update.phase) {
      setStatus(phaseLabels[update.phase] || update.phase);
    }
    if (update.graph_data) {
      renderResult(update);
    }
    if (update.status === "done") {
      renderResult(update);
      setStatus("解析が完了しました");
      setRunning(false);
      resetPreviousJob();
    }
    if (update.status === "error") {
      setStatus(`エラー: ${update.error || "原因不明です"}`);
      setRunning(false);
      resetPreviousJob();
    }
  };
  currentSource.onerror = () => {
    setStatus("サーバーとの接続が切断されました");
    setRunning(false);
    resetPreviousJob();
  };
};

const runAnalysis = async () => {
  const prompt = getPromptValue();
  if (prompt === "") {
    setStatus("プロンプトを入力してください");
    return;
  }

  setStatus("解析を開始しています…");
  setRunning(true);
  clearResult();

  try {
    const data = await engine.run(prompt);
    renderResult(data);
    setStatus("解析が完了しました");
  } catch (error) {
    setStatus(`エラー: ${error.message || "処理に失敗しました"}`);
  } finally {
    setRunning(false);
  }
};

const initializeExamples = async () => {
  if (!exampleDropdown) return;

  try {
    const res = await fetch("./examples");
    const text = await res.text();
    examplePrompts = parseExamplePrompts(text);
  } catch (_) {
    examplePrompts = [];
  }
  populateExampleDropdown();
  setExampleDropdownVisible(examplePrompts.length > 0);
};

const initializeEngine = async () => {
  setStatus("モデルをロード中…");
  submitBtn.disabled = true;

  try {
    await engine.load(({ phase, value }) => {
      const labels = {
        meta:    "トークナイザを読み込み中…",
        lmhead:  "lm_head を読み込み中…",
        chunks:  `モデルを読み込み中… ${Math.round(value * 100)}%`,
        session: "推論セッションを準備中…",
        ready:   "",
      };
      setStatus(labels[phase] ?? "");
    });
    setStatus("");
    submitBtn.disabled = false;
  } catch (e) {
    setStatus(`モデルのロードに失敗しました: ${e.message}`);
  }
};

submitBtn.addEventListener("click", runAnalysis);
promptField.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    runAnalysis();
  }
});
promptField.addEventListener("mousedown", (event) => {
  if (promptField.dataset.mode === "display" && submitBtn.disabled !== true) {
    event.preventDefault();
    enterEditMode({ focus: true, caretToEnd: true });
  }
});
promptField.addEventListener("focus", () => {
  if (promptField.dataset.mode === "display") {
    enterEditMode({ focus: true, caretToEnd: true });
  }
});
promptField.addEventListener("paste", (event) => {
  event.preventDefault();
  const text = (event.clipboardData || window.clipboardData).getData("text");
  document.execCommand("insertText", false, text);
});
if (exampleDropdownToggle) {
  exampleDropdownToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleExampleDropdown();
  });
}
if (clearLink) {
  clearLink.addEventListener("click", () => {
    resetPredictionState({ clearInput: true });
    promptField.focus();
  });
}

initializeExamples();
initializeEngine();
resultSection.hidden = true;
renderGraph(currentGraphData);
