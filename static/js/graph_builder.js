/**
 * graph.py の JS 移植 — React Flow 向けノード/エッジデータを生成する
 */

function nodePixelX(nodeId, meta, aX0, predStep) {
  if (nodeId === 'Input')        return 0;
  if (nodeId === 'Output')       return aX0 + meta.n_layer * predStep;
  if (nodeId.startsWith('A')) {
    const layer = parseInt(nodeId.slice(1).split('.')[0], 10);
    return aX0 + layer * predStep;
  }
  if (nodeId.startsWith('MLP')) {
    const layer = parseInt(nodeId.slice(3), 10);
    return aX0 + layer * predStep + predStep / 2;
  }
  return 0;
}

function nodePixelY(nodeId, meta, yScale, yOffset, ySpacing) {
  // top_y: A.H0 と同じ高さ（Python の top_y * y_scale + y_offset に相当）
  const topY = yOffset - (meta.n_head - 1) / 2 * ySpacing * yScale;
  if (nodeId === 'Input' || nodeId === 'Output' || nodeId.startsWith('MLP')) {
    return topY;
  }
  if (nodeId.startsWith('A')) {
    const head = parseInt(nodeId.split('.H')[1], 10);
    const y = (head - (meta.n_head - 1) / 2) * ySpacing;
    return y * yScale + yOffset;
  }
  return topY;
}

/**
 * @param {object} meta           metadata.json の内容（n_layer, n_head, …）
 * @param {object} topPredictions { [nodeId]: [{token, prob}, …] }
 * @returns {{ nodes: object[], edges: object[] }}
 */
export function buildGraphData(meta, topPredictions = {}) {
  const BASE_WIDTH  = 1.5;
  const BASE_HEIGHT = 0.6;
  const X_SCALE     = 55;
  const Y_SCALE     = 92;
  const X_SPACING   = BASE_WIDTH  * 2.35;
  const Y_SPACING   = BASE_HEIGHT * 1.0;
  const PRED_STEP   = 380;

  const aX0   = X_SPACING * X_SCALE;
  const yOffset = ((meta.n_head - 1) / 2) * Y_SPACING * Y_SCALE + 20 + 5 * Y_SPACING * Y_SCALE;

  // ── ノードリスト ────────────────────────────────────────────────
  const nodeIds = ['Input'];
  for (let l = 0; l < meta.n_layer; l++) {
    for (let h = 0; h < meta.n_head; h++) nodeIds.push(`A${l}.H${h}`);
    nodeIds.push(`MLP${l}`);
  }
  nodeIds.push('Output');

  const nodes = [];
  const predictionEdges = [];

  for (const nodeId of nodeIds) {
    let fillcolor, textColor;
    if (nodeId.startsWith('A'))        { fillcolor = '#5d1f24'; textColor = '#ffd5d8'; }
    else if (nodeId.startsWith('MLP')) { fillcolor = '#0f2418'; textColor = '#e8fff1'; }
    else                               { fillcolor = '#e8eef6'; textColor = '#0d141c'; }

    nodes.push({
      id: nodeId,
      type: 'lensNode',
      position: {
        x: nodePixelX(nodeId, meta, aX0, PRED_STEP),
        y: nodePixelY(nodeId, meta, Y_SCALE, yOffset, Y_SPACING),
      },
      data: {
        label: nodeId,
        nodeStyle: {
          width:        110,
          padding:      '10px 14px',
          borderRadius: '16px',
          border:       '2px solid #e8eef6',
          background:   fillcolor,
          color:        textColor,
          fontWeight:   700,
          fontSize:     '18px',
          boxShadow:    '0 10px 22px rgba(0,0,0,0.18)',
        },
      },
      sourcePosition: 'right',
      targetPosition: 'left',
      draggable:    false,
      connectable:  false,
      selectable:   false,
    });

    // MLP ノードの予測サブノード
    const predItems = topPredictions[nodeId] ?? [];
    if (predItems.length && nodeId.startsWith('MLP')) {
      const layer  = parseInt(nodeId.slice(3), 10);
      const predX  = aX0 + 56 + layer * (360 + 20);
      const predY  = ((meta.n_head - 1) / 2) * Y_SPACING * Y_SCALE + 60 + yOffset;

      nodes.push({
        id:   `${nodeId}::predictions`,
        type: 'predictionNode',
        position: { x: predX, y: predY },
        width: 360,
        data: { parentId: nodeId, topPredictions: predItems },
        draggable:   false,
        connectable: false,
        selectable:  false,
      });
      predictionEdges.push({
        id:           `${nodeId}->predictions`,
        source:       nodeId,
        sourceHandle: 'bottom',
        target:       `${nodeId}::predictions`,
        targetHandle: 'top',
        type:         'bezier',
        animated:     false,
        selectable:   false,
        focusable:    false,
        style: { stroke: '#a6b3c2', strokeWidth: 2, opacity: 0.7 },
      });
    }
  }

  // ── エッジリスト ────────────────────────────────────────────────
  const edges = [];

  for (let h = 0; h < meta.n_head; h++) {
    edges.push(['Input', `A0.H${h}`]);
  }
  for (let l = 0; l < meta.n_layer; l++) {
    for (let h = 0; h < meta.n_head; h++) {
      edges.push([`A${l}.H${h}`, `MLP${l}`]);
    }
  }
  for (let l = 0; l < meta.n_layer - 1; l++) {
    for (let h = 0; h < meta.n_head; h++) {
      edges.push([`MLP${l}`, `A${l + 1}.H${h}`]);
    }
  }
  edges.push([`MLP${meta.n_layer - 1}`, 'Output']);

  const edgeObjects = edges.map(([src, tgt]) => ({
    id:        `${src}->${tgt}`,
    source:    src,
    target:    tgt,
    type:      'straight',
    animated:  false,
    selectable: false,
    focusable:  false,
    style: { stroke: '#a6b3c2', strokeWidth: 2.4, opacity: 0.92 },
  }));

  return { nodes, edges: [...edgeObjects, ...predictionEdges] };
}
