// 尝尝咸淡 · 检索工坊 —— 对话 / 检索溯源 / 知识图谱 / 监控
const $ = (s) => document.querySelector(s);
const messages = $("#messages");
const inspector = $("#inspector");
const graph = $("#graph");
const graphEmpty = $("#graphEmpty");
const SVGNS = "http://www.w3.org/2000/svg";

const STRAT_LABEL = { hybrid_traditional: "混合检索", graph_rag: "图谱检索", combined: "组合检索" };
const ROUTE_LABEL = { vector: "向量", bm25: "BM25", dual_level: "双层", graph: "图谱", graph_relation: "图谱·关系", graph_path: "图谱·路径", knowledge_subgraph: "图谱·子图" };

/* ---------- 对话 ---------- */
$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#input").value.trim();
  if (!q) return;
  $("#input").value = "";
  addMsg("user", q);
  const thinking = addMsg("assistant", '<span class="skeleton"><span class="bar" style="width:120px"></span> 正在检索与生成…</span>');
  $("#send").disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (res.status === 429) {
      thinking.querySelector("p").innerHTML = "请求过于频繁，请稍后再试。";
      thinking.querySelector("p").style.color = "var(--danger)";
      return;
    }
    const data = await res.json();
    thinking.remove();
    renderAnswer(data, q);
  } catch (err) {
    thinking.querySelector("p").textContent = "网络错误：" + err.message;
    thinking.querySelector("p").style.color = "var(--danger)";
  } finally {
    $("#send").disabled = false;
  }
});

function addMsg(role, html) {
  const div = document.createElement("div");
  div.className = "msg msg-" + role;
  div.innerHTML = `<p>${html}</p>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function renderAnswer(data, q) {
  const refused = data.refused;
  const tags = [];
  if (data.strategy) tags.push(`<span class="tag t-strat">${STRAT_LABEL[data.strategy] || data.strategy}</span>`);
  if (data.cache_hit) tags.push(`<span class="tag t-hit">缓存命中 ${data.cache_hit}</span>`);
  if (data.latency_ms != null) tags.push(`<span class="tag t-think">${data.latency_ms} ms</span>`);
  if (refused) tags.push(`<span class="tag t-refuse">拒答</span>`);
  const wrap = document.createElement("div");
  wrap.className = "msg msg-assistant";
  wrap.innerHTML = `<p>${escapeHtml(data.answer || "")}</p>${tags.length ? `<div class="msg-tags">${tags.join("")}</div>` : ""}`;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;

  renderInspector(data);
  if (!refused && data.sources && data.sources.length) {
    renderGraph(data.sources.map((s) => s.recipe_name).filter(Boolean));
  } else {
    clearGraph();
  }
}

/* ---------- 检视器 ---------- */
function renderInspector(data) {
  if (data.refused) {
    inspector.innerHTML = `
      <div class="insp-row"><span class="insp-strat" style="color:var(--danger)">拒答</span></div>
      <p class="insp-sub">原因：<code>${escapeHtml(data.reason || "")}</code>　置信度 <b>${fmt(data.confidence)}</b></p>
      <p class="insp-sub" style="margin-top:14px;max-width:46ch">这道菜不在知识库里，系统选择诚实拒答而非编造（防幻觉，详见 P2-3b）。</p>`;
    return;
  }
  const strat = STRAT_LABEL[data.strategy] || data.strategy || "—";
  const hit = data.cache_hit ? `<span class="tag t-hit">缓存 ${data.cache_hit}</span>` : '<span class="tag t-think">未命中缓存</span>';
  let stagesHtml = "";
  if (data.stages) {
    const st = data.stages;
    const total = (st.gate_ms || 0) + (st.retrieve_ms || 0) + (st.generate_ms || 0) || 1;
    stagesHtml = `
      <div class="stages">
        <div class="stages-title">每步耗时（瓶颈在哪）</div>
        ${stageBar("把关 gate", st.gate_ms, total, "f-gate")}
        ${stageBar("检索 retrieve", st.retrieve_ms, total, "")}
        ${stageBar("生成 generate", st.generate_ms, total, "f-gen")}
      </div>`;
  } else if (data.cache_hit) {
    stagesHtml = `
      <div class="stages">
        <div class="stages-title">每步耗时 · 全部跳过</div>
        <div class="skip-row"><span class="skip-dot"></span>拒答闸门（向量查）<span class="skip-tag">跳过</span></div>
        <div class="skip-row"><span class="skip-dot"></span>检索（路由 + 三路并发）<span class="skip-tag">跳过</span></div>
        <div class="skip-row"><span class="skip-dot"></span>LLM 生成<span class="skip-tag">跳过</span></div>
        <p class="insp-sub" style="margin-top:12px;max-width:42ch">命中 ${escapeHtml(data.cache_hit)} —— 整条链路被旁路，这就是缓存把重复/相似查询压到亚毫秒的原理。</p>
      </div>`;
  }
  const srcs = data.sources || [];
  const srcHtml = srcs.length ? `
    <div class="section-title">命中菜谱 <span class="count">${srcs.length} 条</span></div>
    <table class="src-table">
      <thead><tr><th>菜谱</th><th>来源路</th><th style="text-align:right">分数</th></tr></thead>
      <tbody>
        ${srcs.map((s) => `<tr>
          <td class="src-name">${escapeHtml(s.recipe_name || "—")}</td>
          <td><span class="src-route ${routeClass(s.search_type)}">${ROUTE_LABEL[s.search_type] || s.search_type || "—"}</span></td>
          <td class="src-score">${s.score != null ? Number(s.score).toFixed(3) : "—"}</td>
        </tr>`).join("")}
      </tbody>
    </table>` : "";
  inspector.innerHTML = `
    <div class="insp-row">
      <span class="insp-strat"><code>${strat}</code></span>
      ${hit}
      <span class="tag t-think">${data.latency_ms ?? "—"} ms</span>
    </div>
    <p class="insp-sub">路由策略决定走混合检索还是图谱检索；命中缓存则跳过整条链路。</p>
    ${stagesHtml}
    ${srcHtml}`;
}

function stageBar(name, ms, total, cls) {
  const pct = Math.max(2, Math.round(((ms || 0) / total) * 100));
  return `<div class="stage">
    <span class="stage-name">${name}</span>
    <span class="stage-bar"><span class="stage-fill ${cls}" style="width:${pct}%"></span></span>
    <span class="stage-ms">${fmt(ms)} ms</span>
  </div>`;
}
function routeClass(t) { return t ? "r-" + (t.replace(/_enhanced|_level/, "").replace("graph_relation", "graph")) : ""; }

/* ---------- 知识图谱（确定性二分图：菜谱左 / 食材右）---------- */
function clearGraph() {
  graph.innerHTML = ""; graphEmpty.style.display = "flex";
}
async function renderGraph(recipes) {
  graphEmpty.style.display = "none";
  graph.innerHTML = "";
  try {
    const res = await fetch("/api/graph", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recipes }),
    });
    const g = await res.json();
    drawGraph(g);
  } catch (e) {
    graphEmpty.textContent = "图谱加载失败";
    graphEmpty.style.display = "flex";
  }
}

function drawGraph(g) {
  const wrap = $("#graphWrap");
  const W = wrap.clientWidth || 420;
  const H = wrap.clientHeight || 360;
  graph.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!g.nodes || !g.nodes.length) { graphEmpty.style.display = "flex"; graphEmpty.textContent = "无图谱数据"; return; }

  const recipes = g.nodes.filter((n) => n.type === "Recipe");
  const ingredients = g.nodes.filter((n) => n.type === "Ingredient");
  const catsFrom = {}; // recipe id -> [cat label]
  (g.edges || []).forEach((e) => {
    if (e.type === "CATEGORY") {
      const catNode = g.nodes.find((n) => n.id === e.target && n.type === "Category") || g.nodes.find((n) => n.id === e.target);
      if (catNode) (catsFrom[e.source] = catsFrom[e.source] || []).push(catNode.label);
    }
  });
  const reqEdges = (g.edges || []).filter((e) => e.type === "REQUIRES");

  const padY = 40, padX = Math.max(100, Math.min(200, W * 0.15));
  const pos = (nodes, x) => {
    const m = {};
    nodes.forEach((n, i) => {
      m[n.id] = { x, y: nodes.length === 1 ? H / 2 : padY + (i * (H - 2 * padY)) / (nodes.length - 1) };
    });
    return m;
  };
  const rp = pos(recipes, padX);
  const ip = pos(ingredients, W - padX);

  // 边（先画，压在节点下）
  reqEdges.forEach((e) => {
    const a = rp[e.source], b = ip[e.target];
    if (!a || !b) return;
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", a.x); ln.setAttribute("y1", a.y);
    ln.setAttribute("x2", b.x); ln.setAttribute("y2", b.y);
    ln.setAttribute("class", "edge");
    graph.appendChild(ln);
  });
  // 节点 + 标签
  const node = (p, r, fill, label, anchor, sub) => {
    const c = document.createElementNS(SVGNS, "circle");
    c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", r);
    c.setAttribute("fill", fill);
    graph.appendChild(c);
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", p.x); t.setAttribute("y", p.y + r + 14);
    t.setAttribute("text-anchor", "middle"); t.setAttribute("class", "node-label");
    t.textContent = truncate(label, 8);
    graph.appendChild(t);
    if (sub && sub.length) {
      const ts = document.createElementNS(SVGNS, "text");
      ts.setAttribute("x", p.x); ts.setAttribute("y", p.y + r + 28);
      ts.setAttribute("text-anchor", "middle");
      ts.setAttribute("fill", "var(--ink-3)"); ts.style.fontSize = "10px";
      ts.textContent = truncate(sub.join("/"), 8);
      graph.appendChild(ts);
    }
  };
  recipes.forEach((n) => node(rp[n.id], 7, "var(--accent)", n.label, "middle", catsFrom[n.id]));
  ingredients.forEach((n) => node(ip[n.id], 5, "var(--g-ingr)", n.label, "middle"));
}

/* ---------- 监控条 ---------- */
async function pollStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    if (!s || !s.total_requests && s.total_requests !== 0) return;
    const route = Object.entries(s.route_distribution || {}).map(([k, v]) => `${STRAT_LABEL[k] || k} ${v}`).join(" · ") || "—";
    $("#metrics").innerHTML = [
      `<span class="chip">QPS <b>${fmt(s.qps)}</b></span>`,
      `<span class="chip ${s.p99_ms > 5000 ? "warn" : ""}">p99 <b>${fmt(s.p99_ms)}ms</b></span>`,
      `<span class="chip hit">缓存命中 <b>${Math.round((s.cache_hit_rate || 0) * 100)}%</b></span>`,
      `<span class="chip">请求 <b>${s.total_requests}</b></span>`,
      `<span class="chip">${route}</span>`,
    ].join("");
  } catch (e) { /* 静默 */ }
}
setInterval(pollStats, 3000);
pollStats();

/* ---------- 工具 ---------- */
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmt(n) { return n == null ? "—" : (Number(n) >= 100 ? Math.round(Number(n)) : Number(n).toFixed(1)); }
function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : (s || ""); }
