import { api } from "../api.js";
import { h, clear, fmtBytes, fmtPct, escape } from "../ui.js";
import { lineChart, ringGauge } from "../charts.js";

let timer = null;
let healthTimer = null;
const MAXP = 60; // rolling window (~60s at 1s poll)
let history = { cpu: [], gpus: [] };

export function unmount() {
  if (timer) clearInterval(timer);
  if (healthTimer) clearInterval(healthTimer);
  timer = healthTimer = null;
  history = { cpu: [], gpus: [] };
}

export async function render(root) {
  const tiles = h("div", { class: "grid cols-4" });
  const gpuWrap = h("div", { class: "grid cols-2", style: { marginTop: "16px" } });
  const noGpuBanner = h("div");
  const chartCard = h("div", { class: "card", style: { marginTop: "16px" } },
    h("h2", {}, "Utilization (last 60s)"),
    h("canvas", { class: "chart", id: "util-chart" }),
    h("div", { class: "chart-legend", id: "util-legend" }));
  const sysCard = h("div", { class: "card", style: { marginTop: "16px" } },
    h("h2", {}, "System"), h("div", { id: "sys-body", class: "kv" }));

  root.appendChild(tiles);
  root.appendChild(noGpuBanner);
  root.appendChild(gpuWrap);
  root.appendChild(chartCard);
  root.appendChild(sysCard);

  let gpuCount = -1;

  async function tick() {
    let s;
    try { s = await api.get("/api/hardware", { silent: true }); }
    catch (e) { return; }

    // stat tiles
    clear(tiles);
    tiles.appendChild(statTile("CPU", `${fmtPct(s.cpu.percent)}`,
      `${s.cpu.cores} cores`, s.cpu.percent));
    tiles.appendChild(statTile("Memory", fmtBytes(s.memory.used),
      `of ${fmtBytes(s.memory.total)}`, s.memory.percent));
    tiles.appendChild(statTile("Disk (runs)", fmtBytes(s.disk.used),
      `of ${fmtBytes(s.disk.total)}`, s.disk.percent));
    tiles.appendChild(statTile("GPUs", String(s.gpus.length),
      s.gpu_backend === "none" ? "none detected" : `via ${s.gpu_backend}`, null));

    // gpu banner
    clear(noGpuBanner);
    if (s.gpus.length === 0) {
      noGpuBanner.appendChild(h("div", { class: "banner warn", style: { marginTop: "16px" } },
        "No GPU detected — training will run on CPU (slow). GPU detection uses pynvml or nvidia-smi."));
    }

    // gpu cards (rebuild if count changed)
    if (s.gpus.length !== gpuCount) {
      gpuCount = s.gpus.length;
      clear(gpuWrap);
      s.gpus.forEach((g) => gpuWrap.appendChild(gpuCard(g)));
      history.gpus = s.gpus.map(() => []);
    }
    s.gpus.forEach((g) => updateGpuCard(g));

    // rolling history
    history.cpu.push(s.cpu.percent);
    if (history.cpu.length > MAXP) history.cpu.shift();
    s.gpus.forEach((g, i) => {
      if (!history.gpus[i]) history.gpus[i] = [];
      history.gpus[i].push(g.util);
      if (history.gpus[i].length > MAXP) history.gpus[i].shift();
    });
    drawUtilChart(s);
  }

  function drawUtilChart(s) {
    const canvas = document.getElementById("util-chart");
    if (!canvas) return;
    const colors = ["#7c5cff", "#3da5ff", "#ffb020", "#ff5470", "#3ddc84"];
    const series = [{ label: "CPU", color: "#00e5c0", data: history.cpu.slice() }];
    s.gpus.forEach((g, i) => series.push({
      label: `GPU${g.index}`, color: colors[i % colors.length],
      data: (history.gpus[i] || []).slice(),
    }));
    lineChart(canvas, series, { height: 220, yMin: 0, yMax: 100 });
    const legend = document.getElementById("util-legend");
    clear(legend);
    series.forEach((se) => legend.appendChild(h("span", { class: "it" },
      h("span", { class: "sw", style: { background: se.color } }), se.label)));
  }

  async function tickSys() {
    let hd;
    try { hd = await api.get("/api/health", { silent: true }); } catch (e) { return; }
    const body = document.getElementById("sys-body");
    if (!body) return;
    clear(body);
    const rows = [
      ["Python", hd.python],
      ["Torch", hd.torch || "not installed"],
      ["CUDA available", hd.cuda == null ? "—" : (hd.cuda ? "yes" : "no")],
      ["Ultralytics", hd.ultralytics || "not installed"],
      ["GPU backend", hd.gpu_backend],
    ];
    rows.forEach(([k, v]) => {
      body.appendChild(h("span", { class: "k" }, k));
      body.appendChild(h("span", {}, String(v)));
    });
  }

  await tick();
  await tickSys();
  timer = setInterval(tick, 1000);
  healthTimer = setInterval(tickSys, 5000);
}

function statTile(label, value, sub, pct) {
  return h("div", { class: "stat" },
    h("div", { class: "label" }, label),
    h("div", { class: "value" }, value, sub ? h("small", {}, ` ${sub}`) : null),
    pct != null ? h("div", { class: "bar" },
      h("span", { style: { width: `${Math.min(100, pct)}%` } })) : null);
}

function gpuCard(g) {
  const card = h("div", { class: "card gpu-card", dataset: { gpu: String(g.index) } },
    h("div", { class: "gpu-head" },
      h("span", { class: "gpu-name" }, `GPU ${g.index} · ${escape(g.name)}`)),
    h("div", { class: "gpu-body" },
      h("canvas", { class: "gpu-ring", dataset: { ring: String(g.index) } }),
      h("div", { class: "gpu-meta", dataset: { meta: String(g.index) } })));
  return card;
}

function updateGpuCard(g) {
  const ring = document.querySelector(`canvas[data-ring="${g.index}"]`);
  const meta = document.querySelector(`div[data-meta="${g.index}"]`);
  if (!ring || !meta) return;
  ringGauge(ring, g.util, "util");
  const memPct = g.mem_total ? (g.mem_used / g.mem_total) * 100 : 0;
  clear(meta);
  meta.appendChild(metaRow("Memory", `${fmtBytes(g.mem_used)} / ${fmtBytes(g.mem_total)}`));
  meta.appendChild(h("div", { class: "bar" }, h("span", { style: { width: `${memPct}%` } })));
  meta.appendChild(metaRow("Temp", `${g.temp}°C`));
  meta.appendChild(metaRow("Power", `${g.power} / ${g.power_limit} W`));
}

function metaRow(k, v) {
  return h("div", { class: "row" }, h("span", {}, k), h("span", {}, v));
}
