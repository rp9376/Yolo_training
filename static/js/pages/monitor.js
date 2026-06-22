import { api } from "../api.js";
import { h, clear, escape, toast, confirmDialog, emptyState } from "../ui.js";
import { lineChart } from "../charts.js";

let es = null, timer = null;
let epochProgress = {};   // task_id -> {epoch, total}
let activeTaskId = null;

export function unmount() {
  if (es) es.close();
  if (timer) clearInterval(timer);
  es = timer = null;
  epochProgress = {}; activeTaskId = null;
}

export async function render(root) {
  root.appendChild(h("div", { id: "mon-head" }));
  root.appendChild(h("div", { class: "card", style: { marginBottom: "16px" } },
    h("h2", {}, "Tasks"), h("div", { id: "mon-tasks" })));
  root.appendChild(h("div", { class: "grid cols-2", style: { marginBottom: "16px" } },
    h("div", { class: "card" }, h("h2", {}, "Loss"),
      h("canvas", { class: "chart", id: "loss-chart" }), h("div", { class: "chart-legend", id: "loss-legend" })),
    h("div", { class: "card" }, h("h2", {}, "mAP / Precision / Recall"),
      h("canvas", { class: "chart", id: "map-chart" }), h("div", { class: "chart-legend", id: "map-legend" }))));
  root.appendChild(h("div", { class: "card" },
    h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" } },
      h("h2", { style: { margin: 0 } }, "Live log"),
      h("div", { class: "btn-row" },
        h("button", { class: "btn sm ghost", onClick: copyLog }, "Copy"),
        h("button", { class: "btn sm ghost", onClick: clearLog }, "Clear"))),
    h("div", { class: "console", id: "console" })));

  connectSSE();
  await refresh();
  timer = setInterval(refresh, 2000);
}

function connectSSE() {
  es = new EventSource("/api/queue/stream");
  ["log", "status", "epoch", "done"].forEach((type) => {
    es.addEventListener(type, (e) => handleEvent(type, e.data));
  });
  es.onerror = () => { /* EventSource auto-reconnects */ };
}

function handleEvent(type, raw) {
  let obj = {};
  try { obj = JSON.parse(raw); } catch (_) { obj = { message: raw }; }
  if (type === "epoch" && obj.task_id) {
    epochProgress[obj.task_id] = { epoch: obj.epoch, total: obj.total };
    logLine(`epoch ${obj.epoch}/${obj.total}` +
      (obj.metrics && obj.metrics["metrics/mAP50(B)"] != null
        ? ` — mAP50 ${Number(obj.metrics["metrics/mAP50(B)"]).toFixed(3)}` : ""), "epoch");
  } else if (type === "log") {
    if (obj.message) logLine(obj.message, "ln");
  } else if (type === "status") {
    if (obj.scope === "queue") logLine(`queue ${obj.status}`, "status");
    else if (obj.status) logLine(`${obj.name || "task"}: ${obj.status}`, "status");
  } else if (type === "done") {
    logLine(`${obj.name || "task"} ${obj.status}` +
      (obj.best_epoch != null ? ` (best epoch ${obj.best_epoch}, fitness ${obj.best_fitness})` : ""),
      obj.status === "failed" ? "err" : "done");
    refresh();
  }
}

function clearLog() {
  const c = document.getElementById("console");
  if (c) c.innerHTML = "";
}

async function copyLog() {
  const c = document.getElementById("console");
  if (!c) return;
  const text = Array.from(c.querySelectorAll(".ln")).map((n) => n.textContent).join("\n");
  if (!text) { toast("Live log is empty"); return; }
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      // Fallback for non-secure contexts (e.g. plain http over a remote IP).
      const ta = h("textarea", { style: { position: "fixed", opacity: "0" } });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    toast(`Copied ${text.split("\n").length} lines`, "success");
  } catch (e) {
    toast("Copy failed", "error");
  }
}

function logLine(text, cls) {
  const c = document.getElementById("console");
  if (!c) return;
  const atBottom = c.scrollHeight - c.scrollTop - c.clientHeight < 40;
  c.appendChild(h("div", { class: `ln ${cls}` }, text));
  while (c.childNodes.length > 600) c.removeChild(c.firstChild);
  if (atBottom) c.scrollTop = c.scrollHeight;
}

async function refresh() {
  let q, st;
  try {
    [q, st] = await Promise.all([api.get("/api/queue", { silent: true }),
      api.get("/api/queue/status", { silent: true })]);
  } catch (e) { return undefined; }

  renderHead(st);
  renderTasks(q, st);

  // choose active task for charts: running, else last non-pending.
  activeTaskId = st.running_task_id;
  if (!activeTaskId) {
    const done = q.tasks.filter((t) => t.status !== "pending");
    if (done.length) activeTaskId = done[done.length - 1].id;
  }
  if (activeTaskId) await drawCharts(activeTaskId);
  return st;
}

function renderHead(st) {
  const head = document.getElementById("mon-head");
  if (!head) return;
  clear(head);
  const left = st.running
    ? h("span", { class: "running-indicator" }, h("span", { class: "dot" }), "Queue running")
    : h("span", { class: `pill ${st.queue_status}` }, st.queue_status);
  head.appendChild(h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" } },
    h("div", { style: { display: "flex", gap: "10px", alignItems: "center" } }, left,
      h("span", { class: "chip" }, "Completed ", h("b", {}, String(st.counts.completed))),
      h("span", { class: "chip" }, "Failed ", h("b", {}, String(st.counts.failed)))),
    h("button", { class: "btn danger", disabled: !st.running, onClick: stopQueue }, "■ Stop")));
}

function renderTasks(q, st) {
  const box = document.getElementById("mon-tasks");
  if (!box) return;
  clear(box);
  if (!q.tasks.length) { box.appendChild(emptyState("📭", "No tasks", "Build a queue to start training.")); return; }
  q.tasks.forEach((t) => {
    const prog = epochProgress[t.id];
    let pct = 0, label = "";
    if (t.status === "completed") { pct = 100; label = `done · best epoch ${t.best_epoch ?? "—"}`; }
    else if (t.status === "running") {
      if (prog) { pct = Math.round((prog.epoch / prog.total) * 100); label = `epoch ${prog.epoch}/${prog.total}`; }
      else label = "starting…";
    } else if (t.status === "failed") { label = t.error || "failed"; }
    else if (t.status === "canceled") { label = "canceled"; }
    else label = "queued";

    box.appendChild(h("div", { style: { marginBottom: "12px" } },
      h("div", { style: { display: "flex", justifyContent: "space-between", marginBottom: "5px" } },
        h("span", {}, h("b", {}, escape(t.name)), " ", h("span", { class: "small dim" }, `${t.model} · ${escape(t.dataset_name || "")}`)),
        h("span", { style: { display: "flex", gap: "8px", alignItems: "center" } },
          h("span", { class: "small muted" }, label), h("span", { class: `pill ${t.status}` }, t.status))),
      h("div", { class: "progressline" }, h("span", {
        style: { width: `${pct}%`, background: t.status === "failed" ? "var(--danger)" : undefined } }))));
  });
}

async function drawCharts(taskId) {
  let m;
  try { m = await api.get(`/api/queue/tasks/${taskId}/metrics`, { silent: true }); }
  catch (e) { return; }
  const s = m.series || { epoch: [] };
  const x = s.epoch || [];

  lineChart(document.getElementById("loss-chart"), [
    { label: "box", color: "#00e5c0", data: s["train/box_loss"] || [] },
    { label: "cls", color: "#7c5cff", data: s["train/cls_loss"] || [] },
    { label: "dfl", color: "#3da5ff", data: s["train/dfl_loss"] || [] },
  ], { x, height: 200 });
  legend("loss-legend", [["box", "#00e5c0"], ["cls", "#7c5cff"], ["dfl", "#3da5ff"]]);

  lineChart(document.getElementById("map-chart"), [
    { label: "mAP50", color: "#3ddc84", data: s["metrics/mAP50(B)"] || [] },
    { label: "mAP50-95", color: "#00e5c0", data: s["metrics/mAP50-95(B)"] || [] },
    { label: "precision", color: "#ffb020", data: s["metrics/precision(B)"] || [] },
    { label: "recall", color: "#3da5ff", data: s["metrics/recall(B)"] || [] },
  ], { x, height: 200, yMin: 0, yMax: 1 });
  legend("map-legend", [["mAP50", "#3ddc84"], ["mAP50-95", "#00e5c0"], ["precision", "#ffb020"], ["recall", "#3da5ff"]]);
}

function legend(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  clear(el);
  items.forEach(([label, color]) => el.appendChild(
    h("span", { class: "it" }, h("span", { class: "sw", style: { background: color } }), label)));
}

async function stopQueue() {
  if (!(await confirmDialog({ title: "Stop training",
    message: "Stop training now? The model in progress will be killed immediately and its "
      + "progress lost. This cannot be undone." }))) return;
  console.log("[stop] POST /api/queue/stop …");
  toast("Stopping…");
  try {
    const res = await api.post("/api/queue/stop");
    console.log("[stop] response:", res);
  } catch (e) {
    console.error("[stop] request failed:", e);
    toast("Stop request failed — see console", "error");
  }
  const st = await refresh();
  console.log("[stop] status after refresh:", st);
}
