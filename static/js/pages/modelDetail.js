import { api } from "../api.js";
import { h, clear, escape, fmtBytes, fmtNum, toast, confirmDialog, spinner } from "../ui.js";
import { lineChart } from "../charts.js";

let detail = null;
let activeTab = "overview";

export function unmount() { detail = null; activeTab = "overview"; }

export async function render(root, { runName }) {
  root.appendChild(spinner("Loading model…"));
  try { detail = await api.get(`/api/models/${encodeURIComponent(runName)}`); }
  catch (e) { clear(root); root.appendChild(h("div", { class: "banner danger" }, e.message)); return; }
  clear(root);

  const m = detail;
  root.appendChild(h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "16px", marginBottom: "16px" } },
    h("div", {},
      h("a", { href: "#/models", class: "small muted" }, "← Models"),
      h("h2", { style: { fontSize: "22px", marginTop: "6px" } }, escape(m.run_name)),
      h("div", { class: "small muted" }, `${m.family} · ${String(m.size).toUpperCase()} · ${escape(m.dataset_name || "")}`)),
    h("button", { class: "btn danger", onClick: () => del(m.run_name) }, "Delete run")));

  root.appendChild(h("div", { class: "grid cols-4", style: { marginBottom: "18px" } },
    metric("Best fitness", fmtNum(m.best_fitness, 4)),
    metric("Best epoch", `${m.best_epoch ?? "—"} / ${m.total_epochs ?? "—"}`),
    metric("mAP50", fmtNum(m.final?.map50)),
    metric("mAP50-95", fmtNum(m.final?.map50_95))));

  const tabsBar = h("div", { class: "tabs" },
    ...["overview", "curves", "plots", "download"].map((t) =>
      h("button", { class: t === activeTab ? "active" : "", dataset: { tab: t },
        onClick: () => { activeTab = t; paint(); } },
        t[0].toUpperCase() + t.slice(1))));
  const panel = h("div", { id: "tab-panel" });
  root.appendChild(tabsBar);
  root.appendChild(panel);
  paint();

  function paint() {
    tabsBar.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.tab === activeTab));
    clear(panel);
    if (activeTab === "overview") panel.appendChild(overview(m));
    else if (activeTab === "curves") { panel.appendChild(curves()); setTimeout(drawCurves, 0); }
    else if (activeTab === "plots") panel.appendChild(plots(m));
    else if (activeTab === "download") panel.appendChild(download(m));
  }
}

function metric(label, value) {
  return h("div", { class: "stat" }, h("div", { class: "label" }, label),
    h("div", { class: "value" }, value));
}

function overview(m) {
  const cfg = [
    ["Model", m.model], ["Init", m.args?.pretrained ? "pretrained" : "from scratch"],
    ["Epochs", m.epochs], ["Image size", `${m.imgsz}px`],
    ["Batch", m.batch], ["Dataset", m.dataset_name],
  ];
  const fin = m.final || {};
  return h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", {}, "Configuration"),
      h("div", { class: "kv" }, ...cfg.flatMap(([k, v]) =>
        [h("span", { class: "k" }, k), h("span", {}, escape(String(v ?? "—")))]))),
    h("div", { class: "card" }, h("h2", {}, "Final metrics"),
      h("div", { class: "kv" },
        h("span", { class: "k" }, "Precision"), h("span", {}, fmtNum(fin.precision)),
        h("span", { class: "k" }, "Recall"), h("span", {}, fmtNum(fin.recall)),
        h("span", { class: "k" }, "mAP50"), h("span", {}, fmtNum(fin.map50)),
        h("span", { class: "k" }, "mAP50-95"), h("span", {}, fmtNum(fin.map50_95)))),
    h("div", { class: "card", style: { gridColumn: "1 / -1" } },
      h("h2", {}, `Classes (${m.class_names?.length || 0})`),
      m.class_names?.length
        ? h("div", { class: "classlist" }, ...m.class_names.map((n, i) =>
            h("span", { class: "chip" }, h("b", {}, String(i)), escape(n))))
        : h("p", { class: "muted" }, "No class names found.")));
}

function curves() {
  return h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", {}, "Loss"), h("canvas", { class: "chart", id: "d-loss" })),
    h("div", { class: "card" }, h("h2", {}, "mAP / P / R"), h("canvas", { class: "chart", id: "d-map" })));
}

function drawCurves() {
  const s = detail.series || { epoch: [] };
  const x = s.epoch || [];
  const loss = document.getElementById("d-loss");
  const map = document.getElementById("d-map");
  if (loss) lineChart(loss, [
    { label: "box", color: "#00e5c0", data: s["train/box_loss"] || [] },
    { label: "cls", color: "#7c5cff", data: s["train/cls_loss"] || [] },
    { label: "dfl", color: "#3da5ff", data: s["train/dfl_loss"] || [] },
    { label: "val box", color: "#ff5470", data: s["val/box_loss"] || [] },
  ], { x, height: 240 });
  if (map) lineChart(map, [
    { label: "mAP50", color: "#3ddc84", data: s["metrics/mAP50(B)"] || [] },
    { label: "mAP50-95", color: "#00e5c0", data: s["metrics/mAP50-95(B)"] || [] },
    { label: "precision", color: "#ffb020", data: s["metrics/precision(B)"] || [] },
    { label: "recall", color: "#3da5ff", data: s["metrics/recall(B)"] || [] },
  ], { x, height: 240, yMin: 0, yMax: 1 });
}

function plots(m) {
  if (!m.artifacts?.length) return h("div", { class: "empty" }, "No plot images for this run.");
  const base = `/api/models/${encodeURIComponent(m.run_name)}/artifact/`;
  return h("div", { class: "plots-grid" }, ...m.artifacts.map((f) =>
    h("div", {}, h("div", { class: "small muted", style: { marginBottom: "6px" } }, escape(f)),
      h("img", { src: base + encodeURIComponent(f), alt: escape(f), loading: "lazy" }))));
}

function download(m) {
  const base = `/api/models/${encodeURIComponent(m.run_name)}`;
  const exportBox = h("div", { id: "export-box" });
  return h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", {}, "Weights"),
      h("p", { class: "muted small" }, "Download trained PyTorch weights."),
      h("div", { class: "btn-row" },
        h("a", { class: "btn primary", href: `${base}/download?which=best` }, "⬇ best.pt"),
        m.has_last ? h("a", { class: "btn", href: `${base}/download?which=last` }, "⬇ last.pt") : null)),
    h("div", { class: "card" }, h("h2", {}, "Export"),
      h("p", { class: "muted small" }, "Convert best.pt to a deployable format (may take a moment)."),
      h("div", { class: "btn-row" },
        h("button", { class: "btn", onClick: () => doExport(m.run_name, "onnx", exportBox) }, "Export ONNX"),
        h("button", { class: "btn", onClick: () => doExport(m.run_name, "torchscript", exportBox) }, "Export TorchScript")),
      exportBox));
}

async function doExport(runName, format, box) {
  clear(box);
  box.appendChild(h("div", { class: "loading", style: { padding: "12px 0" } },
    h("span", { class: "spinner" }), `Exporting ${format}…`));
  try {
    await api.post(`/api/models/${encodeURIComponent(runName)}/export`, { format });
    clear(box);
    box.appendChild(h("div", { style: { marginTop: "12px" } },
      h("a", { class: "btn primary",
        href: `/api/models/${encodeURIComponent(runName)}/download_export?format=${format}` },
        `⬇ Download ${format}`)));
    toast(`${format} export ready`, "success");
  } catch (e) { clear(box); }
}

async function del(runName) {
  if (!(await confirmDialog({ title: "Delete run",
    message: `Delete run "${runName}" and all its files? This cannot be undone.` }))) return;
  try {
    await api.del(`/api/models/${encodeURIComponent(runName)}`);
    toast("Run deleted", "success");
    location.hash = "#/models";
  } catch (e) {}
}
