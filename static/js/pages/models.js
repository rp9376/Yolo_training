import { api } from "../api.js";
import { h, clear, escape, fmtBytes, fmtNum, fmtDate, spinner, emptyState, toast, confirmDialog } from "../ui.js";

const selected = new Set();
let allModels = [];

export function unmount() { selected.clear(); }

export async function render(root) {
  root.appendChild(h("div", { id: "models-list" }, spinner()));
  const list = document.getElementById("models-list");
  try { allModels = await api.get("/api/models"); }
  catch (e) { clear(list); list.appendChild(h("div", { class: "banner danger" }, e.message)); return; }
  selected.clear();
  paint(list);
}

function paint(list) {
  clear(list);
  if (!allModels.length) {
    list.appendChild(emptyState("🧠", "No trained models yet",
      "Finished training runs will appear here. Build a queue and start training."));
    return;
  }

  list.appendChild(h("div", { class: "btn-row", style: { marginBottom: "12px" } },
    h("span", { class: "small dim" }, `${selected.size} selected`),
    h("button", { class: "btn sm", disabled: !selected.size, onClick: downloadSelected },
      "⬇ Download selected"),
    h("button", { class: "btn sm danger", disabled: !selected.size, onClick: deleteSelected },
      "✕ Delete selected")));

  const allChecked = allModels.length > 0 && selected.size === allModels.length;
  const headCb = h("input", { type: "checkbox", checked: allChecked,
    onChange: (e) => {
      if (e.target.checked) allModels.forEach((m) => selected.add(m.run_name));
      else selected.clear();
      paint(list);
    } });

  const rows = allModels.map((m) => {
    const cb = h("input", { type: "checkbox", checked: selected.has(m.run_name),
      onClick: (e) => e.stopPropagation(),
      onChange: (e) => {
        if (e.target.checked) selected.add(m.run_name); else selected.delete(m.run_name);
        paint(list);
      } });
    return h("tr", { class: "clickable",
      onClick: () => { location.hash = `#/models/${encodeURIComponent(m.run_name)}`; } },
      h("td", { onClick: (e) => e.stopPropagation() }, cb),
      h("td", {}, h("b", {}, escape(m.run_name))),
      h("td", {}, `${m.family} · ${String(m.size).toUpperCase()}`),
      h("td", {}, escape(m.dataset_name || "")),
      h("td", {}, String(m.epochs ?? "—")),
      h("td", {}, String(m.best_epoch ?? "—")),
      h("td", {}, fmtNum(m.final?.map50)),
      h("td", {}, fmtNum(m.final?.map50_95)),
      h("td", {}, fmtBytes(m.size_bytes)),
      h("td", {}, fmtDate(m.mtime)));
  });

  list.appendChild(h("div", { class: "card", style: { padding: "4px 0" } },
    h("table", { class: "tbl" },
      h("thead", {}, h("tr", {},
        h("th", {}, headCb),
        h("th", {}, "Run"), h("th", {}, "Model"), h("th", {}, "Dataset"), h("th", {}, "Epochs"),
        h("th", {}, "Best"), h("th", {}, "mAP50"), h("th", {}, "mAP50-95"),
        h("th", {}, "Size"), h("th", {}, "Date"))),
      h("tbody", {}, ...rows))));
}

function downloadSelected() {
  for (const name of selected) {
    const a = document.createElement("a");
    a.href = `/api/models/${encodeURIComponent(name)}/download?which=best`;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
}

async function deleteSelected() {
  const names = [...selected];
  if (!(await confirmDialog({ title: "Delete models",
    message: `Permanently delete ${names.length} model${names.length > 1 ? "s" : ""}? This removes the run directory and weights.` })))
    return;
  let failed = 0;
  for (const name of names) {
    try { await api.del(`/api/models/${encodeURIComponent(name)}`, { silent: true }); selected.delete(name); }
    catch (e) { failed++; }
  }
  allModels = await api.get("/api/models");
  selected.clear();
  const list = document.getElementById("models-list");
  paint(list);
  if (failed) toast(`${failed} model${failed > 1 ? "s" : ""} failed to delete`, "error");
  else toast(`Deleted ${names.length} model${names.length > 1 ? "s" : ""}`, "success");
}
