import { api } from "../api.js";
import { h, clear, escape, toast, confirmDialog, spinner, emptyState } from "../ui.js";

let timer = null;
let weights = {}, datasets = [], gpuCount = 0;
let running = false;
let inCustomEpochs = false;
const state = { family: "yolov26", size: "x", init: "scratch", dataset: "",
  epochs: 100, batch: -1, imgsz: 640, device: "auto" };

export function unmount() { if (timer) clearInterval(timer); timer = null; }

export async function render(root) {
  try {
    [weights, datasets] = await Promise.all([api.get("/api/weights"), api.get("/api/datasets")]);
    const hw = await api.get("/api/hardware", { silent: true }).catch(() => ({ gpus: [] }));
    gpuCount = (hw.gpus || []).length;
  } catch (e) {
    root.appendChild(h("div", { class: "banner danger" }, e.message)); return;
  }
  if (datasets.length && !state.dataset) state.dataset = datasets[0].name;

  root.appendChild(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", {}, "Add training task"), buildForm()),
    h("div", { class: "card" }, h("h2", {}, "Run control"), h("div", { id: "run-control" }, spinner()))));
  root.appendChild(h("div", { class: "card", style: { marginTop: "16px" } },
    h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } },
      h("h2", { style: { margin: 0 } }, "Queue"),
      h("div", { class: "btn-row" },
        h("button", { class: "btn sm", onClick: () => clearQueue("completed") }, "Clear completed"),
        h("button", { class: "btn sm danger", onClick: () => clearQueue("all") }, "Clear all"))),
    h("div", { id: "queue-list", style: { marginTop: "12px" } }, spinner())));

  await refresh();
  timer = setInterval(refresh, 2500);
}

function seg(options, value, onPick, labels, disabledFn) {
  return h("div", { class: "seg" }, ...options.map((o) => {
    const btn = h("button", {
      class: value === o ? "sel" : "",
      disabled: disabledFn && disabledFn(o),
      onClick: (e) => { e.preventDefault(); onPick(o); },
    }, labels ? labels(o) : String(o));
    btn._segValue = o;
    return btn;
  }));
}

function activateSeg(id, value) {
  const box = document.getElementById(id);
  if (!box) return;
  box.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("sel", b._segValue === value));
}

function buildForm() {
  const form = h("form", { onSubmit: (e) => { e.preventDefault(); addTask(); } });

  const familyBox = h("div", { class: "field" }, h("label", {}, "Family"),
    h("div", { id: "f-family" }));
  const sizeBox = h("div", { class: "field" }, h("label", {}, "Model size"), h("div", { id: "f-size" }));
  const initBox = h("div", { class: "field" }, h("label", {}, "Initialization"), h("div", { id: "f-init" }));
  const dsBox = h("div", { class: "field" }, h("label", {}, "Dataset"), h("div", { id: "f-ds" }));
  const epochsBox = h("div", { class: "field" }, h("label", {}, "Epochs"), h("div", { id: "f-epochs" }));
  const batchBox = h("div", { class: "field" }, h("label", {}, "Batch size"), h("div", { id: "f-batch" }));
  const imgszBox = h("div", { class: "field" }, h("label", {}, "Image size"), h("div", { id: "f-imgsz" }));
  const deviceBox = h("div", { class: "field" }, h("label", {}, "Device"), h("div", { id: "f-device" }));
  const preview = h("div", { class: "chip", id: "name-preview", style: { marginBottom: "12px" } });
  const submit = h("button", { class: "btn primary", type: "submit", style: { width: "100%" } }, "+ Add to queue");

  form.append(familyBox, sizeBox, initBox, dsBox, epochsBox, batchBox, imgszBox, deviceBox, preview, submit);
  setTimeout(paintForm, 0);
  return form;
}

function paintForm() {
  const SIZES = ["n", "s", "m", "l", "x"];
  const SIZE_LABELS = { n: "Nano", s: "Small", m: "Medium", l: "Large", x: "XLarge" };

  setSeg("f-family", ["yolov26", "yolov8"], state.family, (v) => { state.family = v; fixInit(); paintForm(); },
    (v) => v === "yolov26" ? "YOLOv26" : "YOLOv8");
  setSeg("f-size", SIZES, state.size, (v) => { state.size = v; fixInit(); paintForm(); },
    (v) => `${v.toUpperCase()}`);

  // init: disable pretrained if no weight available
  const avail = weights?.[state.family]?.[state.size]?.available;
  setSeg("f-init", ["pretrained", "scratch"], state.init, (v) => { state.init = v; paintForm(); },
    (v) => v === "pretrained" ? (avail ? "Pretrained" : "Pretrained (n/a)") : "From scratch",
    (v) => v === "pretrained" && !avail);

  // dataset select
  const dsBox = document.getElementById("f-ds"); clear(dsBox);
  if (!datasets.length) {
    dsBox.appendChild(h("div", { class: "banner warn", style: { margin: 0 } },
      "No datasets. Add one on the Datasets page first."));
  } else {
    const sel = h("select", { onChange: (e) => { state.dataset = e.target.value; updatePreview(); } },
      ...datasets.map((d) => h("option", { value: d.name, selected: d.name === state.dataset }, d.name)));
    dsBox.appendChild(sel);
  }

  // epochs presets + custom
  const epBox = document.getElementById("f-epochs"); clear(epBox);
  const presets = [5, 50, 100, 200, 300, 500];
  const isPreset = presets.includes(state.epochs) && !inCustomEpochs;
  epBox.appendChild(seg([...presets, "custom"], isPreset ? state.epochs : "custom",
    (v) => {
      if (v === "custom") { inCustomEpochs = true; paintForm(); }
      else { inCustomEpochs = false; state.epochs = v; updatePreview(); paintForm(); }
    },
    (v) => v === "custom" ? "Custom" : String(v)));
  if (!isPreset) {
    if (!epBox.querySelector("input[type=number]"))
      epBox.appendChild(h("input", { type: "number", min: "1", value: String(state.epochs || 100),
        style: { marginTop: "8px" },
        onInput: (e) => { state.epochs = parseInt(e.target.value) || 1; updatePreview(); } }));
  }

  // Multi-GPU (DDP) can't use AutoBatch — force a concrete batch that's a
  // multiple of the GPU count (8 images per GPU, ultralytics' default).
  const multiGpu = state.device === "all" && gpuCount > 1;
  if (multiGpu && state.batch < 1) state.batch = gpuCount * 8;
  setSeg("f-batch", [-1, 16, 32, 64, 128], state.batch,
    (v) => { state.batch = v; activateSeg("f-batch", v); },
    (v) => v === -1 ? "Auto" : String(v),
    (v) => multiGpu && v === -1);
  const batchBox = document.getElementById("f-batch");
  if (multiGpu && batchBox && !batchBox.querySelector(".small")) {
    batchBox.appendChild(h("div", { class: "small dim", style: { marginTop: "6px" } },
      `Auto batch is unavailable with multiple GPUs. Pick a value that's a multiple of ${gpuCount}.`));
  }
  setSeg("f-imgsz", [640, 1280], state.imgsz,
    (v) => { state.imgsz = v; activateSeg("f-imgsz", v); },
    (v) => `${v}px`);

  const devices = [
    "auto", "cpu",
    ...(gpuCount > 1 ? ["all"] : []),
    ...Array.from({ length: gpuCount }, (_, i) => String(i)),
  ];
  setSeg("f-device", devices, state.device,
    (v) => { state.device = v; paintForm(); },
    (v) => v === "auto" ? "Auto" : v === "cpu" ? "CPU" : v === "all" ? "All GPUs" : `GPU ${v}`);

  updatePreview();
}

function setSeg(id, options, value, onPick, labels, disabledFn) {
  const box = document.getElementById(id);
  if (!box) return;
  clear(box);
  box.appendChild(seg(options, value, onPick, labels, disabledFn));
}

function fixInit() {
  const avail = weights?.[state.family]?.[state.size]?.available;
  if (state.init === "pretrained" && !avail) state.init = "scratch";
}

function updatePreview() {
  const el = document.getElementById("name-preview");
  if (el) el.innerHTML = `Run name: <b>${escape(state.size)}_e${state.epochs}_&lt;timestamp&gt;</b>`;
}

async function addTask() {
  if (!state.dataset) { toast("Select a dataset first", "error"); return; }
  try {
    await api.post("/api/queue/tasks", {
      family: state.family, size: state.size, init: state.init, dataset: state.dataset,
      epochs: state.epochs, batch: state.batch, imgsz: state.imgsz, device: state.device,
    });
    toast("Task added to queue", "success");
    refresh();
  } catch (e) { /* toast shown */ }
}

async function refresh() {
  let q, st;
  try {
    [q, st] = await Promise.all([api.get("/api/queue", { silent: true }),
      api.get("/api/queue/status", { silent: true })]);
  } catch (e) { return; }
  running = st.running;
  renderRunControl(q, st);
  renderQueueList(q);
}

function renderRunControl(q, st) {
  const box = document.getElementById("run-control");
  if (!box) return;
  clear(box);
  const pending = q.tasks.filter((t) => t.status === "pending").length;
  box.appendChild(h("div", { class: "kv", style: { marginBottom: "14px" } },
    h("span", { class: "k" }, "Status"), h("span", { class: `pill ${st.queue_status}` }, st.queue_status),
    h("span", { class: "k" }, "Pending"), h("span", {}, String(pending)),
    h("span", { class: "k" }, "Completed"), h("span", {}, String(st.counts.completed)),
    h("span", { class: "k" }, "Failed"), h("span", {}, String(st.counts.failed))));
  box.appendChild(h("div", { class: "btn-row" },
    h("button", { class: "btn primary", disabled: running || pending === 0,
      onClick: startQueue }, running ? "Running…" : "▶ Start queue"),
    h("button", { class: "btn danger", disabled: !running, onClick: stopQueue }, "■ Stop"),
    h("a", { class: "btn ghost", href: "#/monitor" }, "Open Monitor")));
}

function renderQueueList(q) {
  const list = document.getElementById("queue-list");
  if (!list) return;
  clear(list);
  if (!q.tasks.length) { list.appendChild(emptyState("📋", "Queue is empty", "Add a task above.")); return; }

  const tbody = h("tbody", {});
  q.tasks.forEach((t, idx) => {
    const canMutate = !running && t.status === "pending";
    const tr = h("tr", { draggable: canMutate, dataset: { id: t.id } },
      h("td", { class: "dim", style: { cursor: canMutate ? "grab" : "default" } }, canMutate ? "⠿" : String(idx + 1)),
      h("td", {}, h("b", {}, escape(t.name))),
      h("td", {}, `${t.model} · ${t.init}`),
      h("td", {}, escape(t.dataset_name || "")),
      h("td", {}, `e${t.epochs} · b${t.batch === -1 ? "auto" : t.batch} · ${t.imgsz}px`),
      h("td", {}, h("span", { class: `pill ${t.status}` }, t.status)),
      h("td", { style: { textAlign: "right" } },
        h("button", { class: "btn sm danger", disabled: !canMutate,
          onClick: () => removeTask(t.id) }, "✕")));
    if (canMutate) attachDnd(tr, q);
    tbody.appendChild(tr);
  });

  list.appendChild(h("table", { class: "tbl" },
    h("thead", {}, h("tr", {},
      h("th", { style: { width: "30px" } }, "#"), h("th", {}, "Run name"), h("th", {}, "Model"),
      h("th", {}, "Dataset"), h("th", {}, "Config"), h("th", {}, "Status"),
      h("th", { style: { textAlign: "right" } }, ""))),
    tbody));
}

let dragId = null;
function attachDnd(tr, q) {
  tr.addEventListener("dragstart", () => { dragId = tr.dataset.id; tr.style.opacity = "0.4"; });
  tr.addEventListener("dragend", () => { tr.style.opacity = "1"; });
  tr.addEventListener("dragover", (e) => e.preventDefault());
  tr.addEventListener("drop", async (e) => {
    e.preventDefault();
    const targetId = tr.dataset.id;
    if (!dragId || dragId === targetId) return;
    const ids = q.tasks.map((t) => t.id);
    const from = ids.indexOf(dragId), to = ids.indexOf(targetId);
    ids.splice(to, 0, ids.splice(from, 1)[0]);
    try { await api.post("/api/queue/reorder", { order: ids }); refresh(); }
    catch (err) { /* toast shown */ }
  });
}

async function removeTask(id) {
  try { await api.del(`/api/queue/tasks/${id}`); refresh(); } catch (e) { /* toast */ }
}

async function startQueue() {
  try { await api.post("/api/queue/start"); toast("Queue started", "success");
    setTimeout(() => { location.hash = "#/monitor"; }, 400); }
  catch (e) { /* toast */ }
}

async function stopQueue() {
  if (!(await confirmDialog({ title: "Stop training",
    message: "Stop training now? The model in progress will be killed immediately and its "
      + "progress lost. This cannot be undone." }))) return;
  try { await api.post("/api/queue/stop"); toast("Stopping…"); refresh(); } catch (e) {}
}

async function clearQueue(scope) {
  if (scope === "all") {
    const msg = running
      ? "This will stop the training currently in progress (killing it immediately, progress "
        + "lost) and remove every task from the queue. This cannot be undone."
      : "Remove all tasks from the queue? This cannot be undone.";
    if (!(await confirmDialog({ title: "Clear all", message: msg }))) return;
  }
  try { await api.post("/api/queue/clear", { scope }); toast("Cleared"); refresh(); } catch (e) {}
}
