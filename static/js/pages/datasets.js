import { api } from "../api.js";
import { h, clear, escape, fmtBytes, toast, modal, confirmDialog, spinner, emptyState } from "../ui.js";

let container = null;

export function unmount() { container = null; }

export async function render(root) {
  container = root;
  root.appendChild(h("div", { class: "btn-row", style: { marginBottom: "16px" } },
    h("button", { class: "btn primary", onClick: openAddModal }, "+ Add dataset"),
    h("button", { class: "btn", onClick: reload }, "↻ Refresh")));
  root.appendChild(h("div", { id: "ds-list" }, spinner()));
  await reload();
}

async function reload() {
  const list = document.getElementById("ds-list");
  if (!list) return;
  clear(list); list.appendChild(spinner());
  let items;
  try { items = await api.get("/api/datasets"); }
  catch (e) { clear(list); list.appendChild(h("div", { class: "banner danger" }, e.message)); return; }
  clear(list);
  if (!items.length) {
    list.appendChild(emptyState("📁", "No datasets yet",
      "Upload a .zip or register a server path to get started."));
    return;
  }
  const rows = items.map((d) => h("tr", {},
    h("td", {}, h("b", {}, escape(d.name)),
      !d.valid ? h("ul", { class: "issues" }, ...d.issues.map((i) => h("li", {}, escape(i)))) : null),
    h("td", {}, String(d.nc)),
    h("td", {}, `${d.counts.train} / ${d.counts.valid} / ${d.counts.test}`),
    h("td", {}, fmtBytes(d.size_bytes)),
    h("td", {}, h("span", { class: "chip" }, d.source)),
    h("td", {}, d.valid ? h("span", { class: "pill completed" }, "valid")
      : h("span", { class: "pill failed" }, "issues")),
    h("td", { style: { textAlign: "right" } },
      h("div", { class: "btn-row", style: { justifyContent: "flex-end" } },
        h("button", { class: "btn sm ghost", onClick: () => viewClasses(d) }, "Classes"),
        h("button", { class: "btn sm ghost", onClick: () => revalidate(d.name) }, "Validate"),
        h("button", { class: "btn sm danger", onClick: () => del(d.name) }, "Delete")))));

  list.appendChild(h("div", { class: "card", style: { padding: "4px 0" } },
    h("table", { class: "tbl" },
      h("thead", {}, h("tr", {},
        h("th", {}, "Name"), h("th", {}, "Classes"), h("th", {}, "Train/Valid/Test"),
        h("th", {}, "Size"), h("th", {}, "Source"), h("th", {}, "Status"),
        h("th", { style: { textAlign: "right" } }, "Actions"))),
      h("tbody", {}, ...rows))));
}

function viewClasses(d) {
  modal({ title: `${d.name} — ${d.nc} classes`,
    body: d.names.length
      ? h("div", { class: "classlist" }, ...d.names.map((n, i) =>
          h("span", { class: "chip" }, h("b", {}, String(i)), escape(n))))
      : h("p", { class: "muted" }, "No class names found.") });
}

async function revalidate(name) {
  try {
    const v = await api.post(`/api/datasets/${encodeURIComponent(name)}/validate`);
    if (v.valid) toast(`${name}: valid`, "success");
    else toast(`${name}: ${v.issues.join("; ")}`, "error", 6000);
    reload();
  } catch (e) { /* toast shown */ }
}

async function del(name) {
  if (!(await confirmDialog({ title: "Delete dataset",
    message: `Delete "${name}"? Registered datasets only remove the link; uploaded ones are deleted from disk.`,
    confirmText: "Delete" }))) return;
  try { await api.del(`/api/datasets/${encodeURIComponent(name)}`); toast("Deleted", "success"); reload(); }
  catch (e) { /* toast shown */ }
}

function openAddModal() {
  let activeTab = "upload";
  const bodyUpload = uploadTab();
  const bodyRegister = registerTab();
  const body = h("div", {},
    h("div", { class: "tabs" },
      h("button", { class: "active", dataset: { tab: "upload" }, onClick: switchTab }, "Upload .zip"),
      h("button", { dataset: { tab: "register" }, onClick: switchTab }, "Register path")),
    bodyUpload, bodyRegister);
  bodyRegister.style.display = "none";

  function switchTab(e) {
    activeTab = e.target.dataset.tab;
    body.querySelectorAll(".tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === activeTab));
    bodyUpload.style.display = activeTab === "upload" ? "" : "none";
    bodyRegister.style.display = activeTab === "register" ? "" : "none";
  }
  modal({ title: "Add dataset", body });
}

function uploadTab() {
  const wrap = h("div", {});
  const fileInput = h("input", { type: "file", accept: ".zip", style: { display: "none" } });
  const nameInput = h("input", { type: "text", placeholder: "Optional name (defaults to archive name)" });
  const progress = h("div", { class: "progressline", style: { display: "none", marginTop: "12px" } },
    h("span", { style: { width: "0%" } }));
  const dz = h("div", { class: "dropzone" },
    h("div", { style: { fontSize: "26px" } }, "⬆"),
    h("div", {}, "Drop a .zip here or click to browse"),
    h("div", { class: "small dim", style: { marginTop: "4px" } }, "Must contain data.yaml"));

  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("over");
    if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) doUpload(fileInput.files[0]); });

  async function doUpload(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) { toast("Please select a .zip file", "error"); return; }
    const fd = new FormData();
    fd.append("file", file);
    if (nameInput.value.trim()) fd.append("name", nameInput.value.trim());
    progress.style.display = "block";
    const bar = progress.firstChild;
    try {
      const info = await api.upload("/api/datasets/upload", fd, (f) => { bar.style.width = `${Math.round(f * 100)}%`; });
      toast(`Added "${info.name}"`, "success");
      document.querySelector(".modal-backdrop")?.remove();
      reload();
    } catch (e) { progress.style.display = "none"; }
  }

  wrap.appendChild(h("div", { class: "field" },
    h("label", {}, "Dataset name"), nameInput));
  wrap.appendChild(dz);
  wrap.appendChild(fileInput);
  wrap.appendChild(progress);
  return wrap;
}

function registerTab() {
  const pathInput = h("input", { type: "text", placeholder: "/absolute/path/to/dataset" });
  const nameInput = h("input", { type: "text", placeholder: "Optional name" });
  const btn = h("button", { class: "btn primary", onClick: doRegister }, "Register");

  async function doRegister() {
    if (!pathInput.value.trim()) { toast("Enter a server path", "error"); return; }
    btn.disabled = true;
    try {
      const info = await api.post("/api/datasets/register",
        { path: pathInput.value.trim(), name: nameInput.value.trim() || null });
      toast(`Registered "${info.name}"`, "success");
      document.querySelector(".modal-backdrop")?.remove();
      reload();
    } catch (e) { btn.disabled = false; }
  }
  return h("div", {},
    h("div", { class: "field" }, h("label", {}, "Server path"), pathInput),
    h("div", { class: "field" }, h("label", {}, "Name"), nameInput),
    h("div", { class: "banner info" },
      "Creates a symlink under datasets/. The dataset must contain a valid data.yaml."),
    h("div", { style: { textAlign: "right" } }, btn));
}
