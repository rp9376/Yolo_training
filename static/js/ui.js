// Tiny DOM helpers — no framework.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === "style" && typeof v === "object") {
      Object.assign(el.style, v);
    } else if (v === true) {
      el.setAttribute(k, "");
    } else {
      el.setAttribute(k, v);
    }
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.appendChild(typeof c === "string" || typeof c === "number"
      ? document.createTextNode(String(c)) : c);
  }
  return el;
}

export function escape(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

export function toast(message, type = "info", timeout = 4000) {
  const root = document.getElementById("toasts");
  const t = h("div", { class: `toast ${type}` }, message);
  root.appendChild(t);
  setTimeout(() => {
    t.style.transition = "opacity .3s";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 300);
  }, timeout);
}

export function modal({ title, body, footer, onClose }) {
  const root = document.getElementById("modal-root");
  const close = () => { backdrop.remove(); onClose && onClose(); };
  const backdrop = h("div", { class: "modal-backdrop",
    onClick: (e) => { if (e.target === backdrop) close(); } },
    h("div", { class: "modal" },
      h("div", { class: "modal-head" },
        h("h2", {}, title || ""),
        h("button", { class: "x-btn", onClick: close }, "×")),
      h("div", { class: "modal-body" }, body),
      footer ? h("div", { class: "modal-foot" }, footer) : null));
  root.appendChild(backdrop);
  return { close, el: backdrop };
}

export function confirmDialog({ title = "Confirm", message, confirmText = "Confirm",
                                danger = true } = {}) {
  return new Promise((resolve) => {
    let m;
    // Guard against double-resolve: m.close() triggers modal's onClose, which
    // also resolves. Without this latch, clicking Confirm would resolve(true)
    // *after* onClose already resolved(false) — so every confirm read as false
    // and the confirmed action silently never ran.
    let settled = false;
    const finish = (val) => { if (settled) return; settled = true; resolve(val); };
    const cancel = h("button", { class: "btn ghost",
      onClick: () => { finish(false); m.close(); } }, "Cancel");
    const ok = h("button", { class: `btn ${danger ? "danger" : "primary"}`,
      onClick: () => { finish(true); m.close(); } }, confirmText);
    m = modal({ title, body: h("p", { class: "muted" }, message),
      footer: [cancel, ok], onClose: () => finish(false) });
  });
}

export function pill(status) {
  return h("span", { class: `pill ${status || "pending"}` }, status || "pending");
}

export function fmtBytes(b) {
  if (!b && b !== 0) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0; let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

export function fmtPct(n) { return n == null ? "—" : `${Number(n).toFixed(0)}%`; }

export function fmtNum(n, d = 3) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

export function fmtDate(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

export function spinner(text = "Loading…") {
  return h("div", { class: "loading" }, h("span", { class: "spinner" }), text);
}

export function emptyState(icon, title, sub) {
  return h("div", { class: "empty" },
    h("div", { class: "ico" }, icon),
    h("div", { style: { fontWeight: 600, color: "var(--text)" } }, title),
    sub ? h("div", { class: "small", style: { marginTop: "6px" } }, sub) : null);
}
