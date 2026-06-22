// Bootstrap + hash router.
import { api } from "./api.js";
import { clear, h } from "./ui.js";
import * as dashboard from "./pages/dashboard.js";
import * as datasets from "./pages/datasets.js";
import * as queue from "./pages/queue.js";
import * as monitor from "./pages/monitor.js";
import * as models from "./pages/models.js";
import * as modelDetail from "./pages/modelDetail.js";

const ICONS = {
  dashboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>',
  datasets: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
  queue: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3.5" cy="6" r="1.5"/><circle cx="3.5" cy="12" r="1.5"/><circle cx="3.5" cy="18" r="1.5"/></svg>',
  monitor: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h4l3 8 4-16 3 8h4"/></svg>',
  models: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l8 4.5v9L12 20l-8-4.5v-9z"/><path d="M12 11l8-4.5M12 11v9M12 11L4 6.5"/></svg>',
};

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: ICONS.dashboard },
  { id: "datasets", label: "Datasets", icon: ICONS.datasets },
  { id: "queue", label: "Queue", icon: ICONS.queue },
  { id: "monitor", label: "Monitor", icon: ICONS.monitor },
  { id: "models", label: "Models", icon: ICONS.models },
];

const PAGES = { dashboard, datasets, queue, monitor, models };
const TITLES = { dashboard: "Dashboard", datasets: "Datasets", queue: "Training Queue",
  monitor: "Monitor", models: "Models" };

let current = null;

function buildNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  for (const item of NAV) {
    nav.appendChild(h("a", { href: `#/${item.id}`, dataset: { route: item.id } },
      h("span", { html: item.icon, style: { display: "inline-flex" } }),
      h("span", {}, item.label)));
  }
}

function setActiveNav(route) {
  document.querySelectorAll("#nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
}

async function route() {
  const hash = location.hash.replace(/^#\/?/, "") || "dashboard";
  const parts = hash.split("/");
  const root = document.getElementById("page");
  const titleEl = document.getElementById("page-title");

  if (current && current.unmount) {
    try { current.unmount(); } catch (e) { console.error(e); }
  }
  current = null;
  clear(root);

  let page, params = {}, navId, title;
  if (parts[0] === "models" && parts[1]) {
    page = modelDetail; params = { runName: decodeURIComponent(parts[1]) };
    navId = "models"; title = "Model Detail";
  } else {
    navId = PAGES[parts[0]] ? parts[0] : "dashboard";
    page = PAGES[navId]; title = TITLES[navId];
  }
  setActiveNav(navId);
  titleEl.textContent = title;
  current = page;
  try {
    await page.render(root, params);
  } catch (e) {
    console.error(e);
    clear(root);
    root.appendChild(h("div", { class: "banner danger" },
      `Failed to render page: ${e.message}`));
  }
}

// --- global status poller (topbar + health) ---
async function pollHealth() {
  const dot = document.getElementById("health-dot");
  const txt = document.getElementById("health-text");
  try {
    const hdata = await api.get("/api/health", { silent: true });
    dot.className = "health-dot ok";
    const bits = [`Py ${hdata.python}`];
    if (hdata.gpu_count) bits.push(`${hdata.gpu_count} GPU`);
    if (hdata.cuda) bits.push("CUDA"); else if (hdata.torch) bits.push("CPU");
    txt.textContent = bits.join(" · ");
  } catch (e) {
    dot.className = "health-dot bad";
    txt.textContent = "backend offline";
  }
}

async function pollTopbar() {
  const el = document.getElementById("topbar-status");
  try {
    const [st, hw] = await Promise.all([
      api.get("/api/queue/status", { silent: true }),
      api.get("/api/hardware", { silent: true }),
    ]);
    clear(el);
    if (st.running) {
      el.appendChild(h("a", { href: "#/monitor", class: "running-indicator" },
        h("span", { class: "dot" }), "Training running"));
    }
    (hw.gpus || []).slice(0, 4).forEach((g) => {
      const memPct = g.mem_total ? Math.round((g.mem_used / g.mem_total) * 100) : 0;
      el.appendChild(h("span", { class: "chip", title: g.name },
        `GPU${g.index} `, h("b", {}, `${g.util}%`), ` · ${memPct}%`));
    });
  } catch (e) { /* offline; health dot covers it */ }
}

function startPollers() {
  pollHealth(); pollTopbar();
  setInterval(pollHealth, 5000);
  setInterval(pollTopbar, 2500);
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  buildNav();
  if (!location.hash) location.hash = "#/dashboard";
  route();
  startPollers();
});
