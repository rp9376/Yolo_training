// Zero-dependency Canvas 2D charts: lineChart + ringGauge. DPR-aware.

const FONT = "11px ui-sans-serif, -apple-system, Segoe UI, sans-serif";

function setup(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 300;
  const cssH = cssHeight || canvas.clientHeight || 200;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

/**
 * lineChart(canvas, series, opts)
 *  series: [{ label, color, data:[numbers] }]   (null entries are gaps)
 *  opts:   { x:[numbers]?, height?, yMin?, yMax?, xLabel?, rolling? }
 */
export function lineChart(canvas, series, opts = {}) {
  const { ctx, w, h } = setup(canvas, opts.height || 220);
  ctx.clearRect(0, 0, w, h);

  const padL = 44, padR = 14, padT = 12, padB = 24;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const allY = [];
  series.forEach((s) => s.data.forEach((v) => { if (v != null && !isNaN(v)) allY.push(v); }));
  if (allY.length === 0) {
    ctx.fillStyle = "#5c6680"; ctx.font = FONT; ctx.textAlign = "center";
    ctx.fillText("no data yet", w / 2, h / 2);
    return;
  }
  let yMin = opts.yMin != null ? opts.yMin : Math.min(...allY);
  let yMax = opts.yMax != null ? opts.yMax : Math.max(...allY);
  if (yMin === yMax) { yMax = yMin + 1; yMin -= 1; }
  const range = yMax - yMin;
  yMin -= range * 0.08; yMax += range * 0.08;

  const n = Math.max(...series.map((s) => s.data.length));
  const xs = opts.x || Array.from({ length: n }, (_, i) => i + 1);
  const xMin = xs[0] ?? 0, xMax = xs[xs.length - 1] ?? 1;
  const xSpan = (xMax - xMin) || 1;

  const px = (xv) => padL + ((xv - xMin) / xSpan) * plotW;
  const py = (yv) => padT + plotH - ((yv - yMin) / (yMax - yMin)) * plotH;

  // grid + y labels
  ctx.strokeStyle = "#2a3346"; ctx.fillStyle = "#5c6680";
  ctx.font = FONT; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + ((yMax - yMin) * i) / 4;
    const y = py(yv);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.globalAlpha = 0.4;
    ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillText(fmtTick(yv), padL - 6, y);
  }
  // x labels (a few)
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const ticks = Math.min(6, xs.length);
  for (let i = 0; i < ticks; i++) {
    const idx = Math.round((xs.length - 1) * (i / Math.max(1, ticks - 1)));
    const xv = xs[idx];
    ctx.fillText(String(xv), px(xv), h - padB + 6);
  }

  // lines
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (v == null || isNaN(v)) { started = false; return; }
      const X = px(xs[i] ?? i + 1), Y = py(v);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    });
    ctx.stroke();
  });
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a !== 0 && a < 1) return v.toFixed(2);
  if (a < 100) return v.toFixed(a < 10 ? 1 : 0);
  return Math.round(v).toString();
}

/** ringGauge(canvas, pct 0..100, label, opts{color}) */
export function ringGauge(canvas, pct, label, opts = {}) {
  const size = opts.size || 96;
  canvas.style.width = size + "px";
  const { ctx } = setup(canvas, size);
  canvas.style.width = size + "px";
  const w = size, h = size;
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, r = size / 2 - 8;
  const start = -Math.PI / 2;
  const frac = Math.max(0, Math.min(1, (pct || 0) / 100));

  ctx.lineWidth = 8; ctx.lineCap = "round";
  ctx.strokeStyle = "#212a3d";
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, opts.color || "#00e5c0");
  grad.addColorStop(1, "#7c5cff");
  ctx.strokeStyle = grad;
  ctx.beginPath(); ctx.arc(cx, cy, r, start, start + frac * Math.PI * 2); ctx.stroke();

  ctx.fillStyle = "#e6e9f2"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = "600 18px ui-sans-serif, sans-serif";
  ctx.fillText(`${Math.round(pct || 0)}%`, cx, cy - 4);
  if (label) {
    ctx.fillStyle = "#8d97ad"; ctx.font = "10px ui-sans-serif, sans-serif";
    ctx.fillText(label, cx, cy + 14);
  }
}
