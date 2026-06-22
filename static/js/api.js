// fetch wrapper with JSON + error handling.
import { toast } from "./ui.js";

async function req(method, path, body, opts = {}) {
  const init = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (!opts.silent) toast(`Network error: ${e.message}`, "error");
    throw e;
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    if (!opts.silent) toast(`${detail}`, "error");
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res;
}

export const api = {
  get: (p, o) => req("GET", p, undefined, o),
  post: (p, b, o) => req("POST", p, b, o),
  put: (p, b, o) => req("PUT", p, b, o),
  del: (p, o) => req("DELETE", p, undefined, o),

  // Upload with progress via XHR (fetch lacks upload progress).
  upload(path, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", path);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText || "{}"));
        } else {
          let d = xhr.statusText;
          try { d = JSON.parse(xhr.responseText).detail || d; } catch (_) {}
          toast(d, "error");
          reject(new Error(d));
        }
      };
      xhr.onerror = () => { toast("Upload failed", "error"); reject(new Error("upload failed")); };
      xhr.send(formData);
    });
  },
};
