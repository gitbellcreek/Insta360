// The app: folder picking, file list, stitching queue, viewer, downloads.
import { GL } from "./gl.js";
import { accelFromImu, exifSegment, parseInsp, siblingPrefix } from "./insp.js";
import { encodePanoJpeg } from "./jpeg.js";
import { Stitcher, defaultOptions } from "./stitch.js";
import { PanoViewer } from "./viewer.js";

const $ = id => document.getElementById(id);
const CACHE_KEY = "insta360stitch.calibration";

class App {
  constructor() {
    this.entries = [];         // { name, file?, handle?, insp?, result?, blob?, info?, existing? }
    this.selected = -1;
    this.dirHandle = null;
    this.busy = false; this.cancel = false; this.queue = [];
    try {
      this.viewer = new PanoViewer($("view"), $("hud"));
      this.stitcher = new Stitcher(new GL());
    } catch (e) {
      $("unsupportedWhy").textContent = e.message; $("unsupported").hidden = false; return;
    }
    try { this.stitcher.calibCache = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}"); } catch (e) { /* ignore */ }
    this._buildWidths();
    this._wire();
    this.setStatus("Ready. Open the folder that holds your .insp files (the camera's DCIM/Camera01 folder works directly).");
  }

  _buildWidths() {
    const sel = $("width");
    const max = this.stitcher.gl.maxTex;
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    const widths = [2048, 3040, 4096, 6080, 8192].filter(w => w <= max && (!isSafari || w * w / 2 <= 16777216));
    for (const w of widths) { const o = document.createElement("option"); o.value = w; o.textContent = w === 6080 ? "6080 (native)" : String(w); sel.appendChild(o); }
    sel.value = widths.includes(4096) ? "4096" : String(widths[widths.length - 1]);
  }

  _wire() {
    $("btnOpen").onclick = () => this.openFolder();
    $("btnFiles").onclick = () => $("filesInput").click();
    $("folderInput").onchange = e => this.loadFileList([...e.target.files], "folder");
    $("filesInput").onchange = e => this.loadFileList([...e.target.files], "files");
    $("btnStitch").onclick = () => { if (this.selected >= 0) this.enqueue([this.selected]); };
    $("btnStitchAll").onclick = () => { const todo = this.entries.map((e, i) => (!e.blob && !e.existing) ? i : -1).filter(i => i >= 0); this.enqueue(todo.length ? todo : this.entries.map((_, i) => i)); };
    $("btnCancel").onclick = () => { this.cancel = true; this.queue = []; this.setStatus("Cancelling after the current file…"); };
    $("btnDownload").onclick = () => this.download();
    ["optParallax", "optLevel", "optRefine", "width"].forEach(id => { $(id).onchange = () => {}; });
  }

  options() {
    return { ...defaultOptions(), width: parseInt($("width").value, 10), parallax: $("optParallax").checked, level: $("optLevel").checked, refine: $("optRefine").checked };
  }
  setStatus(t) { $("status").textContent = t; }

  // ---------------------------------------------------------- folder handling
  async openFolder() {
    if (window.showDirectoryPicker) {
      try {
        const dir = await window.showDirectoryPicker({ mode: "readwrite" });
        this.dirHandle = dir;
        const files = [];
        for await (const [name, h] of dir.entries()) if (h.kind === "file" && name.toLowerCase().endsWith(".insp")) files.push({ name, handle: h });
        let stitched = null;
        try { stitched = await dir.getDirectoryHandle("stitched"); } catch (e) { /* none yet */ }
        const existing = new Map();
        if (stitched) for await (const [name, h] of stitched.entries()) if (h.kind === "file" && name.toLowerCase().endsWith(".jpg")) existing.set(name.replace(/\.jpg$/i, ""), h);
        files.sort((a, b) => a.name.localeCompare(b.name));
        this.entries = files.map(f => ({ name: f.name, handle: f.handle, existing: existing.get(f.name.replace(/\.insp$/i, "")) || null }));
        $("folderName").textContent = dir.name;
        this.afterLoad();
        return;
      } catch (e) {
        if (e.name === "AbortError") return;
        console.warn("directory picker failed, falling back", e);
      }
    }
    $("folderInput").click();
  }

  loadFileList(files, kind) {
    this.dirHandle = null;
    const insp = files.filter(f => f.name.toLowerCase().endsWith(".insp"));
    const stitched = new Map();
    for (const f of files) {
      const rel = f.webkitRelativePath || f.name;
      if (/(^|\/)stitched\/[^/]+\.jpg$/i.test(rel)) stitched.set(f.name.replace(/\.jpg$/i, ""), f);
    }
    insp.sort((a, b) => a.name.localeCompare(b.name));
    this.entries = insp.map(f => ({ name: f.name, file: f, existing: stitched.get(f.name.replace(/\.insp$/i, "")) || null }));
    const first = files[0];
    $("folderName").textContent = kind === "folder" && first && first.webkitRelativePath ? first.webkitRelativePath.split("/")[0] : `${insp.length} file(s)`;
    this.afterLoad();
  }

  afterLoad() {
    this.renderList();
    $("btnStitchAll").disabled = !this.entries.length;
    if (this.entries.length) { this.select(0); this.setStatus(`${this.entries.length} .insp file(s) found`); }
    else { this.viewer.clear("No .insp files in this folder"); this.setStatus("No .insp files found"); }
  }

  renderList() {
    const ul = $("fileList"); ul.innerHTML = "";
    this.entries.forEach((e, i) => {
      const li = document.createElement("li");
      const done = !!(e.blob || e.existing);
      li.innerHTML = `<span class="mark">${e.busy ? "…" : done ? "✓" : ""}</span>${e.name}`;
      if (i === this.selected) li.classList.add("selected");
      if (e.busy) li.classList.add("busy");
      li.onclick = () => this.select(i);
      li.ondblclick = () => this.enqueue([i]);
      ul.appendChild(li);
    });
  }

  async readEntry(e) {
    if (e.insp) return e.insp;
    const file = e.file || await e.handle.getFile();
    e.insp = parseInsp(await file.arrayBuffer(), e.name);
    return e.insp;
  }

  // ---------------------------------------------------------- viewing
  async select(i) {
    this.selected = i; this.renderList();
    const e = this.entries[i];
    $("btnStitch").disabled = false;
    $("btnDownload").disabled = !e.blob;
    try {
      if (e.blob) { await this.showPano(e.blob); this.showInfo(e.info); this.setStatus(`Viewing stitched ${e.name}`); return; }
      if (e.existing) {
        const f = e.existing.getFile ? await e.existing.getFile() : e.existing;
        await this.showPano(f); this.showInfo(e.info || `${e.name}\n(previously stitched output found in stitched/)`); this.setStatus(`Viewing stitched/${f.name}`); return;
      }
      const insp = await this.readEntry(e);
      const src = insp.thumbnail ? new Blob([insp.thumbnail], { type: "image/jpeg" }) : new Blob([insp.jpeg], { type: "image/jpeg" });
      const bmp = await createImageBitmap(src);
      await this.viewer.setFlat(bmp, "Not stitched yet - double-click the file or press 'Stitch selected'");
      bmp.close();
      const acc = accelFromImu(insp.imu);
      this.showInfo(`${e.name}\n${insp.model} ${insp.serial}\nfirmware ${insp.firmware}\ncalibration: ${insp.calibration ? "yes" : "missing"}\nIMU samples: ${insp.imu ? insp.imu.n : 0}${acc ? `  gravity (${acc.map(v => v.toFixed(2)).join(", ")})` : ""}`);
    } catch (err) {
      console.error(err); this.viewer.clear(`Cannot read ${e.name}: ${err.message}`);
    }
  }

  async showPano(blob) {
    let bmp = await createImageBitmap(blob);
    const max = this.viewer.gl.maxTex;
    if (bmp.width > max) {
      const c = document.createElement("canvas"); c.width = max; c.height = max / 2;
      c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height); bmp.close(); bmp = await createImageBitmap(c);
    }
    await this.viewer.setPano(bmp); bmp.close();
  }

  showInfo(info) {
    if (!info) { $("info").textContent = ""; return; }
    if (typeof info === "string") { $("info").textContent = info; return; }
    const lines = [info.file];
    if (info.alignment) lines.push(info.alignment);
    lines.push(info.levelled ? `levelled from IMU (tilt ${info.tiltDeg.toFixed(1)} deg)` : "not levelled (no IMU data or disabled)");
    for (const s of info.seams) if (s.flowPxMean !== undefined) lines.push(`seam ${s.lon > 0 ? "+" : ""}${s.lon}: parallax flow mean ${s.flowPxMean.toFixed(1)}px max ${s.flowPxMax.toFixed(0)}px`);
    lines.push(`stitched in ${info.seconds} s, width ${info.options.width}`);
    $("info").textContent = lines.join("\n");
  }

  download() {
    const e = this.entries[this.selected];
    if (!e || !e.blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(e.blob); a.download = e.name.replace(/\.insp$/i, "") + ".jpg";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  // ---------------------------------------------------------- stitching
  enqueue(indices) {
    for (const i of indices) if (!this.queue.includes(i)) this.queue.push(i);
    if (!this.busy) this.runQueue();
    else this.setStatus(`Queued ${indices.length} file(s)`);
  }

  async runQueue() {
    this.busy = true; this.cancel = false;
    $("btnStitch").disabled = $("btnStitchAll").disabled = true; $("btnCancel").disabled = false; $("progress").hidden = false;
    const opts = this.options();
    while (this.queue.length && !this.cancel) {
      const i = this.queue.shift(); const e = this.entries[i];
      e.busy = true; this.renderList();
      try {
        const insp = await this.readEntry(e);
        if (opts.level && !insp.imu) await this.borrowImu(e);
        const progress = s => this.setStatus(`${e.name}: ${s}`);
        const { rgba, W, H, info } = await this.stitcher.stitch(insp, opts, progress);
        progress("encoding JPEG");
        await new Promise(r => setTimeout(r, 0));
        e.blob = await encodePanoJpeg(rgba, W, H, 0.95, exifSegment(insp.jpeg));
        e.info = info;
        try { localStorage.setItem(CACHE_KEY, JSON.stringify(this.stitcher.calibCache)); } catch (err) { /* ignore */ }
        let where = "kept in memory (use Download)";
        if (this.dirHandle) { try { await this.saveToFolder(e); where = "saved to stitched/"; } catch (err) { console.warn(err); where = `could not write to folder (${err.message}); use Download`; } }
        this.setStatus(`${e.name}: done in ${info.seconds} s, ${where}`);
        if (i === this.selected) { $("btnDownload").disabled = false; await this.showPano(e.blob); this.showInfo(info); }
      } catch (err) {
        console.error(err); this.setStatus(`${e.name}: failed: ${err.message}`);
      }
      e.busy = false; this.renderList();
    }
    this.busy = false;
    $("btnStitch").disabled = this.selected < 0; $("btnStitchAll").disabled = !this.entries.length; $("btnCancel").disabled = true; $("progress").hidden = true;
  }

  async borrowImu(e) {
    const prefix = siblingPrefix(e.name);
    if (!prefix) return;
    for (const o of this.entries) {
      if (o === e || !o.name.startsWith(prefix)) continue;
      try { const s = await this.readEntry(o); if (s.imu) { e.insp.imu = s.imu; this.setStatus(`${e.name}: using IMU data from ${o.name}`); return; } } catch (err) { /* skip */ }
    }
  }

  async saveToFolder(e) {
    const dir = await this.dirHandle.getDirectoryHandle("stitched", { create: true });
    const base = e.name.replace(/\.insp$/i, "");
    const fh = await dir.getFileHandle(base + ".jpg", { create: true });
    const w = await fh.createWritable(); await w.write(e.blob); await w.close();
    const jh = await dir.getFileHandle(base + ".json", { create: true });
    const jw = await jh.createWritable(); await jw.write(JSON.stringify(e.info, null, 2)); await jw.close();
  }
}

window.app = new App();
