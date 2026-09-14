// Interactive 360 viewer on its own WebGL2 canvas.
import { mat3Mul, rotX, rotY } from "./geometry.js";
import { FS_VIEW, GL } from "./gl.js";

export class PanoViewer {
  constructor(canvas, hud) {
    this.canvas = canvas; this.hud = hud;
    this.gl = new GL(canvas);
    this.prog = this.gl.program("view", FS_VIEW);
    this.tex = null; this.texW = 0; this.texH = 0;
    this.mode = "pano"; this.hasPano = false;
    this.yaw = 0; this.pitch = 0; this.fov = 90;
    this.message = "Open a folder with .insp files to begin";
    this._drag = null; this._pinch = null; this._raf = 0;
    canvas.tabIndex = 0;
    canvas.addEventListener("pointerdown", e => { canvas.setPointerCapture(e.pointerId); canvas.focus(); this._onDown(e); });
    canvas.addEventListener("pointermove", e => this._onMove(e));
    canvas.addEventListener("pointerup", e => this._onUp(e));
    canvas.addEventListener("pointercancel", e => this._onUp(e));
    canvas.addEventListener("wheel", e => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1 : -1); }, { passive: false });
    canvas.addEventListener("dblclick", () => this.resetView());
    canvas.addEventListener("keydown", e => {
      const k = e.key.toLowerCase();
      if (k === "arrowleft") this.nudge(-5, 0); else if (k === "arrowright") this.nudge(5, 0);
      else if (k === "arrowup") this.nudge(0, 5); else if (k === "arrowdown") this.nudge(0, -5);
      else if (k === "f") this.toggleMode(); else return;
      e.preventDefault();
    });
    new ResizeObserver(() => this.requestRender()).observe(canvas);
    this._pointers = new Map();
  }

  // ---- content
  async setPano(bitmap) { this._upload(bitmap); this.hasPano = true; this.mode = "pano"; this.message = ""; this.requestRender(); }
  async setFlat(bitmap, message = "") { this._upload(bitmap); this.hasPano = false; this.mode = "flat"; this.message = message; this.requestRender(); }
  clear(message = "") { if (this.tex) { this.gl.deleteTexture(this.tex); this.tex = null; } this.hasPano = false; this.message = message; this.requestRender(); }
  _upload(bitmap) {
    if (this.tex) this.gl.deleteTexture(this.tex);
    const gl = this.gl.gl;
    this.tex = this.gl.texture(bitmap.width, bitmap.height, bitmap, { wrapS: "repeat" });
    gl.bindTexture(gl.TEXTURE_2D, this.tex.t);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    this.texW = bitmap.width; this.texH = bitmap.height;
  }
  toggleMode() { if (!this.hasPano) return; this.mode = this.mode === "pano" ? "flat" : "pano"; this.requestRender(); }
  resetView() { this.yaw = 0; this.pitch = 0; this.fov = 90; this.requestRender(); }
  nudge(dy, dp) { this.yaw = ((this.yaw + dy + 180) % 360 + 360) % 360 - 180; this.pitch = Math.max(-90, Math.min(90, this.pitch + dp)); this.requestRender(); }
  zoom(dir) { if (this.mode !== "pano") return; this.fov = Math.max(20, Math.min(140, this.fov * Math.pow(1.1, dir))); this.requestRender(); }

  // ---- interaction
  _onDown(e) {
    this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this._pointers.size === 1) this._drag = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch };
    else if (this._pointers.size === 2) { const p = [...this._pointers.values()]; this._pinch = { d: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), fov: this.fov }; this._drag = null; }
  }
  _onMove(e) {
    if (!this._pointers.has(e.pointerId)) return;
    this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this._pinch && this._pointers.size === 2) {
      const p = [...this._pointers.values()]; const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
      this.fov = Math.max(20, Math.min(140, this._pinch.fov * this._pinch.d / Math.max(1, d))); this.requestRender(); return;
    }
    if (!this._drag || this.mode !== "pano" || !this.hasPano) return;
    const degPerPx = this.fov / Math.max(1, this.canvas.clientWidth);
    this.yaw = ((this._drag.yaw - (e.clientX - this._drag.x) * degPerPx + 180) % 360 + 360) % 360 - 180;
    this.pitch = Math.max(-90, Math.min(90, this._drag.pitch + (e.clientY - this._drag.y) * degPerPx));
    this.requestRender();
  }
  _onUp(e) { this._pointers.delete(e.pointerId); if (this._pointers.size < 2) this._pinch = null; if (this._pointers.size === 0) this._drag = null; }

  /**
   * Render the current viewpoint as a W x H photo (RGBA, top-down rows).
   * fovH is the horizontal field of view in degrees; defaults to the on-screen one.
   */
  renderView(W, H, fovH = this.fov) {
    if (!this.tex || !this.hasPano) return null;
    W = Math.min(W, this.gl.maxTex); H = Math.min(H, this.gl.maxTex);
    const fbo = this.gl.framebuffer(W, H);
    const R = mat3Mul(rotY(this.yaw), rotX(-this.pitch));
    const focal = (W / 2) / Math.tan(fovH * Math.PI / 360);
    this.gl.draw(this.prog, fbo, (g, u) => {
      g.uniform2f(u.uView, W, H); g.uniform1f(u.uFocal, focal); g.uniformMatrix3fv(u.uR, false, GL.mat3(R));
      g.uniform1i(u.uFlat, 0); g.uniform4f(u.uFlatRect, 0, 0, 1, 1); g.uniform1f(u.uFlipY, -1.0);
    }, { uPano: this.tex });
    const rgba = this.gl.readPixels(fbo);
    this.gl.deleteFramebuffer(fbo);
    this.requestRender();
    return { rgba, W, H };
  }

  /**
   * The on-screen rectangle (CSS px) that a W x H export covers, and the
   * horizontal FOV to render it with.  Wider outputs keep the full width
   * (letterbox); taller outputs keep the full height (pillarbox).
   */
  exportFrame(W, H) {
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    const ao = W / H, ac = cw / ch;
    if (ao >= ac) { const fh = cw / ao; return { x: 0, y: (ch - fh) / 2, w: cw, h: fh, fovH: this.fov }; }
    const fw = ch * ao;
    const fovH = 2 * Math.atan(Math.tan(this.fov * Math.PI / 360) * (fw / cw)) * 180 / Math.PI;
    return { x: (cw - fw) / 2, y: 0, w: fw, h: ch, fovH };
  }

  // ---- rendering
  requestRender() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); }); }
  render() {
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(c.clientWidth * dpr)), h = Math.max(2, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const gl = this.gl.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h);
    gl.clearColor(0.09, 0.09, 0.1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.tex) {
      const flat = this.mode === "flat" || !this.hasPano;
      const s = Math.min(w / this.texW, h / this.texH);
      const fw = this.texW * s, fh = this.texH * s;
      const R = mat3Mul(rotY(this.yaw), rotX(-this.pitch));
      const focal = (w / 2) / Math.tan(this.fov * Math.PI / 360);
      this.gl.draw(this.prog, null, (g, u) => {
        g.uniform2f(u.uView, w, h); g.uniform1f(u.uFocal, focal); g.uniformMatrix3fv(u.uR, false, GL.mat3(R));
        g.uniform1i(u.uFlat, flat ? 1 : 0); g.uniform4f(u.uFlatRect, (w - fw) / 2, (h - fh) / 2, fw, fh); g.uniform1f(u.uFlipY, 1.0);
      }, { uPano: this.tex });
    }
    if (this.hud) {
      const parts = [];
      if (this.message) parts.push(this.message);
      if (this.hasPano && this.mode === "pano") parts.push(`yaw ${this.yaw.toFixed(0)}  pitch ${this.pitch.toFixed(0)}  fov ${this.fov.toFixed(0)}   drag: look · wheel/pinch: zoom · S: save view · F: flat view · double-click: reset`);
      else if (this.hasPano) parts.push("flat view · press F for the interactive view");
      this.hud.textContent = parts.join("   ");
    }
  }
}
