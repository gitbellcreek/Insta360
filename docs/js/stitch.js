// The stitching pipeline in the browser.  Mirrors insta360stitch/stitch.py:
// projection on the GPU, seam work (flow, seam, blend) on the CPU in the two
// narrow seam bands, levelling as a final GPU pass.
import { calibrate } from "./calibrate.js";
import { denseFlow, fbConsistent, cleanFlow } from "./flow.js";
import { DEG, lensesFromCalibration, levelRotation, lonToU, mat3Identity, mat3Mul, rotY, upFromAccel } from "./geometry.js";
import { FS_PROJECT, FS_ROTATE, GL } from "./gl.js";
import { gaussianBlur, pyrDown, pyrUp, resizeBilinear, rgbToGray, rgbaAlphaMask, rgbaToRgbF32, sampleBilinear, sobelX } from "./imageops.js";
import { accelFromImu } from "./insp.js";

export const tick = () => new Promise(r => setTimeout(r, 0));

export const defaultOptions = () => ({ width: 4096, parallax: true, level: true, refine: true, bandHalfDeg: 15, blendLevels: 5, yawOffset: 0, defaultFov: 200 });

// ---------------------------------------------------------------- seams
export function seamCost(A, B, w, h, overlap) {
  const a = gaussianBlur(A, w, h, 3, 1.5), b = gaussianBlur(B, w, h, 3, 1.5);
  const ga = rgbToGray(A, w * h), gb = rgbToGray(B, w * h);
  const sa = sobelX(ga, w, h), sb = sobelX(gb, w, h);
  const cost = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const d = (Math.abs(a[i * 3] - b[i * 3]) + Math.abs(a[i * 3 + 1] - b[i * 3 + 1]) + Math.abs(a[i * 3 + 2] - b[i * 3 + 2])) / 3;
    cost[i] = overlap[i] ? d + Math.abs(sa[i] - sb[i]) * 0.125 + 0.5 : 1e4;
  }
  return cost;
}
export function minCostSeam(cost, w, h) {
  const acc = new Float64Array(w * h), back = new Int8Array(w * h);
  for (let x = 0; x < w; x++) acc[x] = cost[x];
  for (let y = 1; y < h; y++) {
    const p = (y - 1) * w, r = y * w;
    for (let x = 0; x < w; x++) {
      let best = acc[p + x], arg = 0;
      if (x > 0 && acc[p + x - 1] < best) { best = acc[p + x - 1]; arg = -1; }
      if (x < w - 1 && acc[p + x + 1] < best) { best = acc[p + x + 1]; arg = 1; }
      acc[r + x] = cost[r + x] + best; back[r + x] = arg;
    }
  }
  const seam = new Int32Array(h);
  let u = 0; for (let x = 1; x < w; x++) if (acc[(h - 1) * w + x] < acc[(h - 1) * w + u]) u = x;
  for (let y = h - 1; y >= 0; y--) { seam[y] = u; u += back[y * w + u]; }
  return seam;
}

// ---------------------------------------------------------------- parallax
export function flowWarp(A, B, va, vb, w, h, seam, taperPx) {
  // fill invalid areas with the other lens so the flow is ~0 there
  const Af = A.slice(), Bf = B.slice();
  for (let i = 0; i < w * h; i++) { if (!va[i]) { Af[i * 3] = B[i * 3]; Af[i * 3 + 1] = B[i * 3 + 1]; Af[i * 3 + 2] = B[i * 3 + 2]; } if (!vb[i]) { Bf[i * 3] = A[i * 3]; Bf[i * 3 + 1] = A[i * 3 + 1]; Bf[i * 3 + 2] = A[i * 3 + 2]; } }
  let ga = rgbToGray(Af, w * h), gb = rgbToGray(Bf, w * h);
  const scale = w > 400 ? 2 : 1;
  let sw = w, sh = h;
  if (scale === 2) { const pa = pyrDown(ga, w, h, 1), pb = pyrDown(gb, w, h, 1); ga = pa.data; gb = pb.data; sw = pa.w; sh = pa.h; }
  let fab = denseFlow(ga, gb, sw, sh), fba = denseFlow(gb, ga, sw, sh);
  const maxFlow = 0.5 * sw;
  for (let i = 0; i < fab.length; i++) { fab[i] = Math.max(-maxFlow, Math.min(maxFlow, fab[i])); fba[i] = Math.max(-maxFlow, Math.min(maxFlow, fba[i])); }
  const okAB = fbConsistent(fab, fba, sw, sh, 3.0 / scale), okBA = fbConsistent(fba, fab, sw, sh, 3.0 / scale);
  fab = cleanFlow(fab, okAB, sw, sh, 6 / scale); fba = cleanFlow(fba, okBA, sw, sh, 6 / scale);
  if (scale === 2) {
    fab = resizeBilinear(fab, sw, sh, 2, w, h); fba = resizeBilinear(fba, sw, sh, 2, w, h);
    for (let i = 0; i < fab.length; i++) { fab[i] *= 2; fba[i] *= 2; }
  }
  // weight: 1 on the seam, 0 beyond taperPx, restricted to (a softened) overlap
  const soft = new Float32Array(w * h);
  const softR = Math.max(1, taperPx * 0.35);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    // horizontal distance to the nearest non-overlap pixel
    let d = 1e9; for (let x = 0; x < w; x++) { d = (va[row + x] & vb[row + x]) ? d + 1 : 0; soft[row + x] = d; }
    d = 1e9; for (let x = w - 1; x >= 0; x--) { d = (va[row + x] & vb[row + x]) ? d + 1 : 0; if (d < soft[row + x]) soft[row + x] = d; }
    for (let x = 0; x < w; x++) soft[row + x] = Math.min(1, (soft[row + x] + softR * 0.5) / softR);
  }
  const wgt = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) wgt[y * w + x] = Math.max(0, 1 - Math.abs(x - seam[y]) / taperPx) * soft[y * w + x];
  const halfWarp = (img, flow) => {
    const out = new Float32Array(w * h * 3), f = new Float32Array(2);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = y * w + x; const ww = 0.5 * wgt[k];
      let px = x, py = y;
      for (let it = 0; it < 3; it++) { sampleBilinear(flow, w, h, 2, px, py, f, 0); px = x - ww * f[0]; py = y - ww * f[1]; }
      sampleBilinear(img, w, h, 3, px, py, out, k * 3);
    }
    return out;
  };
  const A2 = halfWarp(Af, fab), B2 = halfWarp(Bf, fba);
  let mean = 0, max = 0, n = 0;
  for (let i = 0; i < w * h; i++) if (va[i] & vb[i]) { const m = Math.hypot(fab[i * 2], fab[i * 2 + 1]) * wgt[i]; mean += m; if (m > max) max = m; n++; }
  return { A2, B2, flowMean: n ? mean / n : 0, flowMax: max };
}

// ---------------------------------------------------------------- blending
export function multibandBlend(A, B, maskA, w, h, levels) {
  levels = Math.max(1, Math.min(levels, Math.floor(Math.log2(Math.max(4, Math.min(w, h)))) - 2));
  const gA = [{ data: A, w, h }], gB = [{ data: B, w, h }], gM = [{ data: maskA, w, h }];
  for (let i = 0; i < levels; i++) {
    const a = gA[i]; gA.push(pyrDown(a.data, a.w, a.h, 3));
    const b = gB[i]; gB.push(pyrDown(b.data, b.w, b.h, 3));
    const m = gM[i]; gM.push(pyrDown(m.data, m.w, m.h, 1));
  }
  let out = null;
  for (let i = levels; i >= 0; i--) {
    const a = gA[i], b = gB[i], m = gM[i].data;
    let la, lb;
    if (i === levels) { la = a.data; lb = b.data; }
    else {
      const ua = pyrUp(gA[i + 1].data, gA[i + 1].w, gA[i + 1].h, 3, a.w, a.h), ub = pyrUp(gB[i + 1].data, gB[i + 1].w, gB[i + 1].h, 3, b.w, b.h);
      la = new Float32Array(a.data.length); lb = new Float32Array(b.data.length);
      for (let k = 0; k < la.length; k++) { la[k] = a.data[k] - ua[k]; lb[k] = b.data[k] - ub[k]; }
    }
    const layer = new Float32Array(la.length);
    for (let k = 0; k < a.w * a.h; k++) { const mm = m[k]; layer[k * 3] = la[k * 3] * mm + lb[k * 3] * (1 - mm); layer[k * 3 + 1] = la[k * 3 + 1] * mm + lb[k * 3 + 1] * (1 - mm); layer[k * 3 + 2] = la[k * 3 + 2] * mm + lb[k * 3 + 2] * (1 - mm); }
    if (out === null) out = layer;
    else { const up = pyrUp(out, gA[i + 1].w, gA[i + 1].h, 3, a.w, a.h); for (let k = 0; k < layer.length; k++) layer[k] += up[k]; out = layer; }
  }
  return out;
}

// ---------------------------------------------------------------- gains
export function matchGains(A, B, overlap, n) {
  const ratios = [[], [], []];
  for (let i = 0; i < n; i += 7) {
    if (!overlap[i]) continue;
    const a = [A[i * 3], A[i * 3 + 1], A[i * 3 + 2]], b = [B[i * 3], B[i * 3 + 1], B[i * 3 + 2]];
    if (Math.max(...a) >= 240 || Math.max(...b) >= 240 || Math.min(...a) <= 8 || Math.min(...b) <= 8) continue;
    for (let c = 0; c < 3; c++) ratios[c].push(a[c] / b[c]);
  }
  if (ratios[0].length < 200) return { gA: [1, 1, 1], gB: [1, 1, 1] };
  const gA = [], gB = [];
  for (let c = 0; c < 3; c++) { ratios[c].sort((p, q) => p - q); const r = Math.max(0.5, Math.min(2, ratios[c][ratios[c].length >> 1])); gA.push(1 / Math.sqrt(r)); gB.push(Math.sqrt(r)); }
  return { gA, gB };
}

// ---------------------------------------------------------------- stitcher
export class Stitcher {
  constructor(gl) {
    this.gl = gl || new GL();
    this.calibCache = {};
    this.progProject = this.gl.program("project", FS_PROJECT);
    this.progRotate = this.gl.program("rotate", FS_ROTATE);
  }

  initialLenses(insp, imgW, imgH, opts) {
    if (!insp.calibration) throw new Error(`${insp.name}: no lens calibration found in file`);
    const lenses = lensesFromCalibration(insp.calibration, imgW, imgH, opts.defaultFov);
    const cached = this.calibCache[insp.serial || "unknown"];
    if (cached) {
      for (const l of lenses) { l.fov = cached.fov ?? l.fov; l.k1 = cached.k1 ?? l.k1; l.k2 = cached.k2 ?? l.k2; }
      lenses[1].yaw = cached.yaw2 ?? lenses[1].yaw; lenses[1].pitch = cached.pitch2 ?? lenses[1].pitch; lenses[1].roll = cached.roll2 ?? lenses[1].roll;
    }
    return lenses;
  }
  remember(serial, lenses) {
    this.calibCache[serial || "unknown"] = { fov: lenses[0].fov, k1: lenses[0].k1, k2: lenses[0].k2, yaw2: lenses[1].yaw, pitch2: lenses[1].pitch, roll2: lenses[1].roll };
  }

  _setLensUniforms(gl, u, lenses, srcW, srcH, gains) {
    const M = new Float32Array(18), A = new Float32Array(8), B = new Float32Array(4), G = new Float32Array(6);
    lenses.forEach((l, i) => {
      M.set(GL.mat3(l.M()), i * 9);
      A.set([l.cx, l.cy, l.f, l.thetaMax], i * 4);
      B.set([l.k1, l.k2], i * 2);
      G.set(gains ? gains[i] : [1, 1, 1], i * 3);
    });
    gl.uniformMatrix3fv(u.uM, false, M);
    gl.uniform4fv(u.uLensA, A); gl.uniform2fv(u.uLensB, B); gl.uniform3fv(u.uGain, G);
    gl.uniform2f(u.uSrcSize, srcW, srcH);
    gl.uniform1f(u.uFeather, 1.0 * DEG);
  }

  /** Render one lens (mode 1 or 2) or the composite (mode 0) for columns [u0, u1). */
  _render(fbo, srcTex, lenses, W, H, u0, mode, gains) {
    this.gl.draw(this.progProject, fbo, (gl, u) => {
      this._setLensUniforms(gl, u, lenses, srcTex.w, srcTex.h, gains);
      gl.uniform2f(u.uOutSize, W, H); gl.uniform1f(u.uU0, u0); gl.uniform1i(u.uMode, mode);
    }, { uSrc: srcTex });
  }

  _bandRanges(W, bandHalfDeg) {
    return [-90, 90].map(lon => { const u0 = Math.round(lonToU(lon - bandHalfDeg, W)), u1 = Math.round(lonToU(lon + bandHalfDeg, W)); return { lon, u0, u1 }; });
  }

  /** Read both lenses over a band: rgb float + validity masks. */
  _readBand(srcTex, lenses, W, H, u0, u1, gains) {
    const bw = u1 - u0;
    const fbo = this.gl.framebuffer(bw, H);
    const out = {};
    for (const [key, mode] of [["A", 1], ["B", 2]]) {
      this._render(fbo, srcTex, lenses, W, H, u0, mode, gains);
      const rgba = this.gl.readPixels(fbo);
      out[key] = rgbaToRgbF32(rgba, bw * H);
      out["v" + key] = rgbaAlphaMask(rgba, bw * H);
    }
    this.gl.deleteFramebuffer(fbo);
    return { ...out, w: bw, h: H, u0 };
  }

  async stitch(insp, opts, progress = () => {}) {
    const t0 = performance.now();
    const gl = this.gl;
    const info = { file: insp.name, options: { ...opts }, seams: [] };
    progress("decoding");
    let bitmap = await createImageBitmap(new Blob([insp.jpeg], { type: "image/jpeg" }), { imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
    let scale = 1;
    if (bitmap.width > gl.maxTex) {
      scale = gl.maxTex / bitmap.width;
      const c = document.createElement("canvas"); c.width = Math.round(bitmap.width * scale); c.height = Math.round(bitmap.height * scale);
      c.getContext("2d").drawImage(bitmap, 0, 0, c.width, c.height);
      bitmap.close(); bitmap = await createImageBitmap(c, { imageOrientation: "none", premultiplyAlpha: "none" });
      info.sourceDownscaled = scale;
    }
    const srcW = bitmap.width, srcH = bitmap.height;
    const srcTex = gl.texture(srcW, srcH, bitmap);
    bitmap.close();
    try {
      let lenses = this.initialLenses(insp, Math.round(srcW / scale), Math.round(srcH / scale), opts);
      if (scale !== 1) for (const l of lenses) { l.cx *= scale; l.cy *= scale; l.rCal *= scale; }
      await tick();

      if (opts.refine) {
        progress("aligning lenses (control points)");
        const renderBands = async (ls, renderW, halfDeg) => {
          const RH = renderW / 2, bands = [];
          for (const { u0, u1 } of this._bandRanges(renderW, halfDeg)) {
            const b = this._readBand(srcTex, ls, renderW, RH, u0, u1, null);
            bands.push({ A: rgbToGray(b.A, b.w * b.h), B: rgbToGray(b.B, b.w * b.h), va: b.vA, vb: b.vB, w: b.w, h: b.h, u0 });
          }
          return bands;
        };
        const res = await calibrate(renderBands, lenses, { log: progress, tick });
        info.alignment = res.message; info.alignmentPoints = res.nPoints;
        progress(res.message);
        if (res.accepted) { lenses = res.lenses; this.remember(insp.serial, lenses); }
      }
      info.lenses = lenses.map(l => l.toJSON());

      let R = mat3Identity();
      const accel = accelFromImu(insp.imu);
      if (opts.level && accel) { const up = upFromAccel(accel); R = levelRotation(up); info.levelled = true; info.tiltDeg = Math.acos(Math.max(-1, Math.min(1, up[1]))) / DEG; }
      else info.levelled = false;
      if (opts.yawOffset) R = mat3Mul(R, rotY(opts.yawOffset));
      const rotate = !R.every((v, i) => Math.abs(v - (i % 4 === 0 ? 1 : 0)) < 1e-9);

      const W = opts.width | 0, H = W >> 1;
      const bands = this._bandRanges(W, opts.bandHalfDeg);
      progress(`projecting lenses to ${W}x${H}`);
      const raw = bands.map(b => this._readBand(srcTex, lenses, W, H, b.u0, b.u1, null));
      await tick();
      // exposure gains from the band overlaps
      const nAll = raw.reduce((s, b) => s + b.w * b.h, 0);
      const cA = new Float32Array(nAll * 3), cB = new Float32Array(nAll * 3), cO = new Uint8Array(nAll);
      let off = 0;
      for (const b of raw) { cA.set(b.A, off * 3); cB.set(b.B, off * 3); for (let i = 0; i < b.w * b.h; i++) cO[off + i] = b.vA[i] & b.vB[i]; off += b.w * b.h; }
      const { gA, gB } = matchGains(cA, cB, cO, nAll);
      info.gains = { front: gA, back: gB };
      const gains = [gA, gB];

      const composite = gl.framebuffer(W, H, { wrapS: "repeat" });
      this._render(composite, srcTex, lenses, W, H, 0, 0, gains);
      await tick();

      for (let bi = 0; bi < bands.length; bi++) {
        const { lon, u0, u1 } = bands[bi];
        const b = raw[bi]; const w = b.w, h = b.h, n = w * h;
        const A = b.A, B = b.B;
        for (let i = 0; i < n; i++) { A[i * 3] *= gA[0]; A[i * 3 + 1] *= gA[1]; A[i * 3 + 2] *= gA[2]; B[i * 3] *= gB[0]; B[i * 3 + 1] *= gB[1]; B[i * 3 + 2] *= gB[2]; }
        const overlap = new Uint8Array(n);
        for (let y = 0; y < h; y++) { let any = 0; for (let x = 0; x < w; x++) { const v = b.vA[y * w + x] & b.vB[y * w + x]; overlap[y * w + x] = v; any |= v; } if (!any) overlap[y * w + (w >> 1)] = 1; }
        progress(`seam at ${lon > 0 ? "+" : ""}${lon} deg: finding seam`);
        await tick();
        let seam = minCostSeam(seamCost(A, B, w, h, overlap), w, h);
        let A2 = A, B2 = B, entry = { lon, seamMeanLon: 0 };
        if (opts.parallax) {
          progress(`seam at ${lon > 0 ? "+" : ""}${lon} deg: optical flow parallax compensation`);
          await tick();
          const r = flowWarp(A, B, b.vA, b.vB, w, h, seam, 0.7 * w / 2);
          A2 = r.A2; B2 = r.B2; entry.flowPxMean = r.flowMean; entry.flowPxMax = r.flowMax;
          seam = minCostSeam(seamCost(A2, B2, w, h, overlap), w, h);
        }
        let sm = 0; for (let y = 0; y < h; y++) sm += seam[y]; entry.seamMeanLon = ((u0 + sm / h) / W * 2 - 1) * 180;
        const maskA = new Float32Array(n);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) maskA[y * w + x] = (lon < 0 ? x > seam[y] : x < seam[y]) ? 1 : 0;
        progress(`seam at ${lon > 0 ? "+" : ""}${lon} deg: blending`);
        await tick();
        const blended = multibandBlend(A2, B2, maskA, w, h, opts.blendLevels);
        const rgba = new Uint8Array(n * 4);
        for (let i = 0; i < n; i++) { rgba[i * 4] = Math.max(0, Math.min(255, Math.round(blended[i * 3]))); rgba[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(blended[i * 3 + 1]))); rgba[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(blended[i * 3 + 2]))); rgba[i * 4 + 3] = 255; }
        gl.subImage(composite.tex, u0, 0, w, h, rgba);
        info.seams.push(entry);
        raw[bi] = null;
        await tick();
      }

      let outFbo = composite, rotFbo = null;
      if (rotate) {
        progress("levelling the panorama");
        rotFbo = gl.framebuffer(W, H);
        gl.draw(this.progRotate, rotFbo, (g, u) => { g.uniform2f(u.uSize, W, H); g.uniformMatrix3fv(u.uR, false, GL.mat3(R)); }, { uPano: composite.tex });
        outFbo = rotFbo;
      }
      progress("reading result");
      await tick();
      const rgba = gl.readPixels(outFbo);
      gl.deleteFramebuffer(composite); if (rotFbo) gl.deleteFramebuffer(rotFbo);
      info.seconds = Math.round((performance.now() - t0) / 100) / 10;
      progress(`done in ${info.seconds} s`);
      return { rgba, W, H, info };
    } finally {
      gl.deleteTexture(srcTex);
    }
  }
}
