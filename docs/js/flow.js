// Dense optical flow by coarse-to-fine block matching with zero-mean SAD,
// sub-pixel refinement and a forward/backward consistency check.  Good
// enough for the narrow, mostly-horizontal parallax in a seam band.
import { boxMean, gaussianBlur, median3, pyrDown, resizeBilinear, sampleBilinear } from "./imageops.js";

function matchLevel(a, b, w, h, init, radius, pr, stride) {
  const ma = boxMean(a, w, h, pr), mb = boxMean(b, w, h, pr);
  const gw = Math.ceil(w / stride), gh = Math.ceil(h / stride);
  const grid = new Float32Array(gw * gh * 2);
  const nOff = (2 * radius + 1);
  const costs = new Float32Array(nOff * nOff);
  for (let gy = 0; gy < gh; gy++) {
    const y = Math.min(h - 1, gy * stride);
    for (let gx = 0; gx < gw; gx++) {
      const x = Math.min(w - 1, gx * stride);
      const ii = (y * w + x) * 2;
      const ix = Math.round(init[ii]), iy = Math.round(init[ii + 1]);
      let best = Infinity, bx = 0, by = 0;
      for (let oy = -radius; oy <= radius; oy++) {
        for (let ox = -radius; ox <= radius; ox++) {
          const qx = x + ix + ox, qy = y + iy + oy;
          let cost = 0;
          if (qx < 0 || qy < 0 || qx >= w || qy >= h) cost = 1e9;
          else {
            const meanA = ma[y * w + x], meanB = mb[qy * w + qx];
            for (let dy = -pr; dy <= pr; dy++) {
              let ya = y + dy, yb = qy + dy;
              if (ya < 0) ya = 0; else if (ya >= h) ya = h - 1;
              if (yb < 0) yb = 0; else if (yb >= h) yb = h - 1;
              const ra = ya * w, rb = yb * w;
              for (let dx = -pr; dx <= pr; dx++) {
                let xa = x + dx, xb = qx + dx;
                if (xa < 0) xa = 0; else if (xa >= w) xa = w - 1;
                if (xb < 0) xb = 0; else if (xb >= w) xb = w - 1;
                cost += Math.abs((a[ra + xa] - meanA) - (b[rb + xb] - meanB));
              }
            }
          }
          costs[(oy + radius) * nOff + ox + radius] = cost;
          if (cost < best) { best = cost; bx = ox; by = oy; }
        }
      }
      // sub-pixel parabola fit
      let sx = 0, sy = 0;
      if (bx > -radius && bx < radius) {
        const c0 = costs[(by + radius) * nOff + bx - 1 + radius], c2 = costs[(by + radius) * nOff + bx + 1 + radius];
        const den = c0 - 2 * best + c2; if (den > 1e-6) sx = Math.max(-0.5, Math.min(0.5, 0.5 * (c0 - c2) / den));
      }
      if (by > -radius && by < radius) {
        const c0 = costs[(by - 1 + radius) * nOff + bx + radius], c2 = costs[(by + 1 + radius) * nOff + bx + radius];
        const den = c0 - 2 * best + c2; if (den > 1e-6) sy = Math.max(-0.5, Math.min(0.5, 0.5 * (c0 - c2) / den));
      }
      const gi = (gy * gw + gx) * 2;
      grid[gi] = ix + bx + sx; grid[gi + 1] = iy + by + sy;
    }
  }
  let flow = grid;
  if (stride > 1) flow = resizeBilinear(grid, gw, gh, 2, w, h);
  flow = median3(flow, w, h, 2, 0); flow = median3(flow, w, h, 2, 1);
  return flow;
}

/** Flow from a to b: b(p + flow(p)) ~ a(p).  a, b: gray Float32Array (w*h). */
export function denseFlow(a, b, w, h, opts = {}) {
  const minSize = opts.minSize || 24;
  const pyrA = [{ data: a, w, h }], pyrB = [{ data: b, w, h }];
  while (pyrA.length < 6) {
    const la = pyrA[pyrA.length - 1];
    if (Math.min(la.w, la.h) / 2 < minSize) break;
    pyrA.push(pyrDown(la.data, la.w, la.h, 1));
    const lb = pyrB[pyrB.length - 1];
    pyrB.push(pyrDown(lb.data, lb.w, lb.h, 1));
  }
  let flow = null;
  for (let L = pyrA.length - 1; L >= 0; L--) {
    const la = pyrA[L], lb = pyrB[L];
    let init;
    if (flow === null) init = new Float32Array(la.w * la.h * 2);
    else {
      const prev = pyrA[L + 1];
      init = resizeBilinear(flow, prev.w, prev.h, 2, la.w, la.h);
      for (let i = 0; i < init.length; i++) init[i] *= 2;
    }
    const coarsest = L === pyrA.length - 1;
    const radius = coarsest ? (opts.coarseRadius || 5) : 2;
    const stride = (la.w * la.h > 60000) ? 2 : 1;
    flow = matchLevel(la.data, lb.data, la.w, la.h, init, radius, 3, stride);
  }
  return flow;
}

/** Forward/backward consistency: f(p) + b(p + f(p)) ~ 0.  Returns Uint8Array mask. */
export function fbConsistent(f, b, w, h, tol = 3.0) {
  const ok = new Uint8Array(w * h);
  const tmp = new Float32Array(2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 2;
    const fx = f[i], fy = f[i + 1];
    sampleBilinear(b, w, h, 2, x + fx, y + fy, tmp, 0);
    const err = Math.hypot(fx + tmp[0], fy + tmp[1]);
    ok[y * w + x] = err < tol + 0.1 * Math.hypot(fx, fy) ? 1 : 0;
  }
  return ok;
}

/** Zero unreliable flow, smooth, renormalise by blurred validity. */
export function cleanFlow(flow, ok, w, h, sigma = 6) {
  const f = new Float32Array(flow.length);
  const okf = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { okf[i] = ok[i]; if (ok[i]) { f[i * 2] = flow[i * 2]; f[i * 2 + 1] = flow[i * 2 + 1]; } }
  const fb = gaussianBlur(f, w, h, 2, sigma);
  const ob = gaussianBlur(okf, w, h, 1, sigma);
  for (let i = 0; i < w * h; i++) {
    const s = Math.min(ob[i] * 2, 1) / Math.max(ob[i], 0.15);
    fb[i * 2] *= s; fb[i * 2 + 1] *= s;
  }
  return fb;
}

