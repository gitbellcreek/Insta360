// Typed-array image operations used by the seam pipeline.  Images are
// Float32Array in row-major order; "gray" has 1 channel, "rgb" 3 channels.

export function rgbaToRgbF32(rgba, n) {
  const o = new Float32Array(n * 3);
  for (let i = 0, j = 0; i < n; i++, j += 4) { o[i * 3] = rgba[j]; o[i * 3 + 1] = rgba[j + 1]; o[i * 3 + 2] = rgba[j + 2]; }
  return o;
}
export function rgbaAlphaMask(rgba, n) { const o = new Uint8Array(n); for (let i = 0; i < n; i++) o[i] = rgba[i * 4 + 3] > 127 ? 1 : 0; return o; }
export function rgbToGray(rgb, n) { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = 0.299 * rgb[i * 3 + 2] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3]; return o; }

/** Separable Gaussian blur (border replicate) on a c-channel image. */
export function gaussianBlur(src, w, h, c, sigma) {
  if (sigma <= 0) return src.slice();
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1); let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-i * i / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) { let xx = x + i; if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1; acc += k[i + r] * src[(row + xx) * c + ch]; }
        tmp[(row + x) * c + ch] = acc;
      }
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) { let yy = y + i; if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1; acc += k[i + r] * tmp[(yy * w + x) * c + ch]; }
        out[(y * w + x) * c + ch] = acc;
      }
    }
  }
  return out;
}

/** Box mean over a (2r+1)^2 window, border replicate (single channel). */
export function boxMean(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w; let acc = 0;
    for (let i = -r; i <= r; i++) acc += src[row + Math.min(w - 1, Math.max(0, i))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / n;
      const xa = x - r, xb = x + r + 1;
      acc += src[row + Math.min(w - 1, xb)] - src[row + Math.max(0, xa)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += tmp[Math.min(h - 1, Math.max(0, i)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / n;
      const ya = y - r, yb = y + r + 1;
      acc += tmp[Math.min(h - 1, yb) * w + x] - tmp[Math.max(0, ya) * w + x];
    }
  }
  return out;
}

/** Gaussian 5-tap pyramid reduction (OpenCV pyrDown semantics: ceil sizes). */
export function pyrDown(src, w, h, c) {
  const w2 = Math.ceil(w / 2), h2 = Math.ceil(h / 2);
  const k = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  const tmp = new Float32Array(w2 * h * c);
  for (let y = 0; y < h; y++) for (let x2 = 0; x2 < w2; x2++) for (let ch = 0; ch < c; ch++) {
    let acc = 0;
    for (let i = -2; i <= 2; i++) { let xx = 2 * x2 + i; if (xx < 0) xx = -xx; if (xx >= w) xx = 2 * w - 2 - xx; if (xx < 0) xx = 0; acc += k[i + 2] * src[(y * w + xx) * c + ch]; }
    tmp[(y * w2 + x2) * c + ch] = acc;
  }
  const out = new Float32Array(w2 * h2 * c);
  for (let y2 = 0; y2 < h2; y2++) for (let x = 0; x < w2; x++) for (let ch = 0; ch < c; ch++) {
    let acc = 0;
    for (let i = -2; i <= 2; i++) { let yy = 2 * y2 + i; if (yy < 0) yy = -yy; if (yy >= h) yy = 2 * h - 2 - yy; if (yy < 0) yy = 0; acc += k[i + 2] * tmp[(yy * w2 + x) * c + ch]; }
    out[(y2 * w2 + x) * c + ch] = acc;
  }
  return { data: out, w: w2, h: h2 };
}

/** Pyramid expansion to (W, H) (OpenCV pyrUp semantics). */
export function pyrUp(src, w, h, c, W, H) {
  const k = [1 / 8, 4 / 8, 6 / 8, 4 / 8, 1 / 8]; // *2 for zero insertion, applied per axis
  const tmp = new Float32Array(W * h * c);
  for (let y = 0; y < h; y++) for (let X = 0; X < W; X++) for (let ch = 0; ch < c; ch++) {
    let acc = 0;
    for (let i = -2; i <= 2; i++) { const xs = X + i; if (xs & 1) continue; let xx = xs >> 1; if (xx < 0) xx = -xx; if (xx >= w) xx = 2 * w - 2 - xx; if (xx < 0) xx = 0; acc += k[i + 2] * src[(y * w + xx) * c + ch]; }
    tmp[(y * W + X) * c + ch] = acc;
  }
  const out = new Float32Array(W * H * c);
  for (let Y = 0; Y < H; Y++) for (let X = 0; X < W; X++) for (let ch = 0; ch < c; ch++) {
    let acc = 0;
    for (let i = -2; i <= 2; i++) { const ys = Y + i; if (ys & 1) continue; let yy = ys >> 1; if (yy < 0) yy = -yy; if (yy >= h) yy = 2 * h - 2 - yy; if (yy < 0) yy = 0; acc += k[i + 2] * tmp[(yy * W + X) * c + ch]; }
    out[(Y * W + X) * c + ch] = acc;
  }
  return out;
}

/** Bilinear sample of a c-channel image at (x, y), border replicate. */
export function sampleBilinear(src, w, h, c, x, y, out, oi) {
  if (x < 0) x = 0; else if (x > w - 1) x = w - 1;
  if (y < 0) y = 0; else if (y > h - 1) y = h - 1;
  const x0 = x | 0, y0 = y | 0, x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0, fy = y - y0;
  const i00 = (y0 * w + x0) * c, i01 = (y0 * w + x1) * c, i10 = (y1 * w + x0) * c, i11 = (y1 * w + x1) * c;
  for (let ch = 0; ch < c; ch++) {
    out[oi + ch] = (src[i00 + ch] * (1 - fx) + src[i01 + ch] * fx) * (1 - fy) + (src[i10 + ch] * (1 - fx) + src[i11 + ch] * fx) * fy;
  }
}

/** Resize a c-channel image with bilinear sampling. */
export function resizeBilinear(src, w, h, c, W, H) {
  const out = new Float32Array(W * H * c);
  const sx = w / W, sy = h / H;
  for (let Y = 0; Y < H; Y++) for (let X = 0; X < W; X++) sampleBilinear(src, w, h, c, (X + 0.5) * sx - 0.5, (Y + 0.5) * sy - 0.5, out, (Y * W + X) * c);
  return out;
}

/** Sobel x on a gray image. */
export function sobelX(src, w, h) {
  const out = new Float32Array(w * h);
  const at = (x, y) => src[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    out[y * w + x] = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
  return out;
}

/** Gradient magnitude (central differences) of a gray image. */
export function gradMag(src, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const gx = src[y * w + x + 1] - src[y * w + x - 1], gy = src[(y + 1) * w + x] - src[(y - 1) * w + x];
    out[y * w + x] = Math.hypot(gx, gy);
  }
  return out;
}

/** 3x3 median filter on a single channel of an interleaved image. */
export function median3(src, w, h, c, ch) {
  const out = src.slice();
  const buf = new Float32Array(9);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) buf[n++] = src[((y + dy) * w + x + dx) * c + ch];
    const a = Array.from(buf).sort((p, q) => p - q);
    out[(y * w + x) * c + ch] = a[4];
  }
  return out;
}

export function erodeMask(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let ok = 1;
    for (let dy = -r; dy <= r && ok; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h || !mask[yy * w + xx]) { ok = 0; break; }
    }
    out[y * w + x] = ok;
  }
  return out;
}
export function dilateMask(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let on = 0;
    for (let dy = -r; dy <= r && !on; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && mask[yy * w + xx]) { on = 1; break; }
    }
    out[y * w + x] = on;
  }
  return out;
}
