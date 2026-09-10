// Hugin-style alignment in the browser.  Control points come from the dense
// flow between the two lenses in the overlap bands (textured, forward/backward
// consistent pixels only); the lens FOV and rear-lens yaw/pitch/roll are then
// fitted with Levenberg-Marquardt under a Cauchy robust loss.
import { DEG, equirectDir, mat3Apply, mat3T } from "./geometry.js";
import { denseFlow, fbConsistent } from "./flow.js";
import { erodeMask, gradMag } from "./imageops.js";

function residuals(p, base, obs) {
  const l1 = base[0].clone(), l2 = base[1].clone();
  l1.fov = l2.fov = p[0]; l2.yaw = p[1]; l2.pitch = p[2]; l2.roll = p[3];
  const M1T = mat3T(l1.M()), M2T = mat3T(l2.M());
  const r = new Float64Array(obs.length * 3);
  for (let i = 0; i < obs.length; i++) {
    const o = obs[i];
    const d1 = mat3Apply(M1T, l1.unproject(o[0], o[1]));
    const d2 = mat3Apply(M2T, l2.unproject(o[2], o[3]));
    r[i * 3] = (d1[0] - d2[0]) / DEG; r[i * 3 + 1] = (d1[1] - d2[1]) / DEG; r[i * 3 + 2] = (d1[2] - d2[2]) / DEG;
  }
  return r;
}
function robustCost(r, c) { let s = 0; for (let i = 0; i < r.length; i += 3) s += Math.log1p((r[i] * r[i] + r[i + 1] * r[i + 1] + r[i + 2] * r[i + 2]) / (c * c)); return s; }
export function robustRms(r, c = 1.0) {
  let s = 0, n = 0;
  for (let i = 0; i < r.length; i += 3) { const r2 = r[i] * r[i] + r[i + 1] * r[i + 1] + r[i + 2] * r[i + 2]; if (r2 < 9 * c * c) { s += r2; n++; } }
  return n ? Math.sqrt(s / n) : NaN;
}
function solve4(A, g) {
  // Gaussian elimination on a 4x4 system A x = g (A row-major)
  const n = 4, M = A.map((row, i) => [...row, g[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) return null;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}

export function optimise(lenses, obs, iterations = 25, c = 1.0) {
  let p = [lenses[0].fov, lenses[1].yaw, lenses[1].pitch, lenses[1].roll];
  const steps = [0.05, 0.02, 0.02, 0.02];
  let lam = 1e-2;
  let r = residuals(p, lenses, obs), cost = robustCost(r, c);
  const N = r.length;
  for (let it = 0; it < iterations; it++) {
    const J = [];
    for (let j = 0; j < 4; j++) { const pj = p.slice(); pj[j] += steps[j]; const rj = residuals(pj, lenses, obs); const col = new Float64Array(N); for (let i = 0; i < N; i++) col[i] = (rj[i] - r[i]) / steps[j]; J.push(col); }
    const w = new Float64Array(N);
    for (let i = 0; i < N; i += 3) { const r2 = r[i] * r[i] + r[i + 1] * r[i + 1] + r[i + 2] * r[i + 2]; w[i] = w[i + 1] = w[i + 2] = 1 / (1 + r2 / (c * c)); }
    const A = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], g = [0, 0, 0, 0];
    for (let a = 0; a < 4; a++) { for (let b = a; b < 4; b++) { let s = 0; for (let i = 0; i < N; i++) s += w[i] * J[a][i] * J[b][i]; A[a][b] = A[b][a] = s; } let s = 0; for (let i = 0; i < N; i++) s += w[i] * J[a][i] * r[i]; g[a] = s; }
    let improved = false, delta = null;
    for (let t = 0; t < 8; t++) {
      const Ad = A.map((row, i) => row.map((v, j) => i === j ? v + lam * (v + 1e-9) : v));
      delta = solve4(Ad, g.map(x => -x));
      if (!delta) { lam *= 5; continue; }
      const pn = p.map((v, i) => v + delta[i]);
      const rn = residuals(pn, lenses, obs), cn = robustCost(rn, c);
      if (cn < cost) { p = pn; r = rn; cost = cn; lam = Math.max(lam / 3, 1e-6); improved = true; break; }
      lam *= 5;
    }
    if (!improved || Math.max(...delta.map(Math.abs)) < 1e-5) break;
  }
  const out = [lenses[0].clone(), lenses[1].clone()];
  out[0].fov = out[1].fov = p[0]; out[1].yaw = p[1]; out[1].pitch = p[2]; out[1].roll = p[3];
  return { lenses: out, rms: robustRms(r, c) };
}

/** Control points (stored-frame pixel pairs) from the flow in one band. */
function controlPoints(band, lenses, renderW, renderH, maxPoints) {
  const { A, B, va, vb, w, h, u0 } = band;
  const overlap = new Uint8Array(w * h); for (let i = 0; i < w * h; i++) overlap[i] = va[i] & vb[i];
  const ov = erodeMask(overlap, w, h, 4);
  const fab = denseFlow(A, B, w, h, { coarseRadius: 6 });
  const fba = denseFlow(B, A, w, h, { coarseRadius: 6 });
  const ok = fbConsistent(fab, fba, w, h, 2.0);
  const grad = gradMag(A, w, h);
  const cand = [];
  for (let y = 4; y < h - 4; y += 3) for (let x = 4; x < w - 4; x += 3) { const i = y * w + x; if (ov[i] && ok[i]) cand.push([x, y, grad[i]]); }
  if (cand.length < 10) return [];
  cand.sort((p, q) => q[2] - p[2]);
  const keep = cand.slice(0, Math.min(maxPoints, Math.max(50, Math.floor(cand.length * 0.4))));
  const M0 = lenses[0].M(), M1 = lenses[1].M();
  const obs = [];
  for (const [x, y] of keep) {
    const i = (y * w + x) * 2;
    const dA = equirectDir(x + u0, y, renderW, renderH);
    const dB = equirectDir(x + fab[i] + u0, y + fab[i + 1], renderW, renderH);
    const pA = lenses[0].project(mat3Apply(M0, dA));
    const pB = lenses[1].project(mat3Apply(M1, dB));
    obs.push([pA[0], pA[1], pB[0], pB[1]]);
  }
  return obs;
}

/**
 * renderBands(lenses, renderW, bandHalfDeg) must return, per seam, an object
 * { A, B (gray Float32Array), va, vb (Uint8Array), w, h, u0 } rendered with
 * the given lenses.
 */
export async function calibrate(renderBands, lenses, { rounds = 2, renderW = 2048, minPoints = 60, fovBounds = [170, 240], maxRotChange = 15, log = () => {}, tick = async () => {} } = {}) {
  let cur = lenses.map(l => l.clone());
  let n = 0, rms = NaN, rmsBefore = NaN;
  for (let rnd = 0; rnd < rounds; rnd++) {
    const bands = await renderBands(cur, renderW, 20);
    let obs = [];
    for (const b of bands) { obs = obs.concat(controlPoints(b, cur, renderW, renderW / 2, 2500)); await tick(); }
    n = obs.length;
    log(`alignment round ${rnd + 1}: ${n} control points`);
    if (n < minPoints) return { lenses, nPoints: n, rms, accepted: false, message: `only ${n} control points, keeping previous lens model` };
    if (rnd === 0) rmsBefore = robustRms(residuals([lenses[0].fov, lenses[1].yaw, lenses[1].pitch, lenses[1].roll], lenses, obs));
    ({ lenses: cur, rms } = optimise(cur, obs));
    await tick();
  }
  const fovOk = cur[0].fov >= fovBounds[0] && cur[0].fov <= fovBounds[1];
  const rotOk = ["yaw", "pitch", "roll"].every(k => Math.abs(cur[1][k] - lenses[1][k]) <= maxRotChange);
  const better = Number.isNaN(rmsBefore) || rms <= rmsBefore * 1.05 + 0.05;
  if (!(fovOk && rotOk && better)) return { lenses, nPoints: n, rms, accepted: false, message: `rejected implausible solution (fov ${cur[0].fov.toFixed(1)}, rms ${rms.toFixed(2)} deg, was ${rmsBefore.toFixed(2)})` };
  return { lenses: cur, nPoints: n, rms, accepted: true, message: `fov ${cur[0].fov.toFixed(2)} deg, rear lens ypr (${cur[1].yaw.toFixed(2)}, ${cur[1].pitch.toFixed(2)}, ${cur[1].roll.toFixed(2)}), rms ${rms.toFixed(3)} deg (was ${rmsBefore.toFixed(2)}) from ${n} points` };
}
