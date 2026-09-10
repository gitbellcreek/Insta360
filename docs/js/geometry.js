// Geometry: 3x3 rotations (row-major Float64Array(9)), the fisheye lens model
// and the equirectangular conventions.  Mirrors insta360stitch/geometry.py.
export const DEG = Math.PI / 180;

export function mat3Identity() { return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); }
export function mat3Mul(a, b) {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return o;
}
export function mat3T(a) { return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]); }
export function mat3Apply(m, v) {
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}
export function rotX(deg) { const c = Math.cos(deg * DEG), s = Math.sin(deg * DEG); return new Float64Array([1, 0, 0, 0, c, -s, 0, s, c]); }
export function rotY(deg) { const c = Math.cos(deg * DEG), s = Math.sin(deg * DEG); return new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]); }
export function rotZ(deg) { const c = Math.cos(deg * DEG), s = Math.sin(deg * DEG); return new Float64Array([c, -s, 0, s, c, 0, 0, 0, 1]); }
export function eulerYPR(yaw, pitch, roll) { return mat3Mul(mat3Mul(rotY(yaw), rotX(pitch)), rotZ(roll)); }
export function normalize(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

/** Minimal rotation R with R a = b (unit vectors). */
export function rotationBetween(a, b) {
  a = normalize(a); b = normalize(b);
  const v = cross(a, b);
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const s = Math.hypot(v[0], v[1], v[2]);
  if (s < 1e-12) {
    if (c > 0) return mat3Identity();
    let axis = cross(a, [1, 0, 0]);
    if (Math.hypot(...axis) < 1e-6) axis = cross(a, [0, 1, 0]);
    axis = normalize(axis);
    const o = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[i * 3 + j] = 2 * axis[i] * axis[j] - (i === j ? 1 : 0);
    return o;
  }
  const k = [v[0] / s, v[1] / s, v[2] / s];
  const K = new Float64Array([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]);
  const KK = mat3Mul(K, K);
  const o = mat3Identity();
  for (let i = 0; i < 9; i++) o[i] += s * K[i] + (1 - c) * KK[i];
  return o;
}
/** World rotation R with R (0,1,0) = upCam. */
export function levelRotation(upCam) { return rotationBetween([0, 1, 0], upCam); }
/** ONE X accelerometer (sensor frame) -> camera-frame up vector in our world axes. */
export function upFromAccel(a) { return normalize([a[2], -a[0], a[1]]); }

export function equirectDir(u, v, W, H) {
  const lon = ((u + 0.5) / W * 2 - 1) * Math.PI;
  const lat = (0.5 - (v + 0.5) / H) * Math.PI;
  const cl = Math.cos(lat);
  return [cl * Math.sin(lon), Math.sin(lat), cl * Math.cos(lon)];
}
export function lonToU(lonDeg, W) { return (lonDeg / 180 + 1) * 0.5 * W; }

export class Lens {
  constructor(o) {
    Object.assign(this, { cx: 0, cy: 0, rCal: 1, fov: 200, k1: 0, k2: 0, yaw: 0, pitch: 0, roll: 0, flip: false, edgeFrac: 0.985 }, o);
  }
  clone() { return new Lens(this); }
  g(t) { const t2 = t * t; return t * (1 + t2 * (this.k1 + t2 * this.k2)); }
  dg(t) { const t2 = t * t; return 1 + t2 * (3 * this.k1 + 5 * this.k2 * t2); }
  get f() { return this.rCal / this.g(this.fov * 0.5 * DEG); }
  thetaOfR(r) { const x = r / this.f; let t = x; for (let i = 0; i < 6; i++) t -= (this.g(t) - x) / this.dg(t); return t; }
  get thetaMax() { return this.thetaOfR(this.edgeFrac * this.rCal); }
  /** world -> lens rotation */
  M() { let R = eulerYPR(this.yaw, this.pitch, this.roll); if (this.flip) R = mat3Mul(R, rotY(180)); return mat3T(R); }
  /** lens-frame unit vector -> [X, Y, theta] in the stored image */
  project(v) {
    const theta = Math.acos(Math.max(-1, Math.min(1, v[2])));
    const rho = Math.max(Math.hypot(v[0], v[1]), 1e-9);
    const r = this.f * this.g(theta);
    return [this.cx + r * v[1] / rho, this.cy + r * v[0] / rho, theta];
  }
  /** stored image pixel -> lens-frame unit vector */
  unproject(X, Y) {
    const dx = X - this.cx, dy = Y - this.cy;
    const r = Math.hypot(dx, dy);
    const theta = this.thetaOfR(r);
    const s = Math.sin(theta) / Math.max(r, 1e-9);
    return [dy * s, dx * s, Math.cos(theta)];
  }
  toJSON() { const o = {}; for (const k of ["cx", "cy", "rCal", "fov", "k1", "k2", "yaw", "pitch", "roll", "flip", "edgeFrac"]) o[k] = this[k]; return o; }
}

/** Build the two lenses from a parsed calibration (see insp.js). */
export function lensesFromCalibration(cal, imgW, imgH, fov = 200) {
  const centres = cal.storedCentres(imgW, imgH);
  return cal.lenses.map((lc, i) => new Lens({ cx: centres[i][0], cy: centres[i][1], rCal: centres[i][2], fov, yaw: lc.yaw, pitch: lc.pitch, roll: lc.roll, flip: i === 1 }));
}
