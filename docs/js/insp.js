// Parser for Insta360 .insp files (JPEG + trailer).  Mirrors insta360stitch/insp.py.
const MAGIC = "8db42d694ccc418790edff439fe026bf";
const OFFSET_RE = /(\d+)_(-?[\d.]+(?:_-?[\d.]+){13,})/;

export class Calibration {
  constructor(str) {
    const parts = str.split("_");
    this.version = parseInt(parts[0], 10);
    const n = parts.slice(1).map(Number);
    this.lenses = [0, 6].map(o => ({ cx: n[o], cy: n[o + 1], r: n[o + 2], yaw: n[o + 3], pitch: n[o + 4], roll: n[o + 5] }));
    this.width = n[12] | 0; this.height = n[13] | 0;
    this.extra = n.slice(14);
    this.raw = str;
  }
  /** [[X, Y, r], [X, Y, r]] lens centres in the stored JPEG frame (axes swapped when the
   *  calibration is expressed in the sensor's portrait frame). */
  storedCentres(imgW, imgH) {
    const maxCy = Math.max(...this.lenses.map(l => l.cy));
    const maxCx = Math.max(...this.lenses.map(l => l.cx));
    if (maxCy > imgH || (maxCx <= imgH && maxCy > imgW / 2)) return this.lenses.map(l => [l.cy, l.cx, l.r]);
    return this.lenses.map(l => [l.cx, l.cy, l.r]);
  }
}

function* protobufFields(buf) {
  const n = buf.length; let i = 0;
  const varint = () => { let s = 0, v = 0; while (i < n) { const b = buf[i++]; v += (b & 0x7f) * Math.pow(2, s); s += 7; if (!(b & 0x80)) break; } return v; };
  while (i < n) {
    const key = varint(); const fno = Math.floor(key / 8), wt = key & 7; let val;
    if (wt === 0) val = varint();
    else if (wt === 1) { val = buf.subarray(i, i + 8); i += 8; }
    else if (wt === 2) { const ln = varint(); val = buf.subarray(i, i + ln); i += ln; }
    else if (wt === 5) { val = buf.subarray(i, i + 4); i += 4; }
    else return;
    yield [fno, wt, val];
  }
}
const ascii = (u8) => { let s = ""; for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]); return s; };

function parseImu(rec) {
  const dv = new DataView(rec.buffer, rec.byteOffset, rec.byteLength);
  for (const [size, dbl] of [[56, true], [32, false]]) {
    if (rec.length % size !== 0 || rec.length < size) continue;
    const N = rec.length / size; const out = new Float64Array(N * 7); const norms = [];
    for (let k = 0; k < N; k++) {
      const o = k * size;
      out[k * 7] = Number(dv.getBigUint64(o, true));
      for (let j = 0; j < 6; j++) out[k * 7 + 1 + j] = dbl ? dv.getFloat64(o + 8 + j * 8, true) : dv.getFloat32(o + 8 + j * 4, true);
      norms.push(Math.hypot(out[k * 7 + 1], out[k * 7 + 2], out[k * 7 + 3]));
    }
    norms.sort((a, b) => a - b);
    const med = norms[norms.length >> 1];
    if (med > 0.5 && med < 2.0) return { n: N, data: out };
  }
  return null;
}

/** Mean gravity vector (unit) from the quietest half of the samples, or null. */
export function accelFromImu(imu) {
  if (!imu || imu.n === 0) return null;
  const d = imu.data, N = imu.n;
  const norms = new Float64Array(N);
  for (let k = 0; k < N; k++) norms[k] = Math.hypot(d[k * 7 + 1], d[k * 7 + 2], d[k * 7 + 3]);
  const sorted = Array.from(norms).sort((a, b) => a - b); const med = sorted[N >> 1];
  const dev = Array.from(norms, x => Math.abs(x - med)).sort((a, b) => a - b); const thr = dev[N >> 1];
  const s = [0, 0, 0]; let c = 0;
  for (let k = 0; k < N; k++) if (Math.abs(norms[k] - med) <= thr) { s[0] += d[k * 7 + 1]; s[1] += d[k * 7 + 2]; s[2] += d[k * 7 + 3]; c++; }
  if (!c) return null;
  const n = Math.hypot(...s); return n > 1e-6 ? [s[0] / n, s[1] / n, s[2] / n] : null;
}

/** Parse an ArrayBuffer holding an .insp file. */
export function parseInsp(buffer, name = "") {
  const data = new Uint8Array(buffer);
  if (data[0] !== 0xff || data[1] !== 0xd8) throw new Error(`${name}: not a JPEG based .insp file`);
  const out = { name, jpeg: data, records: new Map(), serial: "", model: "", firmware: "", calibration: null, imu: null, thumbnail: null };
  const tail = ascii(data.subarray(data.length - 32));
  if (tail === MAGIC && data.length > 72) {
    const dv = new DataView(buffer);
    const trailerLen = dv.getUint32(data.length - 40, true);
    const start = data.length - trailerLen;
    if (start > 0 && start < data.length) {
      out.jpeg = data.subarray(0, start);
      let pos = data.length - 72;
      while (pos - 6 >= start) {
        const rid = dv.getUint16(pos - 6, true), rlen = dv.getUint32(pos - 4, true);
        if (rlen > pos - 6 - start) break;
        out.records.set(rid, data.subarray(pos - 6 - rlen, pos - 6));
        pos = pos - 6 - rlen;
      }
    }
  }
  const info = out.records.get(0x0101);
  if (info) {
    for (const [, wt, val] of protobufFields(info)) {
      if (wt !== 2) continue;
      const text = ascii(val);
      const m = OFFSET_RE.exec(text);
      if (m && !out.calibration) { try { out.calibration = new Calibration(m[0]); } catch (e) { /* ignore */ } }
      else if (text.includes("Insta360") && !out.model) out.model = text;
      else if (/^[A-Z0-9]{10,20}$/.test(text) && !out.serial) out.serial = text;
      else if (text.startsWith("v") && text.includes("build") && !out.firmware) out.firmware = text;
    }
  }
  if (!out.calibration) {
    const m = OFFSET_RE.exec(ascii(data.subarray(Math.max(0, data.length - 4096))));
    if (m) out.calibration = new Calibration(m[0]);
  }
  const imu = out.records.get(0x0300);
  if (imu) out.imu = parseImu(imu);
  out.thumbnail = out.records.get(0x0200) || null;
  return out;
}

/** Bracket/burst siblings share the IMG_<date>_<time> prefix; borrow IMU data from one. */
export function siblingPrefix(name) {
  const stem = name.replace(/\.[^.]+$/, "");
  const parts = stem.split("_");
  return parts.length >= 3 ? parts.slice(0, 3).join("_") : null;
}

/** The APP1 Exif segment of a JPEG (Uint8Array) or null. */
export function exifSegment(jpeg) {
  let i = 2;
  while (i + 4 <= jpeg.length && jpeg[i] === 0xff && jpeg[i + 1] >= 0xe0 && jpeg[i + 1] <= 0xef) {
    const ln = (jpeg[i + 2] << 8) | jpeg[i + 3];
    if (jpeg[i + 1] === 0xe1 && ascii(jpeg.subarray(i + 4, i + 10)) === "Exif\0\0") return jpeg.slice(i, i + 2 + ln);
    i += 2 + ln;
  }
  return null;
}
