// Small WebGL2 helper: full-screen-triangle programs, RGBA8 framebuffers,
// readback.  Texture row 0 is always the top image row (we never flip), and
// shaders derive the equirect row from gl_FragCoord.y - 0.5 so readPixels
// returns top-down images.

const VS = `#version 300 es
void main(){ vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0); gl_Position = vec4(p, 0.0, 1.0); }`;

export class GL {
  constructor(canvas) {
    this.canvas = canvas || document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false, premultipliedAlpha: false, alpha: true, depth: false, stencil: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.programs = new Map();
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  }
  program(name, fs) {
    if (this.programs.has(name)) return this.programs.get(name);
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(p));
    const prog = { p, u: {} };
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); prog.u[info.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, info.name); }
    this.programs.set(name, prog);
    return prog;
  }
  texture(w, h, source = null, { linear = true, wrapS = "clamp" } = {}) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (source && !(source instanceof Uint8Array)) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS === "repeat" ? gl.REPEAT : gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return { t, w, h };
  }
  subImage(tex, x, y, w, h, rgba) {
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, tex.t);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  }
  framebuffer(w, h, opts) {
    const gl = this.gl;
    const tex = this.texture(w, h, null, opts);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.t, 0);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error("framebuffer incomplete: " + st);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fb, tex, w, h };
  }
  /** Run a program into a framebuffer (or the canvas when fbo is null). */
  draw(prog, fbo, setUniforms, textures) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ? fbo.fb : null);
    gl.viewport(0, 0, fbo ? fbo.w : this.canvas.width, fbo ? fbo.h : this.canvas.height);
    gl.useProgram(prog.p);
    let unit = 0;
    for (const [name, tex] of Object.entries(textures || {})) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex.t); gl.uniform1i(prog.u[name], unit); unit++;
    }
    setUniforms(gl, prog.u);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  readPixels(fbo, x = 0, y = 0, w = fbo.w, h = fbo.h) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fb);
    const out = new Uint8Array(w * h * 4);
    gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
  deleteFramebuffer(fbo) { this.gl.deleteFramebuffer(fbo.fb); this.gl.deleteTexture(fbo.tex.t); }
  deleteTexture(tex) { this.gl.deleteTexture(tex.t); }
  /** row-major 3x3 -> column-major Float32Array for uniformMatrix3fv */
  static mat3(m) { return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]); }
}

const LENS_GLSL = `
uniform sampler2D uSrc;
uniform vec2 uSrcSize;
uniform mat3 uM[2];
uniform vec4 uLensA[2];   // cx, cy, f, thetaMax
uniform vec2 uLensB[2];   // k1, k2
uniform vec3 uGain[2];
uniform float uFeather;   // radians
// returns rgb, alpha = soft validity; hard validity in .w of second return via out param
vec4 sampleLens(int i, vec3 d, out float hard) {
  vec3 v = uM[i] * d;
  float theta = acos(clamp(v.z, -1.0, 1.0));
  float rho = max(length(v.xy), 1e-9);
  float t2 = theta * theta;
  float r = uLensA[i].z * theta * (1.0 + t2 * (uLensB[i].x + t2 * uLensB[i].y));
  vec2 p = vec2(uLensA[i].x + r * v.y / rho, uLensA[i].y + r * v.x / rho);
  float inside = (p.x > 1.5 && p.x < uSrcSize.x - 2.5 && p.y > 1.5 && p.y < uSrcSize.y - 2.5) ? 1.0 : 0.0;
  hard = (theta < uLensA[i].w) ? inside : 0.0;
  float a = (1.0 - smoothstep(uLensA[i].w - uFeather, uLensA[i].w, theta)) * inside;
  vec3 c = texture(uSrc, (p + 0.5) / uSrcSize).rgb * uGain[i];
  return vec4(c, a);
}`;

export const FS_PROJECT = `#version 300 es
precision highp float;
const float PI = 3.141592653589793;
${LENS_GLSL}
uniform vec2 uOutSize;   // full equirect W, H
uniform float uU0;       // first column of this framebuffer in the equirect
uniform int uMode;       // 0 composite, 1 lens 0 only, 2 lens 1 only
out vec4 o;
void main() {
  float u = uU0 + gl_FragCoord.x - 0.5;
  float v = gl_FragCoord.y - 0.5;
  float lon = ((u + 0.5) / uOutSize.x * 2.0 - 1.0) * PI;
  float lat = (0.5 - (v + 0.5) / uOutSize.y) * PI;
  vec3 d = vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  float ha, hb;
  if (uMode == 1) { vec4 s = sampleLens(0, d, ha); o = vec4(s.rgb * ha, ha); return; }
  if (uMode == 2) { vec4 s = sampleLens(1, d, hb); o = vec4(s.rgb * hb, hb); return; }
  vec4 a = sampleLens(0, d, ha);
  vec4 b = sampleLens(1, d, hb);
  bool front = abs(lon) < PI * 0.5;
  vec4 p = front ? a : b;
  vec4 q = front ? b : a;
  o = vec4(mix(q.rgb, p.rgb, p.a), 1.0);
}`;

export const FS_ROTATE = `#version 300 es
precision highp float;
const float PI = 3.141592653589793;
uniform sampler2D uPano;
uniform vec2 uSize;
uniform mat3 uR;
out vec4 o;
void main() {
  float u = gl_FragCoord.x - 0.5, v = gl_FragCoord.y - 0.5;
  float lon = ((u + 0.5) / uSize.x * 2.0 - 1.0) * PI;
  float lat = (0.5 - (v + 0.5) / uSize.y) * PI;
  vec3 d = vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  vec3 c = uR * d;
  float lon2 = atan(c.x, c.z);
  float lat2 = asin(clamp(c.y, -1.0, 1.0));
  o = vec4(texture(uPano, vec2((lon2 / PI + 1.0) * 0.5, 0.5 - lat2 / PI)).rgb, 1.0);
}`;

export const FS_VIEW = `#version 300 es
precision highp float;
const float PI = 3.141592653589793;
uniform sampler2D uPano;
uniform vec2 uView;      // canvas size in pixels
uniform float uFocal;    // pixels
uniform mat3 uR;         // camera -> world
uniform int uFlat;
uniform vec4 uFlatRect;  // x, y, w, h of the flat image on the canvas
out vec4 o;
void main() {
  if (uFlat == 1) {
    vec2 p = vec2(gl_FragCoord.x, uView.y - gl_FragCoord.y);
    vec2 t = (p - uFlatRect.xy) / uFlatRect.zw;
    if (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0) { o = vec4(0.09, 0.09, 0.1, 1.0); return; }
    o = vec4(texture(uPano, t).rgb, 1.0); return;
  }
  vec3 d = normalize(vec3((gl_FragCoord.x - uView.x * 0.5) / uFocal, (gl_FragCoord.y - uView.y * 0.5) / uFocal, 1.0));
  vec3 c = uR * d;
  float lon = atan(c.x, c.z);
  float lat = asin(clamp(c.y, -1.0, 1.0));
  o = vec4(texture(uPano, vec2((lon / PI + 1.0) * 0.5, 0.5 - lat / PI)).rgb, 1.0);
}`;
