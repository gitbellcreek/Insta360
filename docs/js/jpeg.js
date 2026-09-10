// JPEG output with the original Exif and GPano XMP so viewers treat the file as a 360 photo.

function gpanoXmp(width, height) {
  const xmp = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>' +
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">' +
    '<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">' +
    '<GPano:ProjectionType>equirectangular</GPano:ProjectionType><GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>' +
    `<GPano:FullPanoWidthPixels>${width}</GPano:FullPanoWidthPixels><GPano:FullPanoHeightPixels>${height}</GPano:FullPanoHeightPixels>` +
    `<GPano:CroppedAreaImageWidthPixels>${width}</GPano:CroppedAreaImageWidthPixels><GPano:CroppedAreaImageHeightPixels>${height}</GPano:CroppedAreaImageHeightPixels>` +
    '<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels><GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>' +
    '<GPano:PoseHeadingDegrees>0.0</GPano:PoseHeadingDegrees><GPano:StitchingSoftware>insta360stitch-web</GPano:StitchingSoftware>' +
    '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>';
  const payload = new TextEncoder().encode("http://ns.adobe.com/xap/1.0/\0" + xmp);
  const seg = new Uint8Array(4 + payload.length);
  seg[0] = 0xff; seg[1] = 0xe1; seg[2] = (payload.length + 2) >> 8; seg[3] = (payload.length + 2) & 0xff; seg.set(payload, 4);
  return seg;
}

/** Encode RGBA pixels as a 360 JPEG Blob. */
export async function encodePanoJpeg(rgba, W, H, quality = 0.95, exifSeg = null) {
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  const img = new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, W * H * 4), W, H);
  ctx.putImageData(img, 0, 0);
  const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", quality));
  canvas.width = canvas.height = 1;
  if (!blob) throw new Error(`the browser could not encode a ${W}x${H} JPEG (try a smaller width)`);
  const data = new Uint8Array(await blob.arrayBuffer());
  const parts = [data.subarray(0, 2)];
  let i = 2;
  if (data[2] === 0xff && data[3] === 0xe0) { const ln = (data[4] << 8) | data[5]; parts.push(data.subarray(2, 4 + ln)); i = 4 + ln; }
  if (exifSeg) parts.push(exifSeg);
  parts.push(gpanoXmp(W, H));
  parts.push(data.subarray(i));
  return new Blob(parts, { type: "image/jpeg" });
}
