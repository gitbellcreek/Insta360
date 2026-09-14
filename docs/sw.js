// Network-first service worker: every same-origin request is revalidated with
// the server (cache: "no-cache" => conditional GET, so unchanged files cost a
// 304), which sidesteps GitHub Pages' 10 minute max-age.  The last good copy is
// kept only as an offline fallback, so a plain reload always gets the newest
// version of the app and nobody needs a hard refresh.
const CACHE = "insta360stitch-runtime";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil((async () => {
  const keys = await caches.keys();
  await Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)));
  await self.clients.claim();
})()));

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const fresh = await fetch(req, { cache: "no-cache" });
      if (fresh && fresh.ok) cache.put(req, fresh.clone());
      return fresh;
    } catch (err) {
      const cached = await cache.match(req);
      if (cached) return cached;
      throw err;
    }
  })());
});
