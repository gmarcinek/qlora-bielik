// Fetches every module reachable from index.html through the Vite dev server and reports slow, failing or empty ones.
const base = "http://127.0.0.1:5175";
const seen = new Map();
const queue = ["/"];
const importRe = /(?:import|export)[^'"]*?["'](\/[^"']+)["']|import\(["'](\/[^"']+)["']\)|src="(\/[^"]+)"/g;

async function get(path) {
  const started = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(base + path, { signal: controller.signal });
    const text = await response.text();
    return { status: response.status, ms: Math.round(performance.now() - started), text };
  } catch (error) {
    return { status: "ERR " + error.name, ms: Math.round(performance.now() - started), text: "" };
  } finally {
    clearTimeout(timer);
  }
}

while (queue.length) {
  const path = queue.shift();
  if (seen.has(path)) continue;
  seen.set(path, null);
  const result = await get(path);
  seen.set(path, result);
  for (const match of result.text.matchAll(importRe)) {
    const next = match[1] ?? match[2] ?? match[3];
    if (next && !seen.has(next) && !next.startsWith("/@vite/client")) queue.push(next);
  }
}

const rows = [...seen.entries()].map(([path, r]) => ({ path, status: r.status, ms: r.ms, len: r.text.length, emptyCss: /__vite__css = ""/.test(r.text) }));
console.log(`modules: ${rows.length}`);
for (const row of rows.filter((r) => r.status !== 200 || r.ms > 3000 || r.len < 50 || r.emptyCss)) console.log("PROBLEM", JSON.stringify(row));
for (const row of rows.sort((a, b) => b.ms - a.ms).slice(0, 5)) console.log("slowest", JSON.stringify(row));
const css = await get("/src/styles.css");
const fontImport = css.text.match(/@import url\(\\?"([^"\\]+)/)?.[1];
if (fontImport) {
  const started = performance.now();
  try {
    const response = await fetch(fontImport, { signal: AbortSignal.timeout(10000) });
    console.log(`google fonts ${response.status} in ${Math.round(performance.now() - started)}ms`);
  } catch (error) {
    console.log(`google fonts FAILED after ${Math.round(performance.now() - started)}ms: ${error.name}`);
  }
}
