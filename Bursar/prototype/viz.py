"""Generates a self-contained report.html: an animated replay of the simulation.

No dependencies, no server — data is inlined as JSON, charts are vanilla
JS + canvas. Open the file in any browser.
"""

import json

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Bursar — simulation replay</title>
<style>
 body{background:#111;color:#ddd;font:14px -apple-system,'Helvetica Neue',sans-serif;margin:24px;max-width:1020px}
 h1{font-size:20px;margin-bottom:2px} .sub{color:#888;font-size:13px}
 canvas{background:#181818;border:1px solid #2a2a2a;border-radius:6px;display:block;margin:12px 0}
 #controls{margin:14px 0;display:flex;gap:14px;align-items:center}
 button{background:#2a6df4;border:0;color:#fff;padding:6px 18px;border-radius:5px;font-size:14px;cursor:pointer}
 input[type=range]{width:440px}
 #clock{font-variant-numeric:tabular-nums;color:#aaa}
 #readout{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 2px 0}
 .chip{padding:5px 12px;border-radius:5px;background:#1e1e1e;border-left:4px solid #666;min-width:118px}
 .chip .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:#999}
 .chip .val{font-size:14px;font-variant-numeric:tabular-nums}
 #story{background:#181818;border:1px solid #2a2a2a;border-radius:6px;padding:14px 18px;margin:14px 0;line-height:1.55;max-width:1000px}
 #story h2{font-size:14px;margin:10px 0 4px 0;color:#eee}
 #story p{margin:6px 0;color:#c8c8c8}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 1px}
 .t{color:#eee;font-variant-numeric:tabular-nums;font-weight:600}
</style></head><body>
<h1>Bursar &mdash; simulation replay</h1>
<div class="sub">512-GPU agent-burst pool &middot; 2 h simulated &middot; storm window shaded orange &middot; prototype of paper Secs. 4&ndash;7, 10</div>

<div id="story">
<p><b>The idea being tested:</b> agents can generate unlimited work, but a broker (&ldquo;Bursar&rdquo;) meters how fast each
campaign may <i>consume resources</i>, using a token bucket: tokens refill at a sustained rate R and drain in proportion to
the GPUs a campaign holds. Save up tokens and you may burst far above R; run dry and you are forced back down. Nobody has
to be trusted &mdash; only metered.</p>
<h2>The cast &mdash; five campaigns sharing 512 GPUs</h2>
<p>
<span class="sw" style="background:#4e79a7"></span><b>adaptive-sci</b> reads the live price of capacity and runs its most valuable work first, deferring the rest &middot;
<span class="sw" style="background:#e15759"></span><b>naive-sci</b> has the <i>same task list</i> but always demands 256 GPUs and ignores prices &middot;
<span class="sw" style="background:#f28e2b"></span><b>storm</b> is an adversary that hammers its 384-GPU ceiling for 30 minutes &middot;
<span class="sw" style="background:#59a14f"></span><b>victim</b> is a small interactive user asking for 16 GPUs every 2 minutes &mdash; the fairness probe &middot;
<span class="sw" style="background:#8a8a8a"></span><b>backfill</b> soaks up idle capacity and is evicted on 60&nbsp;s notice.</p>
<h2>The story &mdash; press Play</h2>
<p><span class="t">0:00&ndash;0:40</span> Land rush. Both science campaigns burst; scarcity pushes the price of immediate
capacity to 3&times;. Adaptive (blue) buys only its best work at that price; naive (red) pays full fare for everything.
<span class="t">~0:40</span> naive&rsquo;s token bucket (middle chart, red line) runs dry, and Bursar forces it down to its
sustained 128 GPUs. <span class="t">1:00&ndash;1:30</span> The storm (orange) floods in, drains its own bucket, and is
throttled toward its 64-GPU sustained rate. Watch the bottom chart: the victim&rsquo;s white-outlined dots stay pinned at
zero latency straight through the attack, because no campaign may consume into another&rsquo;s guarantee.
<span class="t">1:30&ndash;2:00</span> Recovery: the storm is gone, and adaptive mops up its deferred low-value work on
cheap preemptible capacity while backfill (gray) fills the gaps.</p>
<p><b>Beyond GPUs:</b> envelopes also meter filesystem bandwidth and model inference (bottom two charts). Both science
campaigns want more I/O than their 6&nbsp;GB/s sustained rate, so Bursar throttles them at the source once their I/O buckets
drain &mdash; naive keeps holding 256 GPUs it cannot feed, while adaptive switches to tasks with the most science per
gigabyte and shrinks to what its I/O can support. And when the storm hammers the inference service at 200 tokens/s,
its inference envelope pins it to 50: even the adversary&rsquo;s <i>thinking</i> is metered, not trusted.</p>
<p><b>The outcome:</b> with identical tasks, the adaptive campaign ends with ~2.2&times; the science at ~3&times; lower cost per
unit than the naive one; the pool still runs at ~89% utilization; and every admission decision is journaled and explainable.</p>
</div>
<div id="controls">
 <button id="play">&#9654; Play</button>
 <input type="range" id="scrub" min="0" value="0">
 <span id="clock"></span>
</div>
<div id="readout"></div>
<canvas id="occ"  width="1000" height="280"></canvas>
<canvas id="tok"  width="1000" height="190"></canvas>
<canvas id="cong" width="1000" height="220"></canvas>
<canvas id="io"   width="1000" height="170"></canvas>
<canvas id="inf"  width="1000" height="170"></canvas>
<script>
const D = __DATA__;
const N = D.t.length, T = D.t[N-1];
const ML = 52, MR = 46, MT = 26, MB = 24;   // chart margins
let cur = N - 1, playing = false;

const scrub = document.getElementById('scrub');
scrub.max = N - 1; scrub.value = cur;

function X(i, W){ return ML + (W - ML - MR) * i / (N - 1); }
function hhmm(s){ const h = Math.floor(s/3600), m = Math.floor(s%3600/60), ss = s%60;
  return `${h}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`; }

function frame(g, c, title){
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = '#bbb'; g.font = '12px sans-serif'; g.textAlign = 'left';
  g.fillText(title, ML, 15);
  // storm shading
  const i0 = D.storm[0] / D.dt, i1 = D.storm[1] / D.dt;
  g.fillStyle = 'rgba(242,142,43,0.07)';
  g.fillRect(X(i0, c.width), MT, X(i1, c.width) - X(i0, c.width), c.height - MT - MB);
}

function cursorLine(g, c){
  g.strokeStyle = '#fff'; g.lineWidth = 1; g.setLineDash([3,3]);
  g.beginPath(); g.moveTo(X(cur, c.width), MT); g.lineTo(X(cur, c.width), c.height - MB); g.stroke();
  g.setLineDash([]);
}

function yAxis(g, c, vmax, fmt){
  g.fillStyle = '#777'; g.font = '11px sans-serif'; g.textAlign = 'right';
  for (let f = 0; f <= 1.001; f += 0.25){
    const y = c.height - MB - f * (c.height - MT - MB);
    g.fillText(fmt(f * vmax), ML - 6, y + 4);
    g.strokeStyle = '#222'; g.beginPath(); g.moveTo(ML, y); g.lineTo(c.width - MR, y); g.stroke();
  }
}

function drawOcc(){
  const c = document.getElementById('occ'), g = c.getContext('2d');
  frame(g, c, 'Pool occupancy — GPUs held (stacked)');
  yAxis(g, c, D.pool, v => Math.round(v));
  const plotH = c.height - MT - MB, y0 = c.height - MB;
  const base = new Array(N).fill(0);
  for (const name of D.campaigns){
    g.beginPath();
    for (let i = 0; i <= cur; i++){
      const y = y0 - (base[i] + D.held[name][i]) / D.pool * plotH;
      i === 0 ? g.moveTo(X(i, c.width), y) : g.lineTo(X(i, c.width), y);
    }
    for (let i = cur; i >= 0; i--) g.lineTo(X(i, c.width), y0 - base[i] / D.pool * plotH);
    g.closePath(); g.fillStyle = D.colors[name] + 'bb'; g.fill();
    for (let i = 0; i < N; i++) base[i] += D.held[name][i];
  }
  // legend
  let lx = ML + 270;
  g.font = '12px sans-serif'; g.textAlign = 'left';
  for (const name of D.campaigns){
    g.fillStyle = D.colors[name]; g.fillRect(lx, 7, 10, 10);
    g.fillStyle = '#bbb'; g.fillText(name, lx + 14, 16);
    lx += 14 + g.measureText(name).width + 18;
  }
  cursorLine(g, c);
}

function drawTok(){
  const c = document.getElementById('tok'), g = c.getContext('2d');
  frame(g, c, 'Token buckets — fill level, % of B  (drain above R forces reversion to sustained rate)');
  yAxis(g, c, 100, v => v + '%');
  const plotH = c.height - MT - MB, y0 = c.height - MB;
  for (const name of D.campaigns){
    if (name === 'backfill') continue;
    g.strokeStyle = D.colors[name]; g.lineWidth = 1.6; g.beginPath();
    for (let i = 0; i <= cur; i++){
      const y = y0 - D.tokens[name][i] * plotH;
      i === 0 ? g.moveTo(X(i, c.width), y) : g.lineTo(X(i, c.width), y);
    }
    g.stroke();
  }
  cursorLine(g, c);
}

function drawCong(){
  const c = document.getElementById('cong'), g = c.getContext('2d');
  frame(g, c, 'Congestion multiplier (line, left)  +  lease grants (dots: latency, right; green=warm  yellow=reclaim  red=cold)');
  const plotH = c.height - MT - MB, y0 = c.height - MB;
  yAxis(g, c, 3, v => v.toFixed(1) + 'x');
  const maxLat = Math.max(90, ...D.grants.map(x => x.lat));
  g.fillStyle = '#777'; g.textAlign = 'left';
  for (let f = 0; f <= 1.001; f += 0.5)
    g.fillText(Math.round(f * maxLat) + 's', c.width - MR + 6, y0 - f * plotH + 4);
  // multiplier step line
  g.strokeStyle = '#c7a34a'; g.lineWidth = 1.6; g.beginPath();
  for (let i = 0; i <= cur; i++){
    const y = y0 - (D.mult[i] / 3) * plotH;
    i === 0 ? g.moveTo(X(i, c.width), y) : g.lineTo(X(i, c.width), y);
  }
  g.stroke();
  // grants scatter
  const pathColor = {warm: '#59a14f', reclaim: '#f2c14e', cold: '#e15759'};
  for (const gr of D.grants){
    if (gr.t > D.t[cur]) break;
    const gx = ML + (c.width - ML - MR) * gr.t / T;
    const gy = y0 - (gr.lat / maxLat) * plotH;
    g.fillStyle = pathColor[gr.path];
    g.beginPath(); g.arc(gx, gy, gr.c === 'victim' ? 3.5 : 2.2, 0, 7); g.fill();
    if (gr.c === 'victim'){ g.strokeStyle = '#fff'; g.lineWidth = 0.7; g.stroke(); }
  }
  g.fillStyle = '#888'; g.fillText('victim grants outlined white', ML + 640, 15);
  cursorLine(g, c);
}

function drawIO(){
  const c = document.getElementById('io'), g = c.getContext('2d');
  frame(g, c, 'Filesystem I/O per campaign, GB/s — dashed line = science campaigns\' sustained rate R_io (bucket empty \u2192 pinned here)');
  const ymax = 18;
  yAxis(g, c, ymax, v => v.toFixed(0));
  const plotH = c.height - MT - MB, y0 = c.height - MB;
  // sustained-rate line
  g.strokeStyle = '#666'; g.setLineDash([5,4]); g.beginPath();
  g.moveTo(ML, y0 - 6/ymax*plotH); g.lineTo(c.width - MR, y0 - 6/ymax*plotH); g.stroke(); g.setLineDash([]);
  for (const name of D.campaigns){
    if (!D.io[name].some(v => v > 0)) continue;
    g.strokeStyle = D.colors[name]; g.lineWidth = 1.6; g.beginPath();
    for (let i = 0; i <= cur; i++){
      const y = y0 - Math.min(D.io[name][i], ymax)/ymax*plotH;
      i === 0 ? g.moveTo(X(i, c.width), y) : g.lineTo(X(i, c.width), y);
    }
    g.stroke();
  }
  cursorLine(g, c);
}

function drawInf(){
  const c = document.getElementById('inf'), g = c.getContext('2d');
  frame(g, c, 'Inference consumption per campaign, tokens/s — the storm demands 200 but its envelope pins it to 50');
  const ymax = 250;
  yAxis(g, c, ymax, v => v.toFixed(0));
  const plotH = c.height - MT - MB, y0 = c.height - MB;
  for (const name of D.campaigns){
    if (!D.inf[name].some(v => v > 0)) continue;
    g.strokeStyle = D.colors[name]; g.lineWidth = 1.6; g.beginPath();
    for (let i = 0; i <= cur; i++){
      const y = y0 - Math.min(D.inf[name][i], ymax)/ymax*plotH;
      i === 0 ? g.moveTo(X(i, c.width), y) : g.lineTo(X(i, c.width), y);
    }
    g.stroke();
  }
  cursorLine(g, c);
}

function readout(){
  const el = document.getElementById('readout');
  let html = `<div class="chip" style="border-color:#c7a34a"><div class="lbl">congestion</div>
              <div class="val">${D.mult[cur].toFixed(2)}x</div></div>`;
  for (const name of D.campaigns){
    html += `<div class="chip" style="border-color:${D.colors[name]}">
      <div class="lbl">${name}</div>
      <div class="val">${D.held[name][cur]} GPUs &middot; ${(D.tokens[name][cur]*100).toFixed(0)}% tok</div>
      <div class="lbl">${Math.round(D.credits[name][cur]).toLocaleString()} cr &middot; ${D.io[name][cur].toFixed(1)} GB/s &middot; ${Math.round(D.inf[name][cur])} tok/s</div></div>`;
  }
  el.innerHTML = html;
  document.getElementById('clock').textContent = 't = ' + hhmm(D.t[cur]);
}

function draw(){ drawOcc(); drawTok(); drawCong(); drawIO(); drawInf(); readout(); scrub.value = cur; }

const step = Math.max(1, Math.round(N / 700));
function loop(){
  if (!playing) return;
  cur = Math.min(N - 1, cur + step);
  draw();
  if (cur >= N - 1){ playing = false; document.getElementById('play').innerHTML = '&#9654; Play'; return; }
  requestAnimationFrame(loop);
}
document.getElementById('play').onclick = () => {
  playing = !playing;
  if (playing && cur >= N - 1) cur = 0;
  document.getElementById('play').innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';
  if (playing) requestAnimationFrame(loop);
};
scrub.oninput = () => { cur = +scrub.value; playing = false;
  document.getElementById('play').innerHTML = '&#9654; Play'; draw(); };
draw();
</script></body></html>
"""


def write_report(path: str, data: dict):
    with open(path, "w") as f:
        f.write(TEMPLATE.replace("__DATA__", json.dumps(data)))
