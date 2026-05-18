/* ============================================================
   Gemma 4 Creative Reasoning — Presentation
   Vanilla JS: nav, transitions, fullscreen, SVG charts
   ============================================================ */

(() => {
  const slides   = Array.from(document.querySelectorAll('.slide'));
  const total    = slides.length;
  const progress = document.getElementById('progress');
  const curr     = document.getElementById('curr');
  const totalEl  = document.getElementById('total');
  let index = 0;

  totalEl.textContent = total;

  // Apply bleed backgrounds from data-bg
  slides.forEach(s => {
    const bg = s.dataset.bg;
    if (bg) s.style.backgroundImage = `url('${bg}')`;
  });

  function go(n) {
    n = Math.max(0, Math.min(total - 1, n));
    if (n === index) return;

    const prev = slides[index];
    const next = slides[n];

    prev.classList.remove('active');
    if (n > index) prev.classList.add('exit-prev');
    else           prev.classList.remove('exit-prev');

    next.classList.remove('exit-prev');
    // Force reflow so re-applied animation keyframes restart
    void next.offsetWidth;
    next.classList.add('active');

    index = n;
    curr.textContent = index + 1;
    progress.style.width = ((index + 1) / total * 100) + '%';

    document.body.classList.add('has-moved');
    location.hash = '#' + (index + 1);
  }

  // Boot: respect hash
  const hashIdx = parseInt(location.hash.slice(1), 10);
  index = (hashIdx && hashIdx >= 1 && hashIdx <= total) ? hashIdx - 1 : 0;
  slides[index].classList.add('active');
  curr.textContent = index + 1;
  progress.style.width = ((index + 1) / total * 100) + '%';

  // Keyboard
  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case 'ArrowRight':
      case 'PageDown':
      case ' ':
        e.preventDefault(); go(index + 1); break;
      case 'ArrowLeft':
      case 'PageUp':
      case 'Backspace':
        e.preventDefault(); go(index - 1); break;
      case 'Home':
        e.preventDefault(); go(0); break;
      case 'End':
        e.preventDefault(); go(total - 1); break;
      case 'f': case 'F':
        e.preventDefault();
        if (!document.fullscreenElement) document.documentElement.requestFullscreen();
        else document.exitFullscreen();
        break;
    }
  });

  // Click anywhere to advance (right half) / back (left half)
  // Excludes interactive chrome and chart areas
  document.addEventListener('click', e => {
    if (e.target.closest('#logo, #counter, #hint, svg, .spec, a, button, code')) return;
    if (e.clientX > window.innerWidth / 2) go(index + 1);
    else go(index - 1);
  });

  // ─────────────────────────────────────────────────────────────
  // Loss curve (real training data)
  // ─────────────────────────────────────────────────────────────
  async function renderLossChart() {
    const host = document.getElementById('loss-chart');
    if (!host) return;

    let data;
    try {
      const r = await fetch('data/loss_history.json');
      data = await r.json();
    } catch (err) {
      host.textContent = 'Failed to load training data.';
      return;
    }

    const pts = data.points;
    const totalSteps = data.total_steps;
    const minLoss = data.min_loss;
    const startLoss = data.start_loss;
    const finalTrain = data.final_train_loss;
    const spikeStep = data.spike_step;
    const spikeLoss = data.spike_loss;

    // Find min point (excluding the spike)
    let minIdx = 0, minVal = Infinity;
    pts.forEach((p, i) => {
      if (p[0] !== spikeStep && p[1] < minVal) { minVal = p[1]; minIdx = i; }
    });

    const W = 1600, H = 700;
    const pad = { t: 40, r: 80, b: 60, l: 70 };

    const xMin = 0, xMax = totalSteps;
    const yMin = 0, yMax = Math.ceil(startLoss * 10) / 10 + 0.2;

    const x = step => pad.l + (step - xMin) / (xMax - xMin) * (W - pad.l - pad.r);
    const y = loss => pad.t + (1 - (loss - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);

    // Smoothed line (light moving average)
    const smooth = pts.map((p, i) => {
      const w = 3;
      let sum = 0, n = 0;
      for (let k = Math.max(0, i - w); k <= Math.min(pts.length - 1, i + w); k++) {
        if (pts[k][0] === spikeStep) continue;
        sum += pts[k][1]; n++;
      }
      return [p[0], n ? sum / n : p[1]];
    });

    let pathD = '';
    smooth.forEach((p, i) => {
      const cmd = i === 0 ? 'M' : 'L';
      pathD += `${cmd}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)} `;
    });

    // Area under curve
    let areaD = pathD + `L${x(pts[pts.length-1][0])},${y(yMin)} L${x(pts[0][0])},${y(yMin)} Z`;

    // Y axis ticks
    const yTicks = [0, 0.5, 1.0, 1.5, 2.0, 2.5];
    const xTicks = [0, 500, 1000, 1500, 2000];

    const svg = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="lossLine" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"  stop-color="#256CC3"/>
            <stop offset="50%" stop-color="#3195F0"/>
            <stop offset="100%" stop-color="#00A9FF"/>
          </linearGradient>
          <linearGradient id="lossArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stop-color="#3195F0" stop-opacity="0.22"/>
            <stop offset="100%" stop-color="#3195F0" stop-opacity="0"/>
          </linearGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <!-- Y grid -->
        ${yTicks.map(t => `
          <line x1="${pad.l}" y1="${y(t)}" x2="${W - pad.r}" y2="${y(t)}"
                stroke="rgba(192,189,186,0.06)" stroke-width="1"/>
          <text x="${pad.l - 14}" y="${y(t) + 4}" text-anchor="end"
                fill="#5A5B62" font-family="JetBrains Mono" font-size="11"
                font-weight="300">${t.toFixed(1)}</text>
        `).join('')}

        <!-- X grid -->
        ${xTicks.map(t => `
          <line x1="${x(t)}" y1="${pad.t}" x2="${x(t)}" y2="${H - pad.b}"
                stroke="rgba(192,189,186,0.04)" stroke-width="1"/>
          <text x="${x(t)}" y="${H - pad.b + 22}" text-anchor="middle"
                fill="#5A5B62" font-family="JetBrains Mono" font-size="11"
                font-weight="300">${t}</text>
        `).join('')}

        <!-- Axis labels -->
        <text x="${pad.l}" y="${pad.t - 16}" fill="#8A8B92"
              font-family="Inter" font-size="11" font-weight="500"
              letter-spacing="2">LOSS</text>
        <text x="${W - pad.r}" y="${H - pad.b + 44}" text-anchor="end"
              fill="#8A8B92" font-family="Inter" font-size="11" font-weight="500"
              letter-spacing="2">TRAINING STEP →</text>

        <!-- Area + line -->
        <path d="${areaD}" fill="url(#lossArea)"/>
        <path d="${pathD}" fill="none" stroke="url(#lossLine)"
              stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"
              filter="url(#glow)"/>

        <!-- Start marker -->
        <g>
          <circle cx="${x(pts[0][0])}" cy="${y(startLoss)}" r="5"
                  fill="#5A5B62" stroke="#0A0A0C" stroke-width="2"/>
          <text x="${x(pts[0][0]) + 14}" y="${y(startLoss) - 4}"
                fill="#C0BDBA" font-family="Inter" font-size="13" font-weight="400">
            Start · 2.72
          </text>
        </g>

        <!-- Min marker -->
        <g>
          <circle cx="${x(pts[minIdx][0])}" cy="${y(minVal)}" r="5"
                  fill="#256CC3" stroke="#0A0A0C" stroke-width="2"/>
          <text x="${x(pts[minIdx][0])}" y="${y(minVal) + 28}"
                text-anchor="middle"
                fill="#C0BDBA" font-family="Inter" font-size="12" font-weight="400">
            min · 0.40
          </text>
        </g>

        <!-- Final marker (last real settled value, second-to-last point) -->
        <g>
          <circle cx="${x(pts[pts.length-2][0])}" cy="${y(pts[pts.length-2][1])}" r="6"
                  fill="#00A9FF" stroke="#0A0A0C" stroke-width="2"
                  filter="url(#glow)"/>
          <text x="${x(pts[pts.length-2][0]) - 14}" y="${y(pts[pts.length-2][1]) + 4}"
                text-anchor="end"
                fill="#F5F5F5" font-family="Inter" font-size="13" font-weight="500">
            Final · ${finalTrain.toFixed(2)}
          </text>
        </g>

        <!-- Spike marker -->
        <g>
          <line x1="${x(spikeStep)}" y1="${y(spikeLoss)}"
                x2="${x(spikeStep)}" y2="${y(0.5)}"
                stroke="#C94B4B" stroke-width="1.2" stroke-dasharray="3,4" opacity="0.6"/>
          <circle cx="${x(spikeStep)}" cy="${y(spikeLoss)}" r="5"
                  fill="#C94B4B" stroke="#0A0A0C" stroke-width="2"/>
          <text x="${x(spikeStep) - 14}" y="${y(spikeLoss) - 8}"
                text-anchor="end"
                fill="#C94B4B" font-family="Inter" font-size="12" font-weight="400">
            anomaly · step ${spikeStep}
          </text>
        </g>
      </svg>
    `;

    host.innerHTML = svg;
  }

  // ─────────────────────────────────────────────────────────────
  // Temperature spectrum chart
  // ─────────────────────────────────────────────────────────────
  function renderTempChart() {
    const host = document.getElementById('temp-chart');
    if (!host) return;

    const W = 1200, H = 280;
    const pad = { t: 60, r: 60, b: 70, l: 60 };
    const trackY = (H - pad.b + pad.t) / 2;

    const tMin = 0, tMax = 1.5;
    const x = t => pad.l + (t - tMin) / (tMax - tMin) * (W - pad.l - pad.r);

    const ticks = [0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5];
    const labels = {
      0.0: { txt: 'baseline', color: '#5A5B62' },
      0.7: { txt: 'sweet spot', color: '#00A9FF', big: true },
      1.5: { txt: 'noise', color: '#C94B4B' },
    };

    const sweetStart = x(0.6), sweetEnd = x(0.8);

    const svg = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="tempTrack" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stop-color="#494849"/>
            <stop offset="40%"  stop-color="#256CC3"/>
            <stop offset="55%"  stop-color="#3195F0"/>
            <stop offset="70%"  stop-color="#00A9FF"/>
            <stop offset="85%"  stop-color="#C94B4B" stop-opacity="0.6"/>
            <stop offset="100%" stop-color="#C94B4B"/>
          </linearGradient>
          <filter id="tempGlow">
            <feGaussianBlur stdDeviation="5"/>
          </filter>
        </defs>

        <!-- Track -->
        <line x1="${pad.l}" y1="${trackY}" x2="${W - pad.r}" y2="${trackY}"
              stroke="url(#tempTrack)" stroke-width="6" stroke-linecap="round"/>

        <!-- Sweet spot band -->
        <rect x="${sweetStart}" y="${trackY - 32}"
              width="${sweetEnd - sweetStart}" height="64"
              fill="rgba(0, 169, 255, 0.10)" stroke="rgba(0, 169, 255, 0.3)"
              stroke-width="1" rx="4"/>
        <text x="${(sweetStart + sweetEnd) / 2}" y="${trackY - 44}"
              text-anchor="middle" fill="#00A9FF"
              font-family="Inter" font-size="13" font-weight="500"
              letter-spacing="1">SWEET SPOT</text>
        <text x="${(sweetStart + sweetEnd) / 2}" y="${trackY + 56}"
              text-anchor="middle" fill="#F5F5F5"
              font-family="JetBrains Mono" font-size="14" font-weight="400">
          0.6&ndash;0.8
        </text>

        <!-- Ticks + labels -->
        ${ticks.map(t => {
          const lbl = labels[t];
          return `
            <line x1="${x(t)}" y1="${trackY + 14}" x2="${x(t)}" y2="${trackY + 22}"
                  stroke="${lbl ? lbl.color : '#494849'}" stroke-width="1.5"/>
            <text x="${x(t)}" y="${trackY + 42}" text-anchor="middle"
                  fill="${lbl ? lbl.color : '#5A5B62'}"
                  font-family="JetBrains Mono" font-size="${lbl && lbl.big ? '13' : '11'}"
                  font-weight="${lbl && lbl.big ? '500' : '300'}">
              ${t.toFixed(1)}
            </text>
            ${lbl && lbl.txt && !lbl.big ? `
              <text x="${x(t)}" y="${trackY - 26}" text-anchor="middle"
                    fill="${lbl.color}" font-family="Inter" font-size="11"
                    font-weight="400" letter-spacing="1.5"
                    text-transform="uppercase">${lbl.txt.toUpperCase()}</text>
            ` : ''}
          `;
        }).join('')}

        <!-- Above 1.0 warning -->
        <text x="${x(1.2)}" y="${trackY - 26}" text-anchor="middle"
              fill="#C94B4B" font-family="Inter" font-size="11"
              font-weight="400" letter-spacing="1.5">UNUSABLE</text>
      </svg>
    `;

    host.innerHTML = svg;
  }

  // Render charts after DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      renderLossChart();
      renderTempChart();
      initRecordMode();
    });
  } else {
    renderLossChart();
    renderTempChart();
    initRecordMode();
  }

  // ─────────────────────────────────────────────────────────────
  // Record mode  (?mode=record)
  // Flow:
  //  1. Click "Start Recording" → browser asks to share tab
  //  2. Audio plays per slide, presentation auto-advances
  //  3. After final slide audio ends → recording stops, MP4 downloads
  // ─────────────────────────────────────────────────────────────
  function initRecordMode() {
    // Always show on load — user can skip if they just want to browse

    // Pre-load all audio elements
    // Try .wav first (voice-cloned), fall back to .mp3 (ArX TTS)
    const audioEls = Array.from({ length: total }, (_, i) => {
      const id  = String(i + 1).padStart(2, '0');
      const wav = `audio/slide_${id}.wav`;
      const mp3 = `audio/slide_${id}.mp3`;
      const a   = new Audio();
      a.src     = wav;
      a.onerror = () => { a.src = mp3; };  // silent fallback
      a.preload = 'auto';
      return a;
    });

    let recorder  = null;
    let chunks    = [];
    let recording = false;

    // ── UI overlay ───────────────────────────────────────────────
    const overlay = document.createElement('div');
    overlay.id = 'rec-overlay';
    Object.assign(overlay.style, {
      position: 'fixed', inset: '0', zIndex: '9999',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      background: 'rgba(10,10,12,0.88)',
      fontFamily: "'Inter', sans-serif", color: '#F5F5F5',
    });
    overlay.innerHTML = `
      <div style="text-align:center;max-width:480px;padding:40px">
        <div style="font-size:11px;letter-spacing:.2em;color:#3195F0;margin-bottom:24px;text-transform:uppercase">
          Record Mode
        </div>
        <h2 style="font-size:28px;font-weight:200;margin-bottom:16px;letter-spacing:-.02em">
          Ready to record
        </h2>
        <p style="font-size:14px;color:#8A8B92;line-height:1.6;margin-bottom:40px">
          Click <b style="color:#F5F5F5">Start</b> — the browser will ask you to share this tab.<br>
          Select <em style="color:#F5F5F5">this browser tab</em> and check <em style="color:#F5F5F5">Share tab audio</em>.<br>
          Slides advance automatically as each voiceover plays.
        </p>
        <div style="display:flex;gap:16px;justify-content:center">
          <button id="rec-start" style="
            padding:14px 36px;border:none;border-radius:4px;
            background:#3195F0;color:#fff;
            font-family:inherit;font-size:15px;font-weight:500;
            cursor:pointer;letter-spacing:.02em;
          ">Start Recording</button>
          <button id="rec-skip" style="
            padding:14px 36px;border:1px solid rgba(192,189,186,0.2);border-radius:4px;
            background:transparent;color:#8A8B92;
            font-family:inherit;font-size:15px;font-weight:400;
            cursor:pointer;letter-spacing:.02em;
          ">Browse only</button>
        </div>
        <div id="rec-status" style="margin-top:24px;font-size:13px;color:#5A5B62;min-height:20px"></div>
      </div>
    `;
    document.body.appendChild(overlay);

    const startBtn  = overlay.querySelector('#rec-start');
    const skipBtn   = overlay.querySelector('#rec-skip');
    const statusEl  = overlay.querySelector('#rec-status');

    skipBtn.addEventListener('click', () => { overlay.remove(); });

    function setStatus(msg) { statusEl.textContent = msg; }

    // ── Play one slide's audio, resolve when done ─────────────────
    function playSlide(i) {
      return new Promise(resolve => {
        const audio = audioEls[i];
        if (!audio.src || audio.error) { resolve(); return; }  // no audio → instant advance
        audio.currentTime = 0;
        audio.onended  = resolve;
        audio.onerror  = resolve;  // missing file → skip
        audio.play().catch(resolve);
      });
    }

    // ── Auto-advance loop ─────────────────────────────────────────
    async function runPresentation() {
      go(0);
      for (let i = 0; i < total; i++) {
        if (i > 0) go(i);
        setStatus(`Slide ${i + 1} / ${total}`);
        await playSlide(i);
        await new Promise(r => setTimeout(r, 600));  // brief pause between slides
      }
    }

    // ── Start recording ───────────────────────────────────────────
    startBtn.addEventListener('click', async () => {
      if (recording) return;
      try {
        setStatus('Requesting screen share…');
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: { frameRate: 30 },
          audio: true,  // captures tab audio — user must check "Share tab audio"
        });

        const mimeType = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
          .find(t => MediaRecorder.isTypeSupported(t)) || 'video/webm';

        chunks   = [];
        recorder = new MediaRecorder(stream, { mimeType });
        recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: mimeType });
          const url  = URL.createObjectURL(blob);
          const a    = document.createElement('a');
          a.href = url; a.download = 'gemma4_creativity_presentation.webm';
          a.click();
          URL.revokeObjectURL(url);
          setStatus('Download started. Done!');
        };

        recorder.start(1000);  // 1-second chunks
        recording = true;

        // Hide overlay, start slides
        overlay.style.display = 'none';
        await runPresentation();

        // End recording
        recorder.stop();
        stream.getTracks().forEach(t => t.stop());

      } catch (err) {
        setStatus(`Error: ${err.message}`);
        console.error(err);
      }
    });
  }

})();
