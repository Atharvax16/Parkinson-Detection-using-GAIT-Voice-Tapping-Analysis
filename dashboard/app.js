'use strict';

const API = window.location.origin;
let gaitChart = null, shapChart = null, tapChart = null;
let lastGaitHash = null, lastTapHash = null, lastVoiceHash = null;

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    pollQR();
    setInterval(pollQR, 5000);   // re-check every 5s — picks up ngrok when it starts
    setInterval(poll, 2000);
    poll();
});

// ── QR Code ─────────────────────────────────────────────────────────────────
let lastPhoneUrl = null;

async function pollQR() {
    let phoneUrl;
    let isHttps = false;
    try {
        const r = await fetch(`${API}/api/server-info`);
        const d = await r.json();
        if (d.base_url) {
            // ngrok URL or self-signed HTTPS — already fully qualified
            phoneUrl = `${d.base_url}/phone`;
            isHttps = !!d.https;
        } else {
            const proto = d.https ? 'https' : 'http';
            const port  = d.port  ? `:${d.port}` : '';
            phoneUrl = `${proto}://${d.ip}${port}/phone`;
            isHttps  = !!d.https;
        }
    } catch {
        phoneUrl = `${window.location.origin}/phone`;
        isHttps  = window.location.protocol === 'https:';
    }

    if (phoneUrl === lastPhoneUrl) return;   // nothing changed
    lastPhoneUrl = phoneUrl;

    const qrEl = document.getElementById('qrCode');
    const urlEl = document.getElementById('qrUrl');

    // Make the URL a clickable link and colour-code http vs https
    urlEl.innerHTML = `<a href="${phoneUrl}" target="_blank"
        style="color:${isHttps ? '#198754' : '#dc3545'};text-decoration:none;word-break:break-all;">
        ${phoneUrl}</a>
        ${isHttps ? '<span style="font-size:10px;background:#d1e7dd;color:#0f5132;padding:2px 6px;border-radius:4px;margin-left:4px;">HTTPS ✓</span>'
                  : '<span style="font-size:10px;background:#f8d7da;color:#842029;padding:2px 6px;border-radius:4px;margin-left:4px;">HTTP — start ngrok!</span>'}`;

    try {
        const qr = qrcode(0, 'M');
        qr.addData(phoneUrl);
        qr.make();
        qrEl.innerHTML = qr.createImgTag(3, 6);
    } catch {
        qrEl.innerHTML = `<div style="font-size:10px;word-break:break-all;color:#6c757d">${phoneUrl}</div>`;
    }
}

// ── Polling ──────────────────────────────────────────────────────────────────
async function poll() {
    try {
        const r = await fetch(`${API}/api/results/latest`);
        const d = await r.json();
        setConnected(true);
        render(d);
    } catch {
        setConnected(false);
    }
}

function setConnected(ok) {
    const el = document.getElementById('connStatus');
    const lbl = document.getElementById('connLabel');
    el.className = 'connection-status ' + (ok ? 'connected' : 'error');
    lbl.textContent = ok ? 'Live' : 'Disconnected';
}

// ── Main render ──────────────────────────────────────────────────────────────
function render(data) {
    renderStatus('gait',    data.gait,    'gait');
    renderStatus('tapping', data.tapping, 'tapping');
    renderStatus('voice',   data.voice,   'voice');
    renderGauge(data.combined);
    renderBreakdown(data);

    if (data.gait) {
        const h = JSON.stringify([data.gait.probability_pd, Object.keys(data.gait.features || {}).length]);
        if (h !== lastGaitHash) {
            lastGaitHash = h;
            renderGaitSignal(data.gait);
            renderSHAP(data.gait);
            renderFeatureCards(data.gait);
        }
    }
    if (data.tapping) {
        const h = JSON.stringify(data.tapping.risk_score);
        if (h !== lastTapHash) {
            lastTapHash = h;
            renderTapping(data.tapping);
        }
    }
    if (data.voice) {
        const h = JSON.stringify(data.voice.probability_pd);
        if (h !== lastVoiceHash) {
            lastVoiceHash = h;
            renderVoice(data.voice);
        }
    }
}

// ── Status cards ─────────────────────────────────────────────────────────────
function renderStatus(test, result, barId) {
    const textEl  = document.getElementById(`${test}StatusText`);
    const badgeEl = document.getElementById(`${test}Badge`);
    const barEl   = document.getElementById(`${barId}Bar`);

    if (!result) {
        textEl.textContent = 'Waiting for data...';
        badgeEl.textContent = '—';
        badgeEl.className = 'status-badge waiting';
        barEl.style.width = '0%';
        barEl.style.background = '#dee2e6';
        return;
    }

    const pct  = getPct(test, result);
    const lvl  = result.risk_level || 'Low';
    const color = lvl === 'High' ? '#dc3545' : lvl === 'Medium' ? '#fd7e14' : '#198754';

    barEl.style.width = pct + '%';
    barEl.style.background = color;
    badgeEl.className = `status-badge ${lvl.toLowerCase()}`;
    badgeEl.textContent = lvl;

    if (test === 'gait')    textEl.textContent = `${lvl} Risk — ${pct.toFixed(1)}% PD probability`;
    if (test === 'tapping') textEl.textContent = `${lvl} Risk — Score ${pct.toFixed(0)}%`;
    if (test === 'voice')   textEl.textContent = `${lvl} Risk — ${pct.toFixed(1)}% PD probability`;
}

function getPct(test, result) {
    if (test === 'tapping') return (result.risk_score || 0) * 100;
    return (result.probability_pd || 0) * 100;
}

// ── Gauge ────────────────────────────────────────────────────────────────────
function renderGauge(combined) {
    const pctEl   = document.getElementById('gaugePct');
    const lblEl   = document.getElementById('gaugeLabel');
    const fillEl  = document.getElementById('gaugeFill');

    if (!combined) {
        pctEl.textContent = '--';
        lblEl.textContent = 'No tests yet';
        fillEl.setAttribute('stroke-dasharray', '0 251.2');
        return;
    }

    const score = combined.score;
    const pct   = (score * 100).toFixed(1);
    const arc   = score * 251.2;   // half-circle circumference
    const color = score < 0.35 ? '#198754' : score < 0.65 ? '#fd7e14' : '#dc3545';

    pctEl.textContent = pct + '%';
    lblEl.textContent = `${combined.risk_level} Risk`;
    fillEl.setAttribute('stroke', color);
    fillEl.setAttribute('stroke-dasharray', `${arc} 251.2`);
}

// ── Risk breakdown bars ───────────────────────────────────────────────────────
function renderBreakdown(data) {
    const el = document.getElementById('riskBreakdown');
    const items = [];

    if (data.gait)    items.push({ label: '🚶 Gait (50%)',    pct: (data.gait.probability_pd    * 100).toFixed(1) });
    if (data.tapping) items.push({ label: '👆 Tapping (20%)', pct: (data.tapping.risk_score     * 100).toFixed(1) });
    if (data.voice)   items.push({ label: '🎙 Voice (30%)',   pct: (data.voice.probability_pd   * 100).toFixed(1) });

    if (!items.length) {
        el.innerHTML = '<div class="breakdown-placeholder">Complete tests to see score breakdown</div>';
        return;
    }

    el.innerHTML = items.map(i => {
        const p = parseFloat(i.pct);
        const c = p < 35 ? '#198754' : p < 65 ? '#fd7e14' : '#dc3545';
        return `
        <div class="breakdown-item fade-in">
            <div class="breakdown-label">${i.label}</div>
            <div class="breakdown-bar-wrap">
                <div class="breakdown-bar-fill" style="width:${p}%;background:${c}"></div>
            </div>
            <div class="breakdown-pct" style="color:${c}">${i.pct}%</div>
        </div>`;
    }).join('');
}

// ── Gait signal chart ─────────────────────────────────────────────────────────
function renderGaitSignal(gait) {
    document.getElementById('gaitChartEmpty').style.display = 'none';
    document.getElementById('gaitChartBadge').textContent = gait.risk_level + ' Risk';

    const ctx = document.getElementById('gaitSignalChart').getContext('2d');
    const time = gait.raw_signal?.time || [];
    const mag  = gait.raw_signal?.magnitude || [];

    // Downsample to max 600 pts
    const step = Math.max(1, Math.floor(time.length / 600));
    const t = [], m = [];
    for (let i = 0; i < time.length; i += step) { t.push(+time[i].toFixed(2)); m.push(+mag[i].toFixed(3)); }

    if (gaitChart) gaitChart.destroy();
    gaitChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: t,
            datasets: [{
                data: m,
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13,110,253,0.08)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 600 },
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Time (s)', font: { size: 11 } }, ticks: { maxTicksLimit: 8 } },
                y: { title: { display: true, text: 'Accel (m/s²)', font: { size: 11 } } }
            }
        }
    });
}

// ── SHAP chart ────────────────────────────────────────────────────────────────
function renderSHAP(gait) {
    document.getElementById('shapChartEmpty').style.display = 'none';
    const ctx = document.getElementById('shapChart').getContext('2d');

    const entries = Object.entries(gait.shap_values || {})
        .map(([k, v]) => ({ name: prettyName(k), val: v }))
        .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
        .slice(0, 10);

    const labels = entries.map(e => e.name);
    const vals   = entries.map(e => e.val);
    const colors = vals.map(v => v > 0 ? 'rgba(220,53,69,0.82)' : 'rgba(13,110,253,0.82)');

    if (shapChart) shapChart.destroy();
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ data: vals, backgroundColor: colors, borderRadius: 5 }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 600 },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => `${c.raw > 0 ? '↑ PD' : '↓ Healthy'}: ${Math.abs(c.raw).toFixed(4)}` } }
            },
            scales: {
                x: { title: { display: true, text: 'SHAP value  (red = PD risk  |  blue = healthy)', font: { size: 11 } } },
                y: { ticks: { font: { size: 11 } } }
            }
        }
    });
}

// ── Feature cards ─────────────────────────────────────────────────────────────
const REFS = {
    stride_time_mean:  { label: 'Stride Time',         unit: 's',          lo: 0.9,  hi: 1.2  },
    stride_time_cv:    { label: 'Stride Variability',   unit: '%',          lo: 0,    hi: 3.5  },
    cadence:           { label: 'Cadence',              unit: 'steps/min',  lo: 100,  hi: 130  },
    stride_asymmetry:  { label: 'Stride Asymmetry',     unit: '',           lo: 0,    hi: 0.05 },
    stride_time_std:   { label: 'Stride Std Dev',       unit: 's',          lo: 0,    hi: 0.04 },
    stride_time_mad:   { label: 'Stride MAD',           unit: 's',          lo: 0,    hi: 0.025},
    step_regularity:   { label: 'Step Regularity',      unit: '',           lo: 0.6,  hi: 1.0  },
    stride_regularity: { label: 'Stride Regularity',    unit: '',           lo: 0.6,  hi: 1.0  },
};

function renderFeatureCards(gait) {
    const grid = document.getElementById('featureGrid');
    const sec  = document.getElementById('featuresSection');
    sec.style.display = '';
    grid.innerHTML = '';

    for (const [key, ref] of Object.entries(REFS)) {
        const val = gait.features?.[key];
        if (val === undefined || val === null || isNaN(val)) continue;

        const inRange = val >= ref.lo && val <= ref.hi;
        const overLimit = Math.abs(val - (ref.lo + ref.hi) / 2) > (ref.hi - ref.lo);
        const cls = inRange ? 'ok' : overLimit ? 'bad' : 'warn';
        const valCls = inRange ? 'ok-color' : overLimit ? 'bad-color' : 'warn-color';

        const card = document.createElement('div');
        card.className = `feature-card ${cls} fade-in`;
        card.innerHTML = `
            <div class="fc-name">${ref.label}</div>
            <div class="fc-value ${valCls}">${val.toFixed(2)}<small style="font-size:13px;font-weight:500"> ${ref.unit}</small></div>
            <div class="fc-ref">Healthy: ${ref.lo}–${ref.hi} ${ref.unit}</div>`;
        grid.appendChild(card);
    }
}

// ── Tapping results ───────────────────────────────────────────────────────────
function renderTapping(tap) {
    const row = document.getElementById('tappingVoiceRow');
    const sec = document.getElementById('tappingSection');
    row.style.display = '';
    sec.style.display = '';

    document.getElementById('tappingRiskBadge').textContent = tap.risk_level + ' Risk';

    // Tap interval chart
    const intervals = tap.tap_intervals || [];
    if (intervals.length > 0) {
        const ctx = document.getElementById('tapIntervalChart').getContext('2d');
        if (tapChart) tapChart.destroy();
        tapChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: intervals.map((_, i) => i + 1),
                datasets: [{
                    label: 'Tap interval (s)',
                    data: intervals,
                    borderColor: '#6f42c1',
                    backgroundColor: 'rgba(111,66,193,0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true,
                    tension: 0.25
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Tap #', font: { size: 11 } } },
                    y: { title: { display: true, text: 'Interval (s)', font: { size: 11 } } }
                }
            }
        });
    }

    // Metric cards
    const grid = document.getElementById('tappingFeatureGrid');
    const f = tap.features || {};
    const metrics = [
        { name: 'Total Taps',       val: f.total_taps,      unit: '',    lo: 25, hi: 999,  fmt: v => v },
        { name: 'Mean Interval',    val: f.mean_interval,   unit: 's',   lo: 0,  hi: 0.35, fmt: v => v.toFixed(3) },
        { name: 'Variability (CV)', val: f.cv_interval,     unit: '%',   lo: 0,  hi: 10,   fmt: v => v.toFixed(1) },
        { name: 'Fatigue Ratio',    val: f.fatigue_ratio,   unit: '',    lo: 0.9,hi: 1.10, fmt: v => v.toFixed(3) },
        { name: 'Rhythm Score',     val: f.rhythm_regularity,unit: '',   lo: 0.85,hi:1.0,  fmt: v => v.toFixed(3) },
    ];

    grid.innerHTML = metrics.filter(m => m.val != null).map(m => {
        const ok = m.val >= m.lo && m.val <= m.hi;
        const cls = ok ? 'ok' : 'warn';
        const vcls = ok ? 'ok-color' : 'bad-color';
        return `<div class="feature-card ${cls}">
            <div class="fc-name">${m.name}</div>
            <div class="fc-value ${vcls}">${m.fmt(m.val)}<small> ${m.unit}</small></div>
            <div class="fc-ref">Healthy: ${m.lo}–${m.hi !== 999 ? m.hi : '∞'} ${m.unit}</div>
        </div>`;
    }).join('');
}

// ── Voice results ─────────────────────────────────────────────────────────────
function renderVoice(voice) {
    const row  = document.getElementById('tappingVoiceRow');
    const sec  = document.getElementById('voiceSection');
    row.style.display = '';
    sec.style.display = '';

    const pct = (voice.probability_pd * 100).toFixed(1);
    document.getElementById('voiceRiskBadge').textContent = voice.risk_level + ' Risk';

    const color = voice.probability_pd < 0.35 ? '#198754' : voice.probability_pd < 0.65 ? '#fd7e14' : '#dc3545';

    const el = document.getElementById('voiceResult');
    const f  = voice.features || {};
    const demo = f.demo_mode;

    el.innerHTML = `
        <div style="text-align:center;padding:20px 0">
            <div style="font-size:48px;font-weight:800;color:${color}">${pct}%</div>
            <div style="font-size:14px;color:#6c757d;margin-top:4px">${voice.risk_level} PD Risk</div>
            ${demo ? '<div style="font-size:11px;color:#adb5bd;margin-top:8px">Demo mode — record voice on phone for real analysis</div>' : ''}
        </div>
        ${!demo && f['MDVP:Jitter(%)'] !== undefined ? `
        <div class="voice-metric"><span class="voice-metric-name">Jitter (%)</span><span class="voice-metric-val">${(f['MDVP:Jitter(%)']*100||0).toFixed(3)}%</span></div>
        <div class="voice-metric"><span class="voice-metric-name">Shimmer</span><span class="voice-metric-val">${(f['MDVP:Shimmer']||0).toFixed(4)}</span></div>
        <div class="voice-metric"><span class="voice-metric-name">HNR</span><span class="voice-metric-val">${(f['HNR']||0).toFixed(2)} dB</span></div>
        ` : ''}
    `;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
async function resetAll() {
    await fetch(`${API}/api/results/reset`, { method: 'POST' }).catch(() => {});
    lastGaitHash = lastTapHash = lastVoiceHash = null;
    if (gaitChart)  { gaitChart.destroy();  gaitChart  = null; }
    if (shapChart)  { shapChart.destroy();  shapChart  = null; }
    if (tapChart)   { tapChart.destroy();   tapChart   = null; }
    document.getElementById('gaitChartEmpty').style.display  = '';
    document.getElementById('shapChartEmpty').style.display  = '';
    document.getElementById('featuresSection').style.display = 'none';
    document.getElementById('tappingVoiceRow').style.display = 'none';
    document.getElementById('gaitChartBadge').textContent    = '';
    document.getElementById('riskBreakdown').innerHTML       = '<div class="breakdown-placeholder">Complete tests to see score breakdown</div>';
    renderGauge(null);
    ['gait','tapping','voice'].forEach(t => renderStatus(t, null, t === 'tapping' ? 'tapping' : t));
}

// ── Demo ──────────────────────────────────────────────────────────────────────
async function loadDemo(scenario) {
    const r = await fetch(`${API}/api/demo/${scenario}`, { method: 'POST' });
    if (!r.ok) { alert('Demo load failed'); return; }
    poll();
}

// ── CSV Upload ────────────────────────────────────────────────────────────────
async function uploadCSV(event) {
    const file = event.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${API}/api/gait/upload-csv`, { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) alert(d.error);
    else poll();
    event.target.value = '';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function prettyName(k) {
    return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
