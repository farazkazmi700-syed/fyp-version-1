'use strict';

/**
 * analytics.js — Phase 3 Analytics System frontend controller
 *
 * Responsibilities:
 *  - Fetch /api/analytics/data (FR14)
 *  - Render 7 Chart.js responsive graphs (FR20)
 *  - Populate 6 analytical tables + session insights (FR19, FR21)
 *  - Render KPI summary cards (FR21)
 *  - Display the saved analytics summary for the authenticated user (FR22)
 */

/* ============================================================
   Colour palette — dark-theme consistent with style.css tokens
   ============================================================ */
const PALETTE = {
  blue:       '#3B82F6',
  purple:     '#8B5CF6',
  green:      '#10B981',
  yellow:     '#FACC15',
  orange:     '#F97316',
  pink:       '#EC4899',
  cyan:       '#06B6D4',
  red:        '#EF4444',
  border:     '#334155',
  text:       '#F8FAFC',
  muted:      '#94A3B8',
  background: '#1E293B',
};

const TOPIC_ORDER = [
  'Machine Learning', 'Deep Learning', 'Healthcare AI', 'Power Systems', 'E-commerce AI',
];
const TOPIC_CHART_COLORS = {
  'Machine Learning': PALETTE.red,
  'Deep Learning': PALETTE.orange,
  'Healthcare AI': PALETTE.yellow,
  'Power Systems': PALETTE.green,
  'E-commerce AI': '#5081A2',
};
const TOPIC_FALLBACK_COLORS = [PALETTE.blue, PALETTE.purple, PALETTE.pink, PALETTE.cyan];

function topicLabels(topicAccuracy) {
  return TOPIC_ORDER.filter(label => topicAccuracy && Object.hasOwn(topicAccuracy, label));
}

function topicColor(label, index) {
  return TOPIC_CHART_COLORS[label]
    || TOPIC_FALLBACK_COLORS[index % TOPIC_FALLBACK_COLORS.length];
}

/* Chart.js global defaults for dark theme */
Chart.defaults.color            = PALETTE.muted;
Chart.defaults.borderColor      = PALETTE.border;
Chart.defaults.font.family      = "'Segoe UI', Arial, sans-serif";
Chart.defaults.font.size        = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 14;
Chart.defaults.plugins.legend.labels.padding  = 14;

/* ============================================================
   Utility helpers
   ============================================================ */

/** Escape HTML to prevent XSS when inserting user data into innerHTML */
function esc(val) {
  return String(val ?? '—')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Show or hide an element using the 'hidden' CSS class */
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }

/** Format ISO timestamp for display */
function formatDateTime(iso) {
  if (!iso) return '—';
  try {
    let str = String(iso).trim();
    if (!str.endsWith('Z') && !/[+-]\d{2}(:\d{2})?$/.test(str)) {
      str += 'Z';
    }
    const d = new Date(str);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleTimeString('en-GB', {
      timeZone: 'America/Los_Angeles',
      hour: '2-digit',
      minute: '2-digit',
      hourCycle: 'h23',
    });
  } catch {
    return '—';
  }
}

/** Create a responsive Chart.js chart and return the instance */
function makeChart(canvasId, type, data, options = {}) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;

  const defaults = {
    responsive:          true,
    maintainAspectRatio: false,
    plugins: {
      legend:  { position: 'bottom' },
      tooltip: { callbacks: {} },
    },
    animation: { duration: 600, easing: 'easeInOutQuart' },
    onResize: (chart) => chart.resize(),
  };

  // Deep-merge options into defaults (one level deep for plugins)
  const merged = { ...defaults, ...options };
  if (options.plugins) {
    merged.plugins = { ...defaults.plugins, ...options.plugins };
  }

  return new Chart(ctx, { type, data, options: merged });
}

/* ============================================================
   FR21: KPI card rendering
   ============================================================ */

/**
 * Render the 6 summary KPI cards at the top of the dashboard.
 * @param {Object} d  Full analytics payload from /api/analytics/data
 */
function renderKPICards(d) {
  const grid = document.getElementById('kpi-grid');
  if (!grid) return;

  const cards = [
    {
      label: 'Total Sessions',
      value: d.total_sessions,
      sub: `${d.total_messages} turns recorded`,
      accent: 'blue',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
    },
    {
      label: 'Total Responses',
      value: d.total_messages,
      sub: 'LLaMA assistant messages',
      accent: 'purple',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    },
    {
      label: 'Overall Accuracy',
      value: d.accuracy_rated_count ? `${d.overall_accuracy}%` : '—',
      sub: `${d.accuracy_rated_count} responses scored`,
      accent: 'green',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    },
    {
      label: 'Average Rating',
      value: d.avg_rating_count ? `${d.avg_rating} / 4` : '—',
      sub: `${d.avg_rating_count} ratings submitted`,
      accent: 'yellow',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
    },
    {
      label: 'Correct Replies',
      value: `${d.correct_count} / ${d.rated_count}`,
      sub: `${d.overall_correctness_pct}% correctness ratio`,
      accent: 'green',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    },
    {
      label: 'Average Latency',
      value: d.response_time_stats?.avg ? `${d.response_time_stats.avg} ms` : '—',
      sub: d.response_time_stats?.count ? `Min ${d.response_time_stats.min}ms · Max ${d.response_time_stats.max}ms` : 'No latency recorded',
      accent: 'cyan',
      icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
    },
  ];

  grid.innerHTML = cards.map(c => `
    <div class="kpi-card-modern kpi-${c.accent}">
      <div class="kpi-label-modern">${esc(c.label)}</div>
      <div class="kpi-value-modern">${esc(c.value)}</div>
    </div>
  `).join('');
}

/**
 * Render horizontal quick topic performance strip cards.
 */
function renderTopicStrip(topicAccuracy) {
  const container = document.getElementById('topic-strip');
  if (!container) return;

  const cardsHtml = TOPIC_ORDER.map((topic, idx) => {
    const data = topicAccuracy?.[topic] || { count: 0, rated: 0, accuracy: 0.0 };
    const color = topicColor(topic, idx);
    const acc = data.rated ? data.accuracy : 0;
    const badgeText = data.rated ? `${acc}% Acc` : 'No reviews';

    return `
      <div class="topic-strip-card">
        <div class="topic-strip-header">
          <div class="topic-strip-name">
            <span class="topic-strip-dot" style="background: ${color}"></span>
            <span>${esc(topic)}</span>
          </div>
          <span class="topic-strip-badge">${esc(badgeText)}</span>
        </div>
        <div class="topic-strip-body">
          <div class="topic-strip-progress-bg">
            <div class="topic-strip-progress-fill" style="width: ${data.rated ? Math.min(100, acc) : 0}%; background: ${color};"></div>
          </div>
          <div class="topic-strip-meta">
            <span>${data.count} msg${data.count !== 1 ? 's' : ''}</span>
            <span>${data.rated} rated</span>
          </div>
        </div>
      </div>
    `;
  }).join('');

  container.innerHTML = cardsHtml;
}

/* ============================================================
   FR20: Chart rendering functions
   ============================================================ */

/**
 * FR20: Topic distribution pie chart.
 * Shows the share of messages for each AI domain.
 */
function renderTopicDistChart(topicAccuracy) {
  const labels = topicLabels(topicAccuracy);
  const data   = labels.map(label => topicAccuracy[label]?.count ?? 0);
  const colors = labels.map(topicColor);
  const total  = data.reduce((sum, count) => sum + count, 0);

  makeChart('chart-topic-dist', 'pie', {
    labels,
    datasets: [{
      data,
      backgroundColor: colors,
      borderColor:     PALETTE.background,
      borderWidth:     2,
      hoverOffset:     6,
    }],
  }, {
    plugins: {
      legend: { position: 'right' },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const share = total ? (ctx.parsed / total * 100).toFixed(1) : '0.0';
            return ` ${ctx.label}: ${ctx.parsed} messages (${share}%)`;
          },
        },
      },
    },
  });
}

/**
 * FR20: Accuracy per topic bar chart.
 * Displays accuracy percentage for each classified domain.
 */
function renderTopicAccChart(topicAccuracy) {
  const labels   = topicLabels(topicAccuracy);
  const accData  = labels.map(label => topicAccuracy[label]?.rated ? topicAccuracy[label].accuracy : null);
  const colors   = labels.map(topicColor);
  const dataWithGaps = labels.flatMap((label, index) => index < labels.length - 1
    ? [accData[index], null]
    : [accData[index]]);
  const labelsWithGaps = labels.flatMap((label, index) => index < labels.length - 1
    ? [label, '']
    : [label]);
  const colorsWithGaps = colors.flatMap((color, index) => index < colors.length - 1
    ? [color, 'transparent']
    : [color]);
  if (!accData.some(value => value != null)) {
    makeChart('chart-topic-acc', 'bar', { labels, datasets: [{ data: [] }] });
    return;
  }

  makeChart('chart-topic-acc', 'bar', {
    labels: labelsWithGaps,
    datasets: [{
      label:           'Accuracy %',
      data:            dataWithGaps,
      backgroundColor: colorsWithGaps.map(c => c === 'transparent' ? 'transparent' : c + 'BB'),
      borderColor:     colorsWithGaps,
      borderWidth:     2,
      skipNull:        true,
      borderRadius:    6,
    }],
  }, {
    scales: {
      y: {
        beginAtZero: true,
        max:         100,
        ticks: { callback: v => `${v}%` },
        grid:  { color: PALETTE.border },
      },
      x: { grid: { display: false }, ticks: { autoSkip: false } },
    },
    plugins: {
      legend:  { display: false },
      tooltip: { callbacks: { label: ctx => ctx.raw == null ? '' : ` ${ctx.parsed.y}%` } },
    },
  });
}

/**
 * FR20: Accuracy vs Phase line graph.
 * Shows how accuracy changes across Start, Middle, and End phases.
 */
function renderPhaseAccChart(phaseAccuracy, phaseRated) {
  const phases = ['Start', 'Middle', 'End'];
  const data   = phases.map(p => phaseRated?.[p] ? phaseAccuracy[p] : null);
  const ratedPhases = data.filter(value => value != null).length;
  if (!ratedPhases) {
    makeChart('chart-phase-acc', 'line', { labels: phases, datasets: [{ data: [] }] });
    return;
  }

  makeChart('chart-phase-acc', 'line', {
    labels: phases,
    datasets: [{
      label:           'Accuracy %',
      data,
      spanGaps: false,
      borderColor:     PALETTE.green,
      backgroundColor: PALETTE.green + '33',
      pointBackgroundColor: PALETTE.green,
      pointRadius:     6,
      pointHoverRadius: 8,
      fill:            true,
      tension:         0.35,
    }],
  }, {
    scales: {
      y: {
        beginAtZero: true,
        max:         100,
        ticks: { callback: v => `${v}%` },
        grid:  { color: PALETTE.border },
      },
      x: { grid: { display: false } },
    },
    plugins: {
      legend:  { display: false },
      tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y}%` } },
    },
  });
}

/**
 * FR20: Correctness per reply bar chart.
 * Each bar represents one assistant response's accuracy score.
 */
function renderReplyAccChart(series) {
  // Limit to last 30 replies for readability; user can scroll the table for all.
  const display = series.slice(-30);
  const labels  = display.map(r => `#${r.index}`);
  const data    = display.map(r => r.accuracy);
  const colors  = display.map(r =>
    r.accuracy == null ? PALETTE.muted :
    r.accuracy >= 75 ? PALETTE.green :
    r.accuracy >= 40 ? PALETTE.yellow :
    PALETTE.red
  );

  makeChart('chart-reply-acc', 'bar', {
    labels,
    datasets: [{
      label:           'Accuracy %',
      data,
      backgroundColor: colors.map(c => c + 'BB'),
      borderColor:     colors,
      borderWidth:     1,
      borderRadius:    4,
      skipNull:        true,
    }],
  }, {
    scales: {
      y: {
        beginAtZero: true,
        max:         100,
        ticks: { callback: v => `${v}%` },
        grid:  { color: PALETTE.border },
      },
      x: {
        ticks: { maxRotation: 45, font: { size: 10 } },
        grid:  { display: false },
      },
    },
    plugins: {
      legend:  { display: false },
      tooltip: { callbacks: { label: ctx => ctx.raw == null ? ' Not rated' : ` ${ctx.parsed.y}%` } },
    },
  });
}

/**
 * FR20: Response time trend line graph.
 * Shows latency in milliseconds for each reply in chronological order.
 */
function renderResponseTimeChart(timeSeries) {
  const measuredCount = timeSeries.filter(value => value != null).length;
  const stride = Math.max(1, Math.ceil(timeSeries.length / 8));
  const labels = timeSeries.map((_, index) => index % stride === 0 ? `${index + 1}` : '');
  if (!measuredCount) {
    makeChart('chart-response-time', 'line', { labels, datasets: [{ data: [] }] });
    return;
  }

  makeChart('chart-response-time', 'line', {
    labels,
    datasets: [{
      label:           'Response time trend (ms)',
      data:            timeSeries,
      spanGaps:         false,
      borderColor:     PALETTE.orange,
      backgroundColor: PALETTE.orange + '22',
      pointRadius:     0,
      fill:            false,
      tension:         0,
    }],
  }, {
    scales: {
      y: {
        beginAtZero: true,
        border: { display: true, color: PALETTE.border },
        ticks: { callback: v => `${v}ms` },
        grid:  { color: PALETTE.border },
      },
      x: {
        ticks: { maxRotation: 0, font: { size: 9 }, maxTicksLimit: 8 },
        grid:  { display: false },
      },
    },
    plugins: {
      legend:  { display: false },
      tooltip: { callbacks: { label: ctx => ctx.raw == null ? '' : ` ${ctx.parsed.y} ms` } },
    },
  });
}

/**
 * FR20: Rating distribution bar chart.
 * Counts of each star rating (1–4) given as feedback.
 */
function renderRatingDistChart(ratingDist) {
  const labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars'];
  const data   = ['1', '2', '3', '4'].map(k => ratingDist[k] ?? 0);
  const colors = [PALETTE.red, PALETTE.orange, PALETTE.yellow, PALETTE.green];

  makeChart('chart-rating-dist', 'bar', {
    labels,
    datasets: [{
      label:           'Count',
      data,
      backgroundColor: colors.map(c => c + 'BB'),
      borderColor:     colors,
      borderWidth:     2,
      borderRadius:    6,
    }],
  }, {
    scales: {
      y: {
        beginAtZero: true,
        ticks: { stepSize: 1 },
        grid:  { color: PALETTE.border },
      },
      x: { grid: { display: false } },
    },
    plugins: {
      legend:  { display: false },
      tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} responses` } },
    },
  });
}

/**
 * FR20: Length preference pie chart.
 * Shows proportion of Short / To the Point / Lengthy selections.
 */
function renderLengthDistChart(lengthDist) {
  const labels = Object.keys(lengthDist);
  const data   = Object.values(lengthDist);
  const colors = [PALETTE.orange, PALETTE.yellow, PALETTE.green];

  makeChart('chart-length-dist', 'pie', {
    labels,
    datasets: [{
      data,
      backgroundColor: colors,
      borderColor:     PALETTE.background,
      borderWidth:     2,
      hoverOffset:     6,
    }],
  }, {
    plugins: {
      legend: { position: 'right' },
      tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}` } },
    },
  });
}

/* ============================================================
   FR19: Table population helpers
   ============================================================ */

/** Helper: inject rows into a <tbody> given an array of <tr> HTML strings */
function fillTable(tbodyId, rowsHtml) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = rowsHtml.length
    ? rowsHtml.join('')
    : '<tr><td colspan="99" class="an-empty">No data available yet.</td></tr>';
}

/** Return a coloured badge span for correctness values */
function correctnessBadge(val) {
  const cls = val === 'Correct'   ? 'badge-correct'
            : val === 'Partial'   ? 'badge-partial'
            : val === 'Incorrect' ? 'badge-incorrect'
            : 'badge-none';
  return `<span class="an-badge ${cls}">${esc(val)}</span>`;
}

/** Return a coloured badge span for phase values */
function phaseBadge(val) {
  const cls = val === 'Start'  ? 'badge-start'
            : val === 'Middle' ? 'badge-middle'
            : 'badge-end';
  return `<span class="an-badge ${cls}">${esc(val)}</span>`;
}

/**
 * FR19: Topic vs accuracy table
 */
function renderTopicAccTable(topicAccuracy, replySeries = []) {
  const replyCounts = replySeries.reduce((counts, reply) => {
    if (!counts[reply.topic]) counts[reply.topic] = { correct: 0, partial: 0, incorrect: 0 };
    const status = String(reply.correctness || '').toLowerCase();
    if (Object.hasOwn(counts[reply.topic], status)) counts[reply.topic][status] += 1;
    return counts;
  }, {});

  const rows = TOPIC_ORDER.map((topic, idx) => [topic, topicAccuracy?.[topic]]).filter(([, v]) => v).map(([topic, v], idx) => {
    const outcomes = replyCounts[topic] || { correct: 0, partial: 0, incorrect: 0 };
    const notRated = Math.max(0, v.count - v.rated);
    const color = topicColor(topic, idx);
    const acc = v.rated ? v.accuracy : 0;
    const badgeCls = !v.rated ? 'badge-neutral' : acc >= 70 ? 'badge-success' : acc >= 40 ? 'badge-warning' : 'badge-danger';
    return `
    <tr>
      <td>
        <div class="topic-td-name">
          <span class="topic-td-indicator" style="background: ${color};"></span>
          <strong>${esc(topic)}</strong>
        </div>
      </td>
      <td class="td-num"><span class="table-num-pill">${v.count}</span></td>
      <td class="td-num">${v.rated}</td>
      <td class="td-num td-muted">${notRated}</td>
      <td class="td-num accent-green">${outcomes.correct}</td>
      <td class="td-num accent-yellow">${outcomes.partial}</td>
      <td class="td-num accent-red">${outcomes.incorrect}</td>
      <td>
        <div class="table-acc-cell">
          <div class="table-acc-bar-wrap">
            <div class="table-acc-bar-fill" style="width: ${v.rated ? Math.min(100, acc) : 0}%; background: ${color};"></div>
          </div>
          <span class="an-badge ${badgeCls}">${v.rated ? `${acc}%` : 'Not Rated'}</span>
        </div>
      </td>
    </tr>
  `;
  });
  fillTable('tbl-topic-acc-body', rows);
}

/**
 * FR19: Reply correctness table (shows all replies with their metadata)
 */
function renderReplyCorrectnessTable(series) {
  const rows = series.map(r => `
    <tr>
      <td class="td-num"><span class="table-num-pill">#${r.index}</span></td>
      <td class="td-muted">${formatDateTime(r.timestamp)}</td>
      <td><span class="session-pill">${esc(r.session_id)}</span></td>
      <td><strong>${esc(r.topic)}</strong></td>
      <td>${phaseBadge(r.phase)}</td>
      <td>${correctnessBadge(r.correctness)}</td>
      <td class="td-num font-mono">${r.accuracy == null ? '—' : `${r.accuracy}%`}</td>
      <td class="td-num">${r.rating ? `★ ${r.rating}` : '—'}</td>
      <td class="td-muted">${esc(r.length)}</td>
    </tr>
  `);
  fillTable('tbl-reply-corr-body', rows);
}

/**
 * FR19: Phase-wise accuracy table
 */
function renderPhaseAccTable(phaseAccuracy, phaseRated, phaseCounts) {
  const phases = ['Start', 'Middle', 'End'];
  const rows = phases.map(p => {
    const acc = phaseAccuracy[p] ?? 0;
    const isRated = phaseRated?.[p] > 0;
    const badgeCls = !isRated ? 'badge-neutral' : acc >= 70 ? 'badge-success' : 'badge-warning';
    return `
    <tr>
      <td>${phaseBadge(p)}</td>
      <td class="td-num">
        <div class="table-acc-cell" style="justify-content: flex-end;">
          <span class="an-badge ${badgeCls}">${isRated ? `${acc}% (${phaseRated[p]}/${phaseCounts?.[p] ?? phaseRated[p]})` : '—'}</span>
        </div>
      </td>
    </tr>
  `;
  });
  fillTable('tbl-phase-acc-body', rows);
}

/**
 * FR19: Response time summary table
 */
function renderResponseTimeSummaryTable(stats) {
  const rows = [
    ['Total Recorded Turns', stats.count ? `${stats.count} turns` : '0'],
    ['Minimum Latency', stats.min ? `${stats.min} ms` : '—'],
    ['Maximum Latency', stats.max ? `${stats.max} ms` : '—'],
    ['Average Latency', stats.avg ? `${stats.avg} ms` : '—'],
  ].map(([label, val]) => `
    <tr>
      <td><strong>${esc(label)}</strong></td>
      <td class="td-num font-mono">${esc(String(val))}</td>
    </tr>
  `);
  fillTable('tbl-resp-time-body', rows);
}

/**
 * FR19: Rating distribution table
 */
function renderRatingDistTable(ratingDist) {
  const total = ['1', '2', '3', '4'].reduce((sum, k) => sum + (ratingDist[k] ?? 0), 0);
  const rows = ['4', '3', '2', '1'].map(k => {
    const count = ratingDist[k] ?? 0;
    const pct = total ? Math.round((count / total) * 100) : 0;
    const stars = '★'.repeat(Number(k));
    return `
    <tr>
      <td><span class="rating-stars-label">${stars} ${k} Star${Number(k) > 1 ? 's' : ''}</span></td>
      <td class="td-num font-mono"><strong>${count}</strong> <span class="td-muted">(${pct}%)</span></td>
    </tr>
  `;
  });
  fillTable('tbl-rating-dist-body', rows);
}

/**
 * FR19: Length preference table
 */
function renderLengthPrefTable(lengthDist) {
  const total = Object.values(lengthDist).reduce((sum, count) => sum + count, 0);
  const rows = Object.entries(lengthDist).map(([cat, count]) => {
    const pct = total ? Math.round((count / total) * 100) : 0;
    return `
    <tr>
      <td><strong>${esc(cat)}</strong></td>
      <td class="td-num font-mono"><strong>${count}</strong> <span class="td-muted">(${pct}%)</span></td>
    </tr>
  `;
  });
  fillTable('tbl-length-pref-body', rows);
}

/**
 * FR21: Session-level insights table
 */
function renderSessionInsightsTable(sessions) {
  const badge = document.getElementById('session-count-badge');
  if (badge) badge.textContent = `${sessions.length} Recorded`;

  const rows = sessions.map(s => {
    const accClass = s.accuracy == null ? 'badge-neutral' : s.accuracy >= 70 ? 'badge-success' : s.accuracy >= 40 ? 'badge-warning' : 'badge-danger';
    const ratingDisplay = s.avg_rating != null ? `★ ${s.avg_rating}` : '—';
    return `
    <tr>
      <td><span class="session-pill" title="${esc(s.full_id)}">${esc(s.session_id)}</span></td>
      <td class="td-muted">${formatDateTime(s.created_at)}</td>
      <td class="td-num"><span class="table-num-pill">${s.message_count}</span></td>
      <td>
        <div class="topic-tag-group">
          ${s.topics.map((t, idx) => `<span class="topic-pill" style="border-color: ${topicColor(t, idx)}33; color: ${topicColor(t, idx)}; background: ${topicColor(t, idx)}12;">${esc(t)}</span>`).join('')}
        </div>
      </td>
      <td>
        <span class="an-badge ${accClass}">${s.accuracy != null ? `${s.accuracy}% (${s.rated_count} rated)` : 'No ratings'}</span>
      </td>
      <td><span class="rating-pill">${ratingDisplay}</span></td>
    </tr>
  `;
  });
  fillTable('tbl-sessions-body', rows);
}

/* ============================================================
   scikit-learn ML Analytics Renderers
   ============================================================ */

function renderMLSupport(mlData) {
  if (!mlData) return;

  // ML KPI cards were removed from the page; nothing to render.
}

/* ============================================================
   Main initialisation — fetch data and wire everything together
   ============================================================ */

async function initAnalytics() {
  show('an-loader');

  try {
    const res  = await fetch('/api/analytics/data', { credentials: 'include' });
    const data = await res.json().catch(() => null);

    if (!res.ok || !data) {
      throw new Error((data && data.error) || `HTTP ${res.status}`);
    }

    hide('an-loader');

    // Render KPI summary cards (FR21)
    renderKPICards(data);

    // Render Topic-Based Performance Strip
    renderTopicStrip(data.topic_accuracy);

    // Render all FR20 charts
    renderTopicDistChart(data.topic_accuracy);
    renderTopicAccChart(data.topic_accuracy);
    renderPhaseAccChart(data.phase_accuracy, data.phase_rated);
    renderReplyAccChart(data.reply_accuracy_series);
    renderResponseTimeChart(data.response_time_stats.series);
    renderRatingDistChart(data.rating_distribution);
    renderLengthDistChart(data.length_distribution);

    // Populate all FR19 tables
    renderTopicAccTable(data.topic_accuracy, data.reply_accuracy_series);
    renderReplyCorrectnessTable(data.reply_accuracy_series);
    renderPhaseAccTable(data.phase_accuracy, data.phase_rated, data.phase_counts);
    renderResponseTimeSummaryTable(data.response_time_stats);
    renderRatingDistTable(data.rating_distribution);
    renderLengthPrefTable(data.length_distribution);

    // FR21: session insights table
    renderSessionInsightsTable(data.session_insights);

    // ML Support (scikit-learn)
    renderMLSupport(data.ml_support);

    // Reveal the full dashboard
    show('an-content');

  } catch (err) {
    hide('an-loader');
    const errorEl = document.getElementById('an-error');
    if (errorEl) {
      errorEl.textContent = `Failed to load analytics: ${err.message}`;
      show('an-error');
    }
  }
}

/* ============================================================
   Bootstrap
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
  // Initialise dashboard data
  initAnalytics();
});
