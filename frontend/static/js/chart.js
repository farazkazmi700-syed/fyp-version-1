/* Modern minimal offline chart renderer for the analytics dashboard.
 * Provides high-DPI, anti-aliased, responsive drawing for:
 *   - Pie / Donut charts (Topic distribution, Length preference)
 *   - Bar charts (Accuracy per topic, Correctness per reply, Rating distribution)
 *   - Line graphs (Accuracy vs phase, Response time trend)
 */
(function (global) {
  'use strict';

  class OfflineChart {
    static defaults = {
      color: '#94A3B8',
      borderColor: 'rgba(255, 255, 255, 0.07)',
      gridColor: 'rgba(255, 255, 255, 0.04)',
      font: { family: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif', size: 11, weight: '500' },
      plugins: { legend: { labels: { boxWidth: 12, padding: 12 } } },
    };

    constructor(context, config) {
      this.ctx = context;
      this.config = config;
      this.onWindowResize = () => this.draw();
      this.resizeObserver = typeof ResizeObserver === 'undefined'
        ? null : new ResizeObserver(() => this.draw());
      this.resizeObserver?.observe(context.canvas.parentElement || context.canvas);
      window.addEventListener('resize', this.onWindowResize);
      this.draw();
    }

    resize() { this.draw(); }

    draw() {
      const canvas = this.ctx.canvas;
      const parent = canvas.parentElement;
      const width = canvas.clientWidth || parent?.clientWidth || 540;
      if (!width) return;
      const ratio = window.devicePixelRatio || 2;
      const height = canvas.clientHeight || parent?.clientHeight || 260;

      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.ctx.clearRect(0, 0, width, height);

      const dataset = this.config.data.datasets?.[0] || { data: [] };
      const labels = this.config.data.labels || [];
      const type = this.config.type;
      const chartOptions = this.config.options || {};
      const hasData = dataset.data && dataset.data.some(value => value != null);

      if (!hasData) {
        this.ctx.fillStyle = '#64748B';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.font = `500 12px ${OfflineChart.defaults.font.family}`;
        this.ctx.fillText('No recorded interactions yet', width / 2, height / 2);
        return;
      }

      if (type === 'pie') {
        return this.drawPie(width, height, labels, dataset);
      }

      this.drawGridAndAxes(width, height, chartOptions);
      if (type === 'line') {
        return this.drawLine(width, height, labels, dataset);
      }
      return this.drawBars(width, height, labels, dataset);
    }

    drawGridAndAxes(width, height, options) {
      const ctx = this.ctx;
      const axis = options.scales?.y || {};
      const min = axis.min ?? 0;
      const dataset = this.config.data.datasets?.[0] || { data: [] };
      const rawValues = dataset.data.filter(value => value != null).map(Number);
      const dataMax = rawValues.length ? Math.max(...rawValues, 1) : 1;
      const isPercent = axis.max === 100;
      const tickStep = isPercent ? 25 : axis.ticks?.stepSize || Math.max(1, Math.ceil((dataMax - min) / 4));
      const tickMax = axis.max ?? Math.max(1, Math.ceil(dataMax / tickStep) * tickStep);

      const leftMargin = 46;
      const rightMargin = width - 14;
      const topMargin = 16;
      const bottomMargin = height - 32;

      ctx.fillStyle = OfflineChart.defaults.color;
      ctx.font = `500 10px ${OfflineChart.defaults.font.family}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      // Subtle horizontal gridlines
      for (let val = min; val <= tickMax; val += tickStep) {
        const ratio = (val - min) / Math.max(tickMax - min, 1);
        const y = bottomMargin - ratio * (bottomMargin - topMargin);

        ctx.strokeStyle = OfflineChart.defaults.gridColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftMargin, y);
        ctx.lineTo(rightMargin, y);
        ctx.stroke();

        const labelText = isPercent ? `${Math.round(val)}%` : String(val);
        ctx.fillText(labelText, leftMargin - 8, y);
      }

      // X-axis baseline
      ctx.strokeStyle = OfflineChart.defaults.borderColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(leftMargin, bottomMargin);
      ctx.lineTo(rightMargin, bottomMargin);
      ctx.stroke();
    }

    drawBars(width, height, labels, dataset) {
      const ctx = this.ctx;
      const values = dataset.data.map(value => value == null ? null : Number(value));
      const axis = this.config.options?.scales?.y || {};
      const rawValues = values.filter(v => v != null);
      const max = axis.max ?? Math.max(1, ...rawValues);
      const min = axis.min ?? 0;
      const range = Math.max(max - min, 1);

      const left = 48;
      const right = width - 16;
      const top = 16;
      const bottom = height - 32;
      const plotW = right - left;
      const plotH = bottom - top;

      const validCount = values.filter(v => v != null).length;
      if (!validCount) return;

      const slotW = plotW / Math.max(validCount, 1);
      const barWidth = Math.min(Math.max(slotW * 0.52, 10), 42);

      let visibleIndex = 0;
      values.forEach((val, idx) => {
        if (val == null) return;
        const normH = Math.max(0, (val - min) / range * plotH);
        const barX = left + visibleIndex * slotW + (slotW - barWidth) / 2;
        const barY = bottom - normH;

        ctx.fillStyle = Array.isArray(dataset.backgroundColor)
          ? dataset.backgroundColor[idx]
          : (dataset.backgroundColor || '#3B82F6');

        // Rounded top bar
        this.roundRect(ctx, barX, barY, barWidth, normH, Math.min(4, barWidth / 2));
        ctx.fill();

        // X-axis label
        const lbl = labels[idx];
        if (lbl) {
          ctx.fillStyle = OfflineChart.defaults.color;
          ctx.font = `500 10px ${OfflineChart.defaults.font.family}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          const truncated = lbl.length > 12 ? lbl.slice(0, 11) + '…' : lbl;
          ctx.fillText(truncated, barX + barWidth / 2, bottom + 8);
        }

        visibleIndex += 1;
      });
    }

    drawLine(width, height, labels, dataset) {
      const ctx = this.ctx;
      const values = dataset.data.map(value => value == null ? null : Number(value));
      const axis = this.config.options?.scales?.y || {};
      const valid = values.filter(v => v != null);
      const max = axis.max ?? Math.max(1, ...valid);
      const min = axis.min ?? 0;
      const range = Math.max(max - min, 1);

      const left = 48;
      const right = width - 16;
      const top = 16;
      const bottom = height - 32;
      const plotW = right - left;
      const plotH = bottom - top;

      const indices = values.map((v, i) => v == null ? null : i).filter(i => i != null);
      if (!indices.length) return;

      const firstIdx = indices[0];
      const lastIdx = indices[indices.length - 1];
      const xCoord = idx => indices.length === 1
        ? left + plotW / 2
        : left + ((idx - firstIdx) / Math.max(lastIdx - firstIdx, 1)) * plotW;
      const yCoord = val => bottom - ((val - min) / range) * plotH;

      const strokeColor = dataset.borderColor || '#10B981';

      // 1. Draw gradient fill area
      ctx.beginPath();
      let started = false;
      let firstX = 0;
      let lastX = 0;

      indices.forEach(idx => {
        const x = xCoord(idx);
        const y = yCoord(values[idx]);
        if (!started) {
          ctx.moveTo(x, y);
          firstX = x;
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
        lastX = x;
      });

      if (started) {
        ctx.lineTo(lastX, bottom);
        ctx.lineTo(firstX, bottom);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, top, 0, bottom);
        grad.addColorStop(0, strokeColor.startsWith('#') ? strokeColor + '30' : 'rgba(59, 130, 246, 0.2)');
        grad.addColorStop(1, 'rgba(15, 23, 42, 0.0)');
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // 2. Draw line
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      started = false;

      indices.forEach(idx => {
        const x = xCoord(idx);
        const y = yCoord(values[idx]);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // 3. Draw dots
      indices.forEach(idx => {
        const x = xCoord(idx);
        const y = yCoord(values[idx]);
        ctx.fillStyle = strokeColor;
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#0F172A';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // 4. Draw X labels
      ctx.fillStyle = OfflineChart.defaults.color;
      ctx.font = `500 10px ${OfflineChart.defaults.font.family}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      const stride = Math.max(1, Math.ceil(labels.length / 8));
      labels.forEach((lbl, idx) => {
        if (lbl && idx % stride === 0) {
          const x = indices.length === 1 ? left + plotW / 2 : left + (idx / Math.max(labels.length - 1, 1)) * plotW;
          ctx.fillText(String(lbl), x, bottom + 8);
        }
      });
    }

    drawPie(width, height, labels, dataset) {
      const ctx = this.ctx;
      const values = dataset.data.map(Number);
      const total = values.reduce((acc, v) => acc + (v > 0 ? v : 0), 0);

      if (!total) {
        ctx.fillStyle = '#64748B';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `500 12px ${OfflineChart.defaults.font.family}`;
        ctx.fillText('No distribution data recorded', width / 2, height / 2);
        return;
      }

      const isWide = width >= 340;
      const radius = Math.min(width * (isWide ? 0.22 : 0.32), height * 0.38);
      const innerRadius = radius * 0.58; // Sleek modern donut cutout
      const cx = isWide ? width * 0.33 : width * 0.5;
      const cy = isWide ? height * 0.5 : height * 0.42;

      let currentAngle = -Math.PI / 2;
      values.forEach((val, idx) => {
        if (val <= 0) return;
        const sliceAngle = (val / total) * Math.PI * 2;
        const endAngle = currentAngle + sliceAngle;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, currentAngle, endAngle);
        ctx.arc(cx, cy, innerRadius, endAngle, currentAngle, true);
        ctx.closePath();

        ctx.fillStyle = dataset.backgroundColor?.[idx] || '#3B82F6';
        ctx.fill();

        // Sleek subtle segment gap
        ctx.strokeStyle = '#0F172A';
        ctx.lineWidth = 2;
        ctx.stroke();

        currentAngle = endAngle;
      });

      // Total count in donut center
      ctx.fillStyle = '#F8FAFC';
      ctx.font = `700 14px ${OfflineChart.defaults.font.family}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(total), cx, cy - 2);

      ctx.fillStyle = OfflineChart.defaults.color;
      ctx.font = `500 9px ${OfflineChart.defaults.font.family}`;
      ctx.fillText('TOTAL', cx, cy + 12);

      // Legend
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.font = `500 10.5px ${OfflineChart.defaults.font.family}`;

      labels.forEach((lbl, idx) => {
        const color = dataset.backgroundColor?.[idx] || '#3B82F6';
        const count = values[idx] || 0;
        const pct = total ? Math.round((count / total) * 100) : 0;

        if (isWide) {
          const itemY = 24 + idx * 24;
          const itemX = width * 0.62;

          ctx.fillStyle = color;
          this.roundRect(ctx, itemX, itemY - 5, 10, 10, 2);
          ctx.fill();

          ctx.fillStyle = '#F8FAFC';
          ctx.fillText(`${lbl}`, itemX + 16, itemY);

          ctx.fillStyle = OfflineChart.defaults.color;
          ctx.fillText(`${count} (${pct}%)`, itemX + 16 + ctx.measureText(lbl + ' ').width, itemY);
        } else {
          const cols = 2;
          const colIdx = idx % cols;
          const rowIdx = Math.floor(idx / cols);
          const itemX = 14 + colIdx * (width / 2);
          const itemY = height - 44 + rowIdx * 18;

          ctx.fillStyle = color;
          this.roundRect(ctx, itemX, itemY - 4, 8, 8, 2);
          ctx.fill();

          ctx.fillStyle = '#F8FAFC';
          ctx.fillText(`${lbl} (${count})`, itemX + 14, itemY);
        }
      });
    }

    roundRect(ctx, x, y, width, height, radius) {
      if (height <= 0) return;
      const r = Math.min(radius, width / 2, height / 2);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + width - r, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + r);
      ctx.lineTo(x + width, y + height);
      ctx.lineTo(x, y + height);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    destroy() {
      this.resizeObserver?.disconnect();
      window.removeEventListener('resize', this.onWindowResize);
      this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    }
  }

  global.Chart = OfflineChart;
}(window));
