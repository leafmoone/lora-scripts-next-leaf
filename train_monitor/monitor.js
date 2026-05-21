const logBox = document.getElementById("logBox");
const followState = document.getElementById("followState");
const jumpLatest = document.getElementById("jumpLatest");
const togglePreview = document.getElementById("togglePreview");
const logDetails = document.getElementById("logDetails");
const logSummary = document.getElementById("logSummary");
let autoFollow = true;
let lastLogText = "";
let lastPreviewKey = "";
let previewEnabled = localStorage.getItem("loraMonitorPreviewEnabled") === "1";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, function(ch) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
  });
}

function isNearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 64;
}

function scrollToLatest() {
  logBox.scrollTop = logBox.scrollHeight;
}

function setAutoFollow(value) {
  autoFollow = value;
  followState.textContent = autoFollow ? "自动跟随最新" : "已暂停，正在查看历史日志";
  followState.classList.toggle("paused", !autoFollow);
  jumpLatest.classList.toggle("hidden", autoFollow);
}

logBox.addEventListener("scroll", function() {
  setAutoFollow(isNearBottom(logBox));
});

jumpLatest.addEventListener("click", function() {
  setAutoFollow(true);
  scrollToLatest();
});

togglePreview.addEventListener("click", function() {
  previewEnabled = !previewEnabled;
  localStorage.setItem("loraMonitorPreviewEnabled", previewEnabled ? "1" : "0");
  lastPreviewKey = "";
  renderPreviewToggle();
  pollStatus();
});

function renderTrainParams(status) {
  const el = document.getElementById("trainParams");
  const gpu = status.gpu_info;
  const params = status.train_params || [];
  const metrics = status.metrics || {};

  var gpuHtml = '';
  if (gpu) {
    const vramPct = gpu.vram_total_mb > 0 ? Math.round(gpu.vram_used_mb * 100 / gpu.vram_total_mb) : 0;
    const vramGB = (gpu.vram_used_mb / 1024).toFixed(1);
    const vramTotalGB = (gpu.vram_total_mb / 1024).toFixed(1);
    const loadPct = gpu.gpu_load || 0;
    const tempColor = (gpu.temperature || 0) >= 80 ? 'var(--err)' : (gpu.temperature || 0) >= 65 ? 'var(--warn)' : 'var(--ok)';
    const tempText = gpu.temperature != null ? gpu.temperature + '°C' : '-';
    const powerText = gpu.power_w != null ? gpu.power_w + 'W' + (gpu.power_limit_w ? ' / ' + gpu.power_limit_w + 'W' : '') : '';
    gpuHtml = '<div class="param-card"><div class="gpu-panel">'
      + '<div class="gpu-header"><span class="gpu-name">' + escapeHtml(gpu.name) + '</span>'
      + '<span class="gpu-temp" style="color:' + tempColor + '">' + tempText + '</span></div>'
      + '<div class="gpu-bars">'
      + '<div class="gpu-bar-row"><div class="gpu-bar-label"><span>GPU Load</span><strong>' + loadPct + '%</strong></div>'
      + '<div class="gpu-bar-track"><div class="gpu-bar-fill load" style="width:' + loadPct + '%"></div></div></div>'
      + '<div class="gpu-bar-row"><div class="gpu-bar-label"><span>VRAM</span><strong>' + vramGB + ' / ' + vramTotalGB + ' GB</strong></div>'
      + '<div class="gpu-bar-track"><div class="gpu-bar-fill vram" style="width:' + vramPct + '%"></div></div></div>'
      + '</div>'
      + (powerText ? '<div class="gpu-power">⚡ ' + powerText + '</div>' : '')
      + '</div></div>';
  } else {
    gpuHtml = '<div class="param-card"><div class="gpu-panel"><span class="muted">GPU 信息不可用</span></div></div>';
  }

  var stepsHero = params.find(function(p) { return p.label === '总步数' || p.label === '设定总步数' || p.label === '每 Epoch'; });
  var stepsHtml = '';
  if (stepsHero) {
    var parts = escapeHtml(stepsHero.value).match(/^(\d+)(.*)/);
    var mainVal = parts ? parts[1] : escapeHtml(stepsHero.value);
    var detail = parts && parts[2] ? parts[2].replace(/^[（(]\s*/, '').replace(/\s*[)）]$/, '') : '';
    stepsHtml = '<div class="param-card"><div class="steps-hero">'
      + '<div class="sh-label">' + escapeHtml(stepsHero.label) + '</div>'
      + '<div class="sh-value">' + mainVal + '</div>'
      + (detail ? '<div class="sh-detail">' + detail + '</div>' : '')
      + '</div></div>';
  } else {
    stepsHtml = '<div class="param-card"><div class="steps-hero"><div class="sh-label">总步数</div><div class="sh-value">-</div></div></div>';
  }

  var summaryItems = [];
  function addParam(label, keys) {
    for (var i = 0; i < keys.length; i++) {
      var p = params.find(function(item) { return item.label === keys[i]; });
      if (p) { summaryItems.push({label: label, value: p.value}); return; }
    }
  }
  var lr = metrics.lr;
  if (!lr) {
    var lrParam = params.find(function(p) { return p.label === '学习率' || p.label === 'UNet LR'; });
    if (lrParam) lr = lrParam.value;
  }
  if (lr) summaryItems.push({label: '学习率', value: lr});
  addParam('优化器', ['优化器']);
  addParam('调度器', ['调度器']);
  addParam('Rank', ['Rank (dim)']);
  addParam('Alpha', ['Alpha']);
  addParam('精度', ['精度']);
  addParam('分辨率', ['分辨率']);
  addParam('总 Epochs', ['总 Epochs']);
  addParam('Noise Offset', ['Noise Offset']);

  var paramHtml = '';
  if (summaryItems.length > 0) {
    paramHtml = '<div class="param-card"><div class="param-summary">'
      + '<div class="param-summary-title">训练参数</div>'
      + '<div class="param-summary-items">'
      + summaryItems.map(function(item) {
          return '<div class="ps-item"><div class="ps-label">' + escapeHtml(item.label) + '</div><div class="ps-value" title="' + escapeHtml(item.value) + '">' + escapeHtml(item.value) + '</div></div>';
        }).join('')
      + '</div></div></div>';
  } else {
    paramHtml = '<div class="param-card"><div class="param-summary"><span class="muted" style="font-size:12px;">等待训练启动后显示参数。</span></div></div>';
  }

  el.innerHTML = '<div class="param-row">' + gpuHtml + stepsHtml + paramHtml + '</div>';
}

function progressText(metrics) {
  if (!metrics || metrics.step === undefined) return "-";
  let text = String(metrics.step) + "/" + String(metrics.total_steps ?? "-");
  if (metrics.percent !== undefined) text += " (" + String(metrics.percent) + "%)";
  return text;
}

function renderCards(status) {
  const metrics = status.metrics || {};
  const cards = [
    ["模型类型", status.model_type || "-"],
    ["状态", status.state || "-"],
    ["进度", progressText(metrics)],
    ["Epoch", metrics.epoch || "-"],
    ["耗时", metrics.duration || metrics.elapsed || "-"],
    ["剩余", status.state === "训练中" ? (metrics.eta || "-") : "-"],
    ["Loss", metrics.loss || "-"],
  ];
  document.getElementById("cards").innerHTML = cards.map(function(card) {
    return '<div class="card"><div class="label">' + escapeHtml(card[0]) + '</div><div class="value">' + escapeHtml(card[1] || "-") + '</div></div>';
  }).join("");
}

function renderHero(status) {
  const metrics = status.metrics || {};
  const modelType = status.model_type || "训练";
  const state = status.state || "-";
  const pct = Number(metrics.percent || 0);
  const hasPct = Number.isFinite(pct) && pct > 0;
  const error = status.error || status.log_error || metrics.has_error;
  const attention = metrics.needs_attention || metrics.progress_stalled;
  let title = modelType + " " + state;
  let copy = "正在等待训练状态。";
  const lossText = metrics.loss ? "Loss " + metrics.loss : "";
  const trendText = metrics.loss_trend ? "，" + metrics.loss_trend : "";

  if (error) {
    title = "训练需要关注";
    copy = "检测到强错误信号，并结合任务状态判断训练可能已异常退出。";
  } else if (attention) {
    title = "训练需要关注";
    copy = metrics.progress_stalled
      ? "训练 step 已长时间未增长，请观察显存和日志。"
      : "日志中出现强错误信号，但训练仍在推进，先不自动展开日志。";
  } else if (state === "训练中") {
    copy = modelType + " 训练中" +
      (metrics.elapsed ? "，已训练 " + metrics.elapsed : "") +
      (lossText ? "，" + lossText + trendText : "") +
      (metrics.eta ? "，预计还需 " + metrics.eta + "。" : "。");
  } else if (state === "已结束") {
    copy = modelType + " 训练已完成" +
      (metrics.duration ? "，总耗时 " + metrics.duration : "") +
      (lossText ? "，" + lossText + trendText : "") + "，最新模型已保存到输出目录。";
  } else if (state === "空闲") {
    copy = "当前没有训练任务。";
  }

  document.getElementById("heroTitle").textContent = title;
  document.getElementById("heroCopy").textContent = copy;
  document.getElementById("heroPercent").textContent = hasPct ? pct.toFixed(pct % 1 === 0 ? 0 : 1) + "%" : "-";
  document.getElementById("progressFill").style.width = Math.max(0, Math.min(100, pct || 0)) + "%";
  renderSamplingProgress(metrics.sampling, status);
}

function renderSamplingProgress(sampling, status) {
  const text = document.getElementById("sampleProgressText");
  const fill = document.getElementById("sampleProgressFill");
  if (!sampling || !sampling.active) {
    fill.style.width = sampling && sampling.percent >= 100 ? "100%" : "0%";
    if (sampling && sampling.percent >= 100) {
      const atStep = sampling.train_step ? "（训练 step " + sampling.train_step + "）" : "";
      text.textContent = "预览图生成进度：最近一次采样已完成" + atStep;
    } else if (status && status.state === "训练中") {
      text.textContent = "预览图生成进度：等待下一次采样";
    } else {
      text.textContent = "预览图生成进度：暂无采样任务";
    }
    return;
  }
  const pct = Number(sampling.percent || 0);
  fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
  const atStep = sampling.train_step ? "（训练 step " + sampling.train_step + "）" : "";
  text.textContent = "预览图生成中" + atStep + "：" + sampling.step + "/" + sampling.total_steps + "（" + pct + "%）" + (sampling.eta ? "，预计 " + sampling.eta : "");
}

function fmtLoss(v) {
  if (v === null || v === undefined || v === "") return "-";
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n);
  if (abs >= 1) return n.toFixed(3);
  if (abs >= 0.01) return n.toFixed(4);
  return n.toExponential(2);
}

var tbLossCharts = {};
var tbLossLayoutKey = "";
var tbLossRange = "20p";
var tbLossManualZoom = false;
var latestStatus = null;
window.addEventListener("resize", function() {
  Object.keys(tbLossCharts).forEach(function(key) {
    tbLossCharts[key].resize();
  });
});

document.getElementById("tbLossControls").addEventListener("click", function(event) {
  const button = event.target.closest("button[data-range]");
  if (!button) return;
  tbLossRange = button.dataset.range === "latest" ? "20p" : button.dataset.range;
  tbLossManualZoom = false;
  Array.from(this.querySelectorAll("button")).forEach(function(btn) {
    btn.classList.toggle("active", btn.dataset.range === tbLossRange || (button.dataset.range === "latest" && btn.dataset.range === "20p"));
  });
  if (latestStatus) renderLossChart(latestStatus.metrics || {}, latestStatus);
});

function renderLossChart(metrics, status) {
  latestStatus = status;
  const summary = document.getElementById("lossSummary");
  const area = document.getElementById("tensorboardLossArea");
  const series = (status && status.tensorboard_loss) || [];
  const fallbackLoss = metrics && metrics.loss ? metrics.loss : "-";

  if (!series.length) {
    Object.keys(tbLossCharts).forEach(function(key) {
      tbLossCharts[key].dispose();
      delete tbLossCharts[key];
    });
    tbLossLayoutKey = "";
    summary.innerHTML = '<span>当前 Loss：<strong>' + escapeHtml(fallbackLoss) + '</strong></span>' +
      '<span class="muted">等待 TensorBoard event 写入 Loss scalar</span>';
    area.className = "tb-loss-empty";
    area.innerHTML = "等待 TensorBoard Loss 数据。训练刚启动时可能需要几十秒。";
    return;
  }

  const latestAverage = series.find(function(item) { return item.tag === "loss/average"; }) || series[0];
  summary.innerHTML = '<span>TensorBoard Loss：<strong>' + escapeHtml(fmtLoss(latestAverage.latest)) + '</strong></span>' +
    '<span class="pill">' + escapeHtml(latestAverage.tag) + '</span>' +
    '<span class="muted">真实 scalar 曲线，和 TensorBoard 同源</span>';

  area.className = "tb-loss-grid";
  const hasLearningRate = series.some(function(item) { return /^lr\//.test(item.tag); });
  const displaySeries = hasLearningRate ? series : series.concat([{
    tag: "lr",
    name: "learning rate",
    points: [],
    latest: null,
    min: null,
    run: "",
    missing: true
  }]);
  const layoutKey = displaySeries.map(function(item) { return item.tag + (item.missing ? ":missing" : ""); }).join("|");
  if (layoutKey !== tbLossLayoutKey) {
    Object.keys(tbLossCharts).forEach(function(key) {
      tbLossCharts[key].dispose();
      delete tbLossCharts[key];
    });
    area.innerHTML = displaySeries.map(function(item, idx) {
      return '<div class="tb-loss-card">' +
        '<div class="tb-loss-head">' +
        '<div><div class="tb-loss-title">' + escapeHtml(item.tag) + '</div>' +
        '<div class="muted">' + escapeHtml(item.run || "logs") + '</div></div>' +
        '<div class="tb-loss-meta" id="tbLossMeta' + idx + '">latest ' + escapeHtml(fmtLoss(item.latest)) +
        '<br>min ' + escapeHtml(fmtLoss(item.min)) + '</div>' +
        '</div>' +
        (item.missing
          ? '<div class="tb-loss-missing">暂无 learning rate scalar</div>'
          : '<div class="tb-loss-chart" id="tbLossChart' + idx + '"></div>') +
        '</div>';
    }).join("");
    tbLossLayoutKey = layoutKey;
  }

  displaySeries.forEach(function(item, idx) {
    const meta = document.getElementById("tbLossMeta" + idx);
    if (meta) meta.innerHTML = "latest " + escapeHtml(fmtLoss(item.latest)) + "<br>min " + escapeHtml(fmtLoss(item.min));
    if (item.missing) return;
    const chartDom = document.getElementById("tbLossChart" + idx);
    if (!chartDom) return;
    const chart = tbLossCharts[item.tag] || echarts.init(chartDom, null, { renderer: "canvas" });
    tbLossCharts[item.tag] = chart;
    const points = item.points || [];
    const data = points.map(function(point) {
      return [Number(point.step) || 0, Number(point.value)];
    });
    const dataZoom = [{
      type: "inside",
      xAxisIndex: 0,
      filterMode: "none",
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
      moveOnMouseWheel: false
    }];
    if (!tbLossManualZoom && tbLossRange !== "all" && data.length) {
      const latestStep = data[data.length - 1][0];
      const firstStep = data[0][0];
      const span = Math.max(1, latestStep - firstStep + 1);
      const percent = tbLossRange.endsWith("p") ? Number(tbLossRange.slice(0, -1)) : 20;
      const range = Math.max(1, Math.round(span * Math.max(1, Math.min(100, percent)) / 100));
      dataZoom[0].startValue = Math.max(firstStep, latestStep - range + 1);
      dataZoom[0].endValue = latestStep;
    }
    if (!chart.__tbLossZoomBound) {
      chart.on("dataZoom", function() {
        tbLossManualZoom = true;
      });
      chart.__tbLossZoomBound = true;
    }
    const option = {
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 46, right: 18, top: 12, bottom: 24, containLabel: false },
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(11,16,32,0.95)",
        borderColor: "#26324d",
        textStyle: { color: "#e5edf8", fontSize: 12 },
        formatter: function(params) {
          if (!params || !params.length) return "";
          var value = params[0].value || [];
          return '<strong>step ' + escapeHtml(value[0]) + '</strong><br>' +
            escapeHtml(item.tag) + ': <strong>' + escapeHtml(fmtLoss(value[1])) + '</strong>';
        }
      },
      xAxis: {
        type: "value",
        axisLine: { lineStyle: { color: "#d6d6d6", opacity: 0.35 } },
        axisTick: { show: false },
        axisLabel: { color: "#9aa7bd", fontSize: 10 },
        splitLine: { lineStyle: { color: "#2a344d", opacity: 0.55 } }
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#9aa7bd", fontSize: 10, formatter: function(v) { return fmtLoss(v); } },
        splitLine: { lineStyle: { color: "#2a344d", opacity: 0.55 } }
      },
      dataZoom: dataZoom,
      series: [{
        name: item.tag,
        type: "line",
        data: data,
        showSymbol: false,
        symbolSize: 2,
        sampling: "lttb",
        lineStyle: { color: "#16bac5", width: 1.5 },
        itemStyle: { color: "#16bac5" },
        areaStyle: { color: "rgba(22,186,197,0.08)" }
      }]
    };
    chart.setOption(option, { replaceMerge: ["dataZoom", "series"] });
  });
}

function renderResult(files) {
  const modelFiles = (files || []).filter(function(item) {
    return /\.(safetensors|ckpt|pt)$/i.test(item.name || "");
  });
  if (modelFiles.length === 0) {
    return '<div class="muted">暂无模型输出。训练完成后会在这里显示模型保存位置。</div>';
  }
  const latest = modelFiles[0];
  const history = modelFiles.slice(1, 6);
  let html = '<div class="result-card">' +
    '<div class="label">最新模型</div>' +
    '<div class="result-name">' + escapeHtml(latest.name) + '</div>' +
    '<div class="muted">' + escapeHtml(latest.size) + ' · ' + escapeHtml(latest.mtime) + '</div>' +
    '<div class="result-path">' + escapeHtml(latest.path) + '</div>' +
    '</div>';
  if (history.length > 0) {
    html += '<details class="result-history"><summary>查看其他 checkpoint</summary><ul>' +
      history.map(function(item) {
        return '<li><code>' + escapeHtml(item.name) + '</code><div class="muted">' +
          escapeHtml(item.size) + ' · ' + escapeHtml(item.mtime) + '</div></li>';
      }).join("") + '</ul></details>';
  }
  return html;
}

function renderResultDuration(metrics) {
  const el = document.getElementById("resultDuration");
  if (!el) return;
  const duration = metrics && (metrics.duration || metrics.elapsed);
  el.textContent = duration ? "本次训练耗时：" + duration : "";
}

function renderPreviewToggle() {
  togglePreview.textContent = previewEnabled ? "关闭预览图" : "开启预览图";
  togglePreview.classList.toggle("primary", !previewEnabled);
  if (!previewEnabled) {
    document.getElementById("previewArea").innerHTML = '<div class="preview-hidden"><strong>预览图未加载</strong>点击右侧"开启预览图"后，当前浏览器才会加载训练图片；关闭时不会请求图片，适合公开端口截图。</div>';
    lastPreviewKey = "";
  }
}

function previewProgressParts(item, metrics) {
  const epoch = Number(item.epoch);
  const maxEpoch = Number(item.max_epoch);
  const totalSteps = Number(metrics && metrics.total_steps);
  if (Number.isFinite(epoch)) {
    let stepText = epoch === 0 ? "Step 0" : "";
    if (Number.isFinite(maxEpoch) && maxEpoch > 0 && Number.isFinite(totalSteps) && totalSteps > 0) {
      const step = Math.round(totalSteps * epoch / maxEpoch);
      stepText = "Step " + step;
    }
    return { step: stepText || "Step -", epoch: "Epoch " + epoch };
  }
  return { step: item.role || "预览图", epoch: "Epoch -" };
}

function renderPreviews(previews, metrics) {
  const area = document.getElementById("previewArea");
  if (!previewEnabled) {
    renderPreviewToggle();
    return;
  }
  if (!previews || previews.length === 0) {
    if (lastPreviewKey !== "__empty__") {
      area.innerHTML = '<div class="muted">还没有训练预览图。通常会在第一次采样后出现在这里。</div>';
      lastPreviewKey = "__empty__";
    }
    return;
  }
  const previewKey = previews.map(function(item) {
    return item.url + "|" + item.mtime + "|" + item.size;
  }).join("||");
  if (previewKey === lastPreviewKey) return;
  lastPreviewKey = previewKey;
  area.innerHTML = '<div class="preview-grid">' + previews.map(function(item) {
    const progress = previewProgressParts(item, metrics || {});
    return '<div class="preview-card">' +
      (item.role ? '<span class="preview-role">' + escapeHtml(item.role) + '</span>' : '') +
      '<img loading="lazy" src="' + escapeHtml(item.url) + '" alt="' + escapeHtml(item.name) + '">' +
      '<div class="preview-meta">' +
      '<div class="preview-step">' + escapeHtml(progress.step) + '</div>' +
      '<div class="preview-epoch">' + escapeHtml(progress.epoch) + '</div>' +
      '<div class="preview-file" title="' + escapeHtml(item.name) + '">' + escapeHtml(item.name) + '</div></div>' +
      '</div>';
  }).join("") + '</div>';
}

function renderStatus(status) {
  document.getElementById("updatedAt").textContent = status.time || "";
  const metrics = status.metrics || {};
  const error = status.error || status.log_error || "";
  const errorBox = document.getElementById("errorBox");
  errorBox.textContent = error;
  errorBox.style.display = error ? "block" : "none";
  renderHero(status);
  renderCards(status);
  renderTrainParams(status);
  renderLossChart(metrics, status);
  renderPreviews(status.previews, metrics);
  document.getElementById("resultFiles").innerHTML = renderResult(status.outputs);
  renderResultDuration(metrics);

  const logLines = status.log_lines || [];
  const hasLogError = Boolean(error || metrics.has_error);
  const hasLogWarning = Boolean(metrics.needs_attention || metrics.progress_stalled || metrics.has_warning);
  logSummary.textContent = hasLogError
    ? "训练日志（检测到错误，已自动展开）"
    : (hasLogWarning ? "训练日志（有警告，未自动展开）" : "训练日志（正常，最近 " + logLines.length + " 行）");
  if (hasLogError) logDetails.open = true;

  const wasNearBottom = autoFollow || isNearBottom(logBox);
  const logText = logLines.slice(-180).join("\n") || "暂无训练日志。";
  if (logText !== lastLogText) {
    logBox.textContent = logText;
    lastLogText = logText;
    if (wasNearBottom) scrollToLatest();
  }
}

async function pollStatus() {
  try {
    const resp = await fetch("/api/status?ts=" + Date.now(), { cache: "no-store" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    renderStatus(await resp.json());
  } catch (err) {
    const errorBox = document.getElementById("errorBox");
    errorBox.textContent = "刷新监控状态失败: " + err;
    errorBox.style.display = "block";
  }
}

renderPreviewToggle();
pollStatus();
setInterval(pollStatus, 2000);
