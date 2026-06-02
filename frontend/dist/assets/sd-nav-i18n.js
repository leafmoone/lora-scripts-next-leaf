/**
 * Sidebar / home hub locale labels when UI is English (en-US).
 * Schema forms use vue-i18n; VuePress sidebar SSR text stays Chinese without this patch.
 */
(function () {
  const STORAGE_KEY = "sd-trainer-ui-locale";

  const ZH_TO_EN = {
    训练: "Training",
    "LoRA训练": "LoRA Training",
    "LoRA 训练": "LoRA Training",
    全量微调: "Full Finetune",
    "Anima Fast": "Anima Fast",
    工具与调试: "Tools",
    数据集打标: "Dataset Tagging",
    标签编辑: "Tag Editor",
    "LoRA 脚本工具": "LoRA Scripts",
    帮助: "Help",
    新手上路: "Getting Started",
    训练参数说明: "Training Parameters",
    其他: "More",
    "UI 设置": "UI Settings",
    关于: "About",
    更新日志: "Changelog",
    终端: "Terminal",
    训练终端: "Training Terminal",
    训练监控: "Train Monitor",
    "自动端口 · 实时日志": "Auto port · Live logs",
    "DiT · 主推": "DiT · Recommended",
    "DiT full finetune · 高显存": "DiT full finetune · High VRAM",
    "SDXL Finetune · Dreambooth": "SDXL finetune · Dreambooth",
    "SD1.5 / SDXL LoRA": "SD1.5 / SDXL LoRA",
    "Flux LoRA": "Flux LoRA",
    "下一代训练 WebUI": "Next-gen training WebUI",
    "Anima DiT 全量微调（full finetune）": "Anima DiT full finetune",
    "更新完整 DiT 权重，适合进阶玩家训练，需充足样本与高显存":
      "Updates full DiT weights; for advanced users with enough data and VRAM (~24 GB)",
    "Anima Finetune 专家模式": "Anima Finetune · Expert mode",
    "Anima LoRA 训练 专家模式": "Anima LoRA · Expert mode",
    "Anima DiT 模型 LoRA 训练 专家模式": "Anima LoRA training · Expert mode",
    "Anima DiT 训练入口，使用 Qwen3 + T5 + Anima 专用参数":
      "Anima DiT LoRA entry (Qwen3 + T5 + Anima-specific options)",
    "参数预览": "Parameter preview",
    全部重置: "Reset all",
    保存参数: "Save parameters",
    读取参数: "Load parameters",
    下载配置文件: "Download config",
    导入配置文件: "Import config",
    "✨加载训练预设✨": "Load training preset",
    开始训练: "Start training",
    终止训练: "Stop training",
    "帮助 → 新手上路": "Help → Getting started",
    "秋叶用户迁移说明": "Migration from Akiba lora-scripts",
    参数释义: "Parameter glossary",
  };

  const EN_TO_ZH = Object.fromEntries(
    Object.entries(ZH_TO_EN).map(([zh, en]) => [en, zh])
  );
  const TERMINAL_MENU_PATH = "/task.html";
  const TERMINAL_PANEL_ID = "sd-terminal-panel";
  const TERMINAL_STYLE_ID = "sd-terminal-style";

  let terminalPollTimer = null;
  let terminalInstallEs = null;
  let terminalTrainEs = null;
  let terminalInstallTaskId = "";
  let terminalTrainTaskId = "";

  function normalize(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function resolveI18nLocale() {
    try {
      const app = document.querySelector("#app")?.__vue_app__;
      const i18n = app?.config?.globalProperties?.$i18n;
      const loc = i18n?.locale;
      if (typeof loc === "string") return loc;
      if (loc && typeof loc.value === "string") return loc.value;
    } catch (e) {
      /* ignore */
    }
    return null;
  }

  function detectEnglishUI() {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored === "en-US") return true;
    if (stored === "zh-CN") return false;

    const i18nLoc = resolveI18nLocale();
    if (i18nLoc) return i18nLoc.toLowerCase().startsWith("en");

    const htmlLang = (document.documentElement.lang || "").toLowerCase();
    if (htmlLang.startsWith("en")) return true;
    if (htmlLang.startsWith("zh")) return false;

    const trainSpan = document.querySelector(
      ".el-button.el-button--primary.is-plain span, .el-button.el-button--primary span"
    );
    const trainText = normalize(trainSpan?.textContent);
    if (/^start\s*training$/i.test(trainText)) return true;
    if (trainText.includes("开始训练")) return false;

    return true;
  }

  function setNodeText(node, text) {
    if (!node || node.nodeType !== Node.TEXT_NODE) return;
    const cur = normalize(node.textContent);
    if (!cur) return;
    node.textContent = " " + text + " ";
  }

  function replaceInElement(el, map) {
    if (!el) return;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const raw = normalize(node.textContent);
      if (!raw) continue;
      if (map[raw]) {
        setNodeText(node, map[raw]);
        continue;
      }
      for (const [from, to] of Object.entries(map)) {
        if (raw.includes(from) && from.length > 2) {
          node.textContent = node.textContent.split(from).join(to);
          break;
        }
      }
    }
    el.querySelectorAll("[aria-label]").forEach((a) => {
      const label = normalize(a.getAttribute("aria-label"));
      if (map[label]) a.setAttribute("aria-label", map[label]);
    });
  }

  function isTerminalPage() {
    return /^\/task(\.html|\.md)?$/i.test(location.pathname || "");
  }

  function closeTerminalStreams() {
    if (terminalInstallEs) {
      terminalInstallEs.close();
      terminalInstallEs = null;
    }
    if (terminalTrainEs) {
      terminalTrainEs.close();
      terminalTrainEs = null;
    }
  }

  function stopTerminalPolling() {
    if (terminalPollTimer) {
      clearInterval(terminalPollTimer);
      terminalPollTimer = null;
    }
  }

  function ensureSidebarTerminalLink() {
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return;
    if (
      sidebar.querySelector('a[href="/task.html"]') ||
      sidebar.querySelector('a[href="/task.md"]')
    ) {
      return;
    }

    let othersGroup = null;
    sidebar.querySelectorAll("li").forEach((li) => {
      if (othersGroup) return;
      const heading = li.querySelector(":scope > p.sidebar-item.sidebar-heading");
      if (!heading) return;
      const text = normalize(heading.textContent);
      if (text === "其他" || text === "More") {
        othersGroup = li.querySelector(":scope > ul.sidebar-item-children");
      }
    });
    if (!othersGroup) return;

    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = TERMINAL_MENU_PATH;
    a.className = "sidebar-item";
    a.setAttribute("aria-label", "终端");
    a.appendChild(document.createTextNode(" 终端 "));
    li.appendChild(a);
    othersGroup.appendChild(li);
  }

  function ensureTerminalStyle() {
    if (document.getElementById(TERMINAL_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = TERMINAL_STYLE_ID;
    style.textContent = `
#${TERMINAL_PANEL_ID} {
  margin: 12px 24px;
  border: 1px solid var(--c-border, #3a3a3a);
  border-radius: 8px;
  background: var(--c-bg, #111);
}
#${TERMINAL_PANEL_ID} .sd-terminal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--c-border, #3a3a3a);
}
#${TERMINAL_PANEL_ID} .sd-terminal-tabs {
  display: flex;
  gap: 8px;
}
#${TERMINAL_PANEL_ID} .sd-terminal-tab {
  border: 1px solid var(--c-border, #666);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  background: transparent;
}
#${TERMINAL_PANEL_ID} .sd-terminal-tab.active {
  border-color: #7c8cff;
  color: #7c8cff;
}
#${TERMINAL_PANEL_ID} .sd-terminal-body {
  padding: 10px 12px;
}
#${TERMINAL_PANEL_ID} .sd-terminal-meta {
  opacity: 0.8;
  font-size: 12px;
  margin-bottom: 8px;
}
#${TERMINAL_PANEL_ID} .sd-terminal-pane {
  display: none;
}
#${TERMINAL_PANEL_ID} .sd-terminal-pane.active {
  display: block;
}
#${TERMINAL_PANEL_ID} pre {
  margin: 0;
  border: 1px solid var(--c-border, #3a3a3a);
  border-radius: 6px;
  background: #0b1220;
  color: #d8e6ff;
  min-height: 220px;
  max-height: 52vh;
  overflow: auto;
  padding: 10px;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.4;
}
#${TERMINAL_PANEL_ID} .sd-terminal-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}
#${TERMINAL_PANEL_ID} .sd-terminal-actions button {
  border: 1px solid var(--c-border, #666);
  border-radius: 6px;
  padding: 4px 8px;
  background: transparent;
  cursor: pointer;
}
`;
    document.head.appendChild(style);
  }

  function ensureTerminalPanel() {
    if (!isTerminalPage()) {
      stopTerminalPolling();
      closeTerminalStreams();
      return;
    }
    const host = document.querySelector(".theme-default-content > div");
    if (!host) return;
    if (!document.getElementById(TERMINAL_PANEL_ID)) {
      ensureTerminalStyle();
      const panel = document.createElement("section");
      panel.id = TERMINAL_PANEL_ID;
      panel.innerHTML = `
<div class="sd-terminal-head">
  <div>
    <strong>终端</strong>
    <span class="sd-terminal-meta" data-terminal-global-status>空闲</span>
  </div>
  <div class="sd-terminal-tabs">
    <button class="sd-terminal-tab active" data-terminal-tab="install">环境部署</button>
    <button class="sd-terminal-tab" data-terminal-tab="train">训练日志</button>
  </div>
</div>
<div class="sd-terminal-body">
  <div class="sd-terminal-pane active" data-terminal-pane="install">
    <div class="sd-terminal-actions">
      <button type="button" data-terminal-copy="install">复制日志</button>
      <button type="button" data-terminal-clear="install">清空</button>
    </div>
    <div class="sd-terminal-meta" data-terminal-install-meta>等待安装任务...</div>
    <pre data-terminal-log="install"></pre>
  </div>
  <div class="sd-terminal-pane" data-terminal-pane="train">
    <div class="sd-terminal-actions">
      <button type="button" data-terminal-copy="train">复制日志</button>
      <button type="button" data-terminal-clear="train">清空</button>
    </div>
    <div class="sd-terminal-meta" data-terminal-train-meta>等待训练任务...</div>
    <pre data-terminal-log="train"></pre>
  </div>
</div>`;
      host.appendChild(panel);
    }
    bindTerminalPanelEvents();
    if (!terminalPollTimer) {
      refreshTerminalPanel();
      terminalPollTimer = setInterval(refreshTerminalPanel, 2000);
    }
  }

  function appendTerminalLog(kind, text) {
    const pre = document.querySelector(`[data-terminal-log="${kind}"]`);
    if (!pre || !text) return;
    pre.textContent += (pre.textContent ? "\n" : "") + String(text).replace(/\r?\n$/, "");
    pre.scrollTop = pre.scrollHeight;
  }

  function setTerminalMeta(kind, text) {
    const el = document.querySelector(
      kind === "install" ? "[data-terminal-install-meta]" : "[data-terminal-train-meta]"
    );
    if (el) el.textContent = text;
  }

  async function fetchJson(url) {
    const r = await fetch(url);
    const j = await r.json();
    return j && j.data ? j.data : {};
  }

  async function fillLogTail(taskId, kind) {
    try {
      const data = await fetchJson(`/api/train/log/tail/${encodeURIComponent(taskId)}?limit=160`);
      const pre = document.querySelector(`[data-terminal-log="${kind}"]`);
      if (pre) pre.textContent = "";
      (data.lines || []).forEach((line) => appendTerminalLog(kind, line));
    } catch (_) {
      appendTerminalLog(kind, "[warn] 无法读取历史日志");
    }
  }

  async function connectTerminalStream(kind, taskId, installAlias) {
    if (!taskId) return;
    if (kind === "install" && terminalInstallTaskId === taskId && terminalInstallEs) return;
    if (kind === "train" && terminalTrainTaskId === taskId && terminalTrainEs) return;

    const streamUrl = installAlias
      ? `/api/plugins/anima-lora/install/log/stream/${encodeURIComponent(taskId)}`
      : `/api/train/log/stream/${encodeURIComponent(taskId)}`;
    await fillLogTail(taskId, kind);

    if (!window.EventSource) {
      appendTerminalLog(kind, "[warn] 当前浏览器不支持实时日志流");
      return;
    }
    if (kind === "install" && terminalInstallEs) terminalInstallEs.close();
    if (kind === "train" && terminalTrainEs) terminalTrainEs.close();

    const es = new EventSource(streamUrl);
    es.onmessage = function (e) {
      try {
        const payload = JSON.parse(e.data);
        if (payload.text) appendTerminalLog(kind, payload.text);
        if (payload.done) appendTerminalLog(kind, "[done] 任务日志流结束");
      } catch (_) {
        appendTerminalLog(kind, e.data);
      }
    };
    es.onerror = function () {
      appendTerminalLog(kind, "[warn] 日志流断开");
      es.close();
      if (kind === "install") terminalInstallEs = null;
      if (kind === "train") terminalTrainEs = null;
    };

    if (kind === "install") {
      terminalInstallTaskId = taskId;
      terminalInstallEs = es;
    } else {
      terminalTrainTaskId = taskId;
      terminalTrainEs = es;
    }
  }

  function findLatestTask(tasks, predicate) {
    const list = Array.isArray(tasks) ? tasks.slice().reverse() : [];
    return list.find(predicate) || null;
  }

  async function refreshTerminalPanel() {
    if (!isTerminalPage()) return;
    try {
      const plugin = await fetchJson("/api/plugins/anima-lora/status");
      const tasksData = await fetchJson("/api/tasks");
      const tasks = tasksData.tasks || [];

      const installTask = findLatestTask(
        tasks,
        (t) => t && t.metadata && t.metadata.kind === "anima_fast_install"
      );
      const runningTrain = findLatestTask(
        tasks,
        (t) =>
          t &&
          (!t.metadata || t.metadata.kind !== "anima_fast_install") &&
          t.status === "RUNNING"
      );
      const latestTrain = runningTrain || findLatestTask(
        tasks,
        (t) => t && (!t.metadata || t.metadata.kind !== "anima_fast_install")
      );

      const global = document.querySelector("[data-terminal-global-status]");
      if (global) {
        const g = installTask && installTask.status === "RUNNING"
          ? "环境部署中"
          : latestTrain && latestTrain.status === "RUNNING"
          ? "训练进行中"
          : plugin.state === "broken"
          ? "插件需修复"
          : "空闲";
        global.textContent = g;
      }

      if (installTask) {
        setTerminalMeta("install", `task=${installTask.id} · ${installTask.status}`);
        connectTerminalStream("install", installTask.id, true);
      } else if (plugin && plugin.facts && plugin.facts.task_id) {
        const taskId = plugin.facts.task_id;
        setTerminalMeta("install", `task=${taskId} · ${plugin.state || "unknown"}`);
        connectTerminalStream("install", taskId, true);
      } else {
        setTerminalMeta("install", `插件状态：${plugin.state || "unknown"}`);
      }

      if (latestTrain) {
        setTerminalMeta("train", `task=${latestTrain.id} · ${latestTrain.status}`);
        connectTerminalStream("train", latestTrain.id, false);
      } else {
        setTerminalMeta("train", "等待训练任务...");
      }
    } catch (err) {
      appendTerminalLog("install", `[error] 终端状态刷新失败: ${err}`);
    }
  }

  function bindTerminalPanelEvents() {
    const panel = document.getElementById(TERMINAL_PANEL_ID);
    if (!panel || panel.dataset.bound === "1") return;
    panel.dataset.bound = "1";
    panel.addEventListener("click", function (ev) {
      const tab = ev.target.closest("[data-terminal-tab]");
      if (tab) {
        const name = tab.getAttribute("data-terminal-tab");
        panel.querySelectorAll("[data-terminal-tab]").forEach((b) => {
          b.classList.toggle("active", b === tab);
        });
        panel.querySelectorAll("[data-terminal-pane]").forEach((p) => {
          p.classList.toggle("active", p.getAttribute("data-terminal-pane") === name);
        });
        return;
      }
      const clearBtn = ev.target.closest("[data-terminal-clear]");
      if (clearBtn) {
        const kind = clearBtn.getAttribute("data-terminal-clear");
        const pre = panel.querySelector(`[data-terminal-log="${kind}"]`);
        if (pre) pre.textContent = "";
        return;
      }
      const copyBtn = ev.target.closest("[data-terminal-copy]");
      if (copyBtn) {
        const kind = copyBtn.getAttribute("data-terminal-copy");
        const pre = panel.querySelector(`[data-terminal-log="${kind}"]`);
        if (pre && navigator.clipboard) {
          navigator.clipboard.writeText(pre.textContent || "");
        }
      }
    });
  }

  function applyNavLocale() {
    const english = detectEnglishUI();
    document.documentElement.dataset.sdUiLocale = english ? "en-US" : "zh-CN";

    const map = english ? ZH_TO_EN : EN_TO_ZH;
    ensureSidebarTerminalLink();
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (sidebar) replaceInElement(sidebar, map);

    const hub = document.querySelector(".sd-home-hub");
    if (hub) replaceInElement(hub, map);

    const main = document.querySelector(".right-container .theme-default-content main");
    if (main) replaceInElement(main, map);

    const rightHeader = document.querySelector(".right-container section > header");
    if (rightHeader) replaceInElement(rightHeader, map);

    const buttons = document.querySelector(".right-container .el-row");
    if (buttons) replaceInElement(buttons.closest(".right-container") || buttons, map);

    const tagline = document.querySelector(".sd-anima-finetune-tagline");
    if (tagline && english) {
      tagline.textContent = "anima-finetune — anything is possible";
    } else if (tagline && !english) {
      tagline.textContent = "anima-finetune ，一切皆有可能";
    }

    ensureTerminalPanel();
  }

  function hookLanguageToggle() {
    const bottom = document.querySelector(".sidebar-bottom");
    if (!bottom || bottom.dataset.sdNavI18nHooked) return;
    bottom.dataset.sdNavI18nHooked = "1";
    bottom.addEventListener(
      "click",
      (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        const row = btn.closest("li.appearance");
        if (!row || !/language/i.test(row.textContent || "")) return;
        const next = detectEnglishUI() ? "zh-CN" : "en-US";
        sessionStorage.setItem(STORAGE_KEY, next);
        setTimeout(applyNavLocale, 80);
        setTimeout(applyNavLocale, 400);
      },
      true
    );
  }

  let scheduled = null;
  function scheduleApply() {
    if (scheduled) clearTimeout(scheduled);
    scheduled = setTimeout(() => {
      scheduled = null;
      applyNavLocale();
      hookLanguageToggle();
      ensureTerminalPanel();
    }, 60);
  }

  function boot() {
    applyNavLocale();
    hookLanguageToggle();
    ensureTerminalPanel();

    const root = document.querySelector("#app");
    if (root) {
      new MutationObserver(scheduleApply).observe(root, {
        childList: true,
        subtree: true,
      });
    }
    window.addEventListener("hashchange", scheduleApply);
    window.addEventListener("popstate", scheduleApply);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  window.addEventListener("beforeunload", function () {
    stopTerminalPolling();
    closeTerminalStreams();
  });
})();
