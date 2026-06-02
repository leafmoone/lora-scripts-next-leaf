/**
 * Anima Fast plugin install panel: status polling, install trigger, guide toggle.
 * Loaded globally so SPA navigation to /lora/anima-fast works (not only full page load).
 */
(function () {
  if (window.__ANIMA_FAST_INSTALL_GUARD__) return;
  window.__ANIMA_FAST_INSTALL_GUARD__ = true;

  const CONFIRM =
    "Anima Fast 为进阶实验插件，需 NVIDIA GPU、约 16GB+ 显存，并会下载独立 Python 环境（数 GB）。\n\n确认已了解并继续安装？";

  let last = { feature_enabled: true, state: "unknown" };
  let es = null;
  let tmr = null;
  let scheduled = false;

  function q(sel) {
    return Array.from(document.querySelectorAll(sel));
  }

  function isFastPage() {
    return /^\/lora\/anima-fast(\.html|\.md)?$/.test(location.pathname);
  }

  function markPage() {
    document.body.classList.toggle("anima-fast-page", isFastPage());
  }

  function setControls(d) {
    if (!isFastPage()) return;
    const kill = !d.feature_enabled;
    const working = d.state === "installing" || d.state === "auditing";
    const ready = d.state === "ready";
    q("[data-anima-fast-install]").forEach(function (b) {
      b.disabled = kill || working;
      b.setAttribute("aria-disabled", b.disabled ? "true" : "false");
    });
    q(".right-container button").forEach(function (b) {
      const t = (b.textContent || "").trim();
      if (
        t === "开始训练" ||
        t === "✨加载训练预设✨" ||
        t === "导入配置文件" ||
        t === "保存参数" ||
        t === "Start training" ||
        t === "Load training preset" ||
        t === "Import config" ||
        t === "Save parameters"
      ) {
        b.disabled = kill || !ready;
        b.setAttribute("aria-disabled", b.disabled ? "true" : "false");
      }
    });
    document.body.classList.toggle("anima-fast-disabled", kill || !ready);
  }

  function label(d) {
    if (!d.feature_enabled) return "功能已关闭";
    if (d.state === "ready") return "插件已就绪";
    if (d.state === "installing") return "安装中";
    if (d.state === "auditing") return "审计中";
    if (d.state === "broken") return "需修复";
    if (d.state === "installed_unverified") return "待审计";
    return "进阶插件 · 待开启";
  }

  function appendLog(x) {
    const p = document.querySelector("[data-anima-fast-log]");
    if (!p) return;
    p.hidden = false;
    p.textContent += (p.textContent ? "\n" : "") + x;
    p.scrollTop = p.scrollHeight;
  }

  function apply(d) {
    last = d || last;
    setControls(last);
    const n = document.querySelector("[data-anima-fast-status]");
    if (n) n.textContent = label(last);
    const a = last.facts && last.facts.audit;
    if (a && !a.ok && a.errors) appendLog("[audit] " + a.errors.join("; "));
  }

  async function status() {
    if (!isFastPage()) return;
    try {
      const r = await fetch("/api/plugins/anima-lora/status");
      const j = await r.json();
      apply(Object.assign({ feature_enabled: true }, j.data || { state: "unknown" }));
    } catch (e) {
      const n = document.querySelector("[data-anima-fast-status]");
      if (n) n.textContent = "状态检查失败";
    }
  }

  function scheduleStatus() {
    if (!isFastPage()) return;
    if (scheduled) return;
    scheduled = true;
    setTimeout(function () {
      scheduled = false;
      markPage();
      initGuideToggle();
      ensureSidebarFastLink();
      status();
    }, 120);
  }

  function openLog(url) {
    if (!url || !window.EventSource) return;
    if (es) es.close();
    appendLog("[log] streaming " + url);
    es = new EventSource(url);
    es.onmessage = function (e) {
      try {
        const d = JSON.parse(e.data);
        if (d.text) appendLog(d.text);
        if (d.done) {
          appendLog("[log] done");
          es.close();
          es = null;
          if (tmr) {
            clearInterval(tmr);
            tmr = null;
          }
          status();
        }
      } catch (_) {
        appendLog(e.data);
      }
    };
    es.onerror = function () {
      appendLog("[log] stream disconnected");
      if (es) {
        es.close();
        es = null;
      }
      status();
    };
  }

  function initGuideToggle() {
    if (!isFastPage()) return;
    q("[data-anima-fast-guide-toggle]").forEach(function (t) {
      const p = t.closest(".anima-fast-guide-collapsible");
      const b = p && p.querySelector(".anima-fast-dataset-guide__body");
      if (!b) return;
      let o = false;
      try {
        o = localStorage.getItem("anima-fast-guide-open") === "1";
      } catch (_) {}
      b.hidden = !o;
      t.setAttribute("aria-expanded", o ? "true" : "false");
      p.classList.toggle("is-open", o);
    });
  }

  function ensureSidebarFastLink() {
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return;
    if (
      sidebar.querySelector('a[href="/lora/anima-fast.html"]') ||
      sidebar.querySelector('a[href="/lora/anima-fast.md"]')
    ) {
      return;
    }
    const finetune = sidebar.querySelector('a[href="/lora/anima-finetune.md"]');
    const anchorLi = finetune && finetune.closest("li");
    const group = anchorLi && anchorLi.parentElement;
    if (!anchorLi || !group) return;

    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "/lora/anima-fast.html";
    a.className = "sidebar-item sidebar-heading";
    a.setAttribute("aria-label", "Anima LoRA Fast");
    a.appendChild(document.createTextNode(" Anima Fast "));
    li.appendChild(a);
    anchorLi.insertAdjacentElement("afterend", li);
  }

  document.addEventListener("click", async function (e) {
    const guideBtn = e.target && e.target.closest && e.target.closest("[data-anima-fast-guide-toggle]");
    if (guideBtn && isFastPage()) {
      const p = guideBtn.closest(".anima-fast-guide-collapsible");
      const b = p && p.querySelector(".anima-fast-dataset-guide__body");
      if (b) {
        const o = b.hidden;
        b.hidden = !o;
        guideBtn.setAttribute("aria-expanded", o ? "true" : "false");
        p.classList.toggle("is-open", o);
        try {
          localStorage.setItem("anima-fast-guide-open", o ? "1" : "0");
        } catch (_) {}
      }
      return;
    }

    const installBtn = e.target && e.target.closest && e.target.closest("[data-anima-fast-install]");
    if (!installBtn || !isFastPage()) return;
    if (!last.feature_enabled) return;
    if (!window.confirm(CONFIRM)) return;

    installBtn.disabled = true;
    const statusEl = document.querySelector("[data-anima-fast-status]");
    const logEl = document.querySelector("[data-anima-fast-log]");
    if (logEl) {
      logEl.hidden = false;
      logEl.textContent = "";
    }
    if (statusEl) statusEl.textContent = "安装任务启动中";

    try {
      const r = await fetch("/api/plugins/anima-lora/install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dry_run: false }),
      });
      const j = await r.json();
      if (j.status !== "success") {
        if (statusEl) statusEl.textContent = j.message || "安装失败";
        appendLog("[error] " + (j.message || "install failed"));
        return;
      }
      const d = j.data || {};
      if (statusEl) statusEl.textContent = "安装中";
      appendLog("[task] " + (d.task_id || "unknown"));
      openLog(
        d.log_stream ||
          d.log_stream_url ||
          (d.task_id ? "/api/plugins/anima-lora/install/log/stream/" + d.task_id : "")
      );
      if (tmr) clearInterval(tmr);
      tmr = setInterval(status, 2000);
      status();
    } catch (err) {
      if (statusEl) statusEl.textContent = "安装失败";
      appendLog("[error] " + err);
    } finally {
      setTimeout(function () {
        setControls(last);
      }, 250);
    }
  });

  new MutationObserver(scheduleStatus).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  window.addEventListener("popstate", scheduleStatus);
  window.addEventListener("hashchange", scheduleStatus);

  function boot() {
    markPage();
    initGuideToggle();
    ensureSidebarFastLink();
    status();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  setTimeout(boot, 0);
})();
