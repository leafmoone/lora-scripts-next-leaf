/**
 * Version chip next to the "Next Trainer" sidebar title (reads /api/version).
 */
(function () {
  const VERSION_URL = "/api/version";
  const CHIP_ID = "sd-brand-version-chip";
  const BRAND_TITLE = "Next Trainer";
  const GAP_PX = 6;
  const OFFSET_Y_PX = 3;

  function versionFromScriptTag() {
    const el = document.querySelector('script[src*="sd-trainer-brand.js"]');
    if (!el) return null;
    try {
      const v = new URL(el.src, window.location.origin).searchParams.get("v");
      return v ? String(v).trim() : null;
    } catch (e) {
      return null;
    }
  }

  async function fetchVersion() {
    try {
      const res = await fetch(VERSION_URL);
      const json = await res.json();
      if (json && json.status === "success" && json.data && json.data.version) {
        return String(json.data.version).trim();
      }
    } catch (e) {
      /* backend offline */
    }
    return versionFromScriptTag();
  }

  function findBrandLink() {
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return null;
    return (
      sidebar.querySelector("li:first-child > a.sidebar-item.sidebar-heading[href='/']") ||
      sidebar.querySelector('a.sidebar-item.sidebar-heading[aria-label="Next Trainer"]')
    );
  }

  function measureBrandTitleRect(link) {
    const walker = document.createTreeWalker(link, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const raw = node.textContent || "";
      const idx = raw.indexOf(BRAND_TITLE);
      if (idx !== -1) {
        const range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + BRAND_TITLE.length);
        const r = range.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) return r;
      }
    }
    return link.getBoundingClientRect();
  }

  function positionChip() {
    const chip = document.getElementById(CHIP_ID);
    const link = findBrandLink();
    if (!chip || !link) {
      if (chip) chip.style.visibility = "hidden";
      return false;
    }

    const linkRect = link.getBoundingClientRect();
    const titleRect = measureBrandTitleRect(link);
    if (linkRect.width <= 0 || linkRect.height <= 0) {
      chip.style.visibility = "hidden";
      return false;
    }

    chip.style.visibility = "visible";
    const anchor = titleRect.height > 0 ? titleRect : linkRect;
    chip.style.top =
      Math.round(anchor.top + (anchor.height - chip.offsetHeight) / 2 + OFFSET_Y_PX) + "px";
    chip.style.left = Math.round(titleRect.right + GAP_PX) + "px";
    chip.style.right = "auto";
    return true;
  }

  function ensureChip(version) {
    if (!version) return;
    document.documentElement.dataset.sdTrainerVersion = version;

    let chip = document.getElementById(CHIP_ID);
    if (!chip) {
      chip = document.createElement("div");
      chip.id = CHIP_ID;
      chip.className = "sd-brand-version-chip";
      chip.setAttribute("title", "Next Trainer / SD-Trainer 版本号");
      document.body.appendChild(chip);
    }
    chip.textContent = "v" + version;
    positionChip();
  }

  let resizeTimer = null;
  function scheduleReposition() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(positionChip, 80);
  }

  async function boot() {
    const version = (await fetchVersion()) || versionFromScriptTag();
    if (!version) return;
    ensureChip(version);
    setupMobileNav();

    let tries = 0;
    const retry = setInterval(function () {
      positionChip();
      if (++tries >= 30) clearInterval(retry);
    }, 200);

    window.addEventListener("resize", scheduleReposition);
    window.addEventListener("scroll", scheduleReposition, true);
  }

  function setupMobileNav() {
    const root = document.querySelector(".theme-container.no-navbar");
    if (!root || document.querySelector(".sd-mobile-nav-toggle")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sd-mobile-nav-toggle";
    btn.setAttribute("aria-label", "打开导航菜单");
    btn.setAttribute("aria-expanded", "false");
    btn.textContent = "\u2630";
    document.body.appendChild(btn);

    const mask = root.querySelector(".sidebar-mask");

    function closeNav() {
      root.classList.remove("sidebar-open");
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-label", "打开导航菜单");
    }

    btn.addEventListener("click", function () {
      if (root.classList.contains("sidebar-open")) {
        closeNav();
        return;
      }
      root.classList.add("sidebar-open");
      btn.setAttribute("aria-expanded", "true");
      btn.setAttribute("aria-label", "关闭导航菜单");
    });

    if (mask) {
      mask.addEventListener("click", closeNav);
    }

    window.addEventListener("resize", function () {
      if (window.innerWidth > 959) {
        closeNav();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
