/**
 * Differential LoRA & Tag-Edit-Leaf sidebar injector.
 * Runs AFTER VuePress hydration to avoid DOM mismatch.
 */
(function () {
  function inject() {
    var sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return;

    // ── 差分 LoRA: 插入到 "LoRA训练" 下面 ──
    if (!sidebar.querySelector('a[href="/lora/differential-lora.html"]')) {
      var loraHeading = sidebar.querySelector('a[href="/lora/index.md"]');
      if (loraHeading) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = "/lora/differential-lora.html";
        a.className = "sidebar-item";
        a.setAttribute("target", "_self");
        a.setAttribute("aria-label", "Differential LoRA");
        a.appendChild(document.createTextNode(" 🔀 差分 LoRA "));
        li.appendChild(a);
        loraHeading.closest("li").after(li);
      }
    }

    // ── Tag-Edit-Leaf: 插入到 "原生标签编辑" 下面 ──
    if (!sidebar.querySelector('a[href="/tag-edit-leaf.html"]')) {
      var native = sidebar.querySelector('a[href="/native-tageditor.html"]');
      if (native) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = "/tag-edit-leaf.html";
        a.className = "sidebar-item sidebar-heading";
        a.setAttribute("target", "_self");
        a.setAttribute("aria-label", "DiffSynth Tagger");
        a.appendChild(document.createTextNode(" 🏷️ DiffSynth Tagger "));
        li.appendChild(a);
        native.closest("li").after(li);
      }
    }
  }

  // Inject after VuePress hydration (multiple tries)
  inject();
  setTimeout(inject, 200);
  setTimeout(inject, 600);
  setTimeout(inject, 1500);
})();
