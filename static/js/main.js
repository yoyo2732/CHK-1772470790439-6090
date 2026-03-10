/**
 * GovShield – main.js
 * Client-side utilities:
 *   • Language-aware link updating
 *   • Fraud meter animation
 *   • Admin table row highlighting
 */

(function () {
  "use strict";

  // ── Animate fraud meter bar on results page ──
  const fill = document.querySelector(".fraud-meter-fill");
  if (fill) {
    const target = fill.style.width;
    fill.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = target; }, 80);
    });
  }

  // ── Keep lang param on all internal links ──
  const lang = document.body.dataset.lang || "en";
  document.querySelectorAll("a[href]").forEach(link => {
    const href = link.getAttribute("href");
    if (href && href.startsWith("/") && !href.includes("lang=")) {
      link.href = href + (href.includes("?") ? "&" : "?") + "lang=" + lang;
    }
  });

  // ── Animate scheme cards sequentially ──
  document.querySelectorAll(".scheme-card").forEach((card, i) => {
    card.style.animationDelay = `${i * 0.07}s`;
    card.style.opacity = "0";
    card.style.animation = `fadeIn 0.4s ease ${i * 0.07}s both`;
  });

  // ── Admin: highlight flagged rows ──
  document.querySelectorAll(".table-row-high").forEach(row => {
    row.style.background = "rgba(239,68,68,0.04)";
  });

})();
