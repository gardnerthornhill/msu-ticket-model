/* Davis Wade Forecast: compact / deep-dive toggle, season filter, sortable table, scatter tooltip.
   Everything on the page is already in the HTML; this only shows, hides and reorders. */
(function () {
  "use strict";

  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode or blocked storage */ }
  }
  function recall(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  /* Compact vs deep dive on the big board. Mirrors ?view=deep-dive in the URL so a row can be shared expanded. */
  var toggle = document.querySelector(".view-toggle");
  if (toggle) {
    var buttons = toggle.querySelectorAll("button[data-view]");
    var dives = document.querySelectorAll(".row-dive");
    var setView = function (view, fromClick) {
      buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.view === view)); });
      dives.forEach(function (d) { d.hidden = view !== "deep-dive"; });
      if (fromClick) {
        store("view", view);
        var url = new URL(window.location.href);
        if (view === "deep-dive") { url.searchParams.set("view", "deep-dive"); } else { url.searchParams.delete("view"); }
        window.history.replaceState(null, "", url);
      }
    };
    buttons.forEach(function (b) { b.addEventListener("click", function () { setView(b.dataset.view, true); }); });
    var fromUrl = new URL(window.location.href).searchParams.get("view");
    setView(fromUrl === "deep-dive" ? "deep-dive" : (recall("view") === "deep-dive" && !fromUrl ? "deep-dive" : "compact"), false);
  }

  /* Season filter scopes the table and the scatter together. */
  var filters = document.querySelector(".filters");
  if (filters) {
    var chips = filters.querySelectorAll("button[data-season]");
    var rows = document.querySelectorAll("tr[data-season]");
    var dots = document.querySelectorAll(".scatter .dot-link");
    var setSeason = function (season) {
      chips.forEach(function (c) { c.setAttribute("aria-pressed", String(c.dataset.season === season)); });
      rows.forEach(function (r) { r.classList.toggle("row-off", season !== "all" && r.dataset.season !== season); });
      dots.forEach(function (d) {
        var label = d.querySelector(".dot-hit").dataset.label || "";
        d.classList.toggle("dot-off", season !== "all" && label.indexOf(season + " ") !== 0);
      });
    };
    chips.forEach(function (c) { c.addEventListener("click", function () { setSeason(c.dataset.season); }); });
  }

  /* Click a column heading to sort; click again to flip. */
  document.querySelectorAll("table[data-sortable]").forEach(function (table) {
    var body = table.tBodies[0];
    table.querySelectorAll("th[data-sort]").forEach(function (th, index) {
      var colIndex = Array.prototype.indexOf.call(th.parentNode.children, th);
      th.tabIndex = 0;
      var sort = function () {
        var current = th.getAttribute("aria-sort");
        var dir = current === "ascending" ? "descending" : "ascending";
        table.querySelectorAll("th[aria-sort]").forEach(function (o) { o.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", dir);
        var numeric = th.dataset.sort === "num";
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var ca = a.children[colIndex], cb = b.children[colIndex];
          var va = ca.dataset.value !== undefined ? ca.dataset.value : ca.textContent.trim();
          var vb = cb.dataset.value !== undefined ? cb.dataset.value : cb.textContent.trim();
          var r = numeric ? (parseFloat(va) - parseFloat(vb)) : va.localeCompare(vb);
          return dir === "ascending" ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      };
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
    });
  });

  /* Scatter tooltip: the <title> inside each dot already gives a native tooltip; this is the readable version. */
  var scatter = document.querySelector("[data-scatter]");
  var tip = document.querySelector(".tooltip");
  if (scatter && tip) {
    var wrap = tip.parentNode;
    var show = function (hit) {
      tip.innerHTML = "<strong></strong>Forecast <span></span>, actual <span></span> (<span></span>)";
      var parts = tip.querySelectorAll("strong, span");
      parts[0].textContent = hit.dataset.label;
      parts[1].textContent = hit.dataset.forecast;
      parts[2].textContent = hit.dataset.actual;
      parts[3].textContent = hit.dataset.error;
      var box = hit.getBoundingClientRect(), base = wrap.getBoundingClientRect();
      tip.style.left = (box.left - base.left + box.width / 2) + "px";
      tip.style.top = (box.top - base.top) + "px";
      tip.hidden = false;
    };
    var hide = function () { tip.hidden = true; };
    scatter.querySelectorAll(".dot-link").forEach(function (link) {
      var hit = link.querySelector(".dot-hit");
      link.addEventListener("mouseenter", function () { show(hit); });
      link.addEventListener("mouseleave", hide);
      link.addEventListener("focus", function () { show(hit); });
      link.addEventListener("blur", hide);
    });
  }
})();
