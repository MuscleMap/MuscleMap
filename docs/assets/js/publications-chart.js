(function () {
  function isYearHeading(el) {
    if (!el) return false;
    const t = (el.textContent || "").trim();
    return /^\d{4}$/.test(t);
  }

  function countListItemsBetween(startHeading, endHeading) {
    let count = 0;
    let node = startHeading.nextElementSibling;
    while (node && node !== endHeading) {
      if (node.querySelectorAll) count += node.querySelectorAll("li").length;
      node = node.nextElementSibling;
    }
    return count;
  }

  function buildChart(data) {
    const chart = document.getElementById("pub-year-chart");
    if (!chart) return;

    const max = Math.max(1, ...data.map((d) => d.count));
    chart.innerHTML =
      "<h4>Publications per year</h4>" +
      data
        .map((d) => {
          const pct = Math.round((d.count / max) * 100);
          return `
            <div class="pub-year-row">
              <div class="pub-year-label">${d.year}</div>
              <div class="pub-year-bar"><span style="width:${pct}%"></span></div>
              <div class="pub-year-value">${d.count}</div>
            </div>
          `;
        })
        .join("");
  }

  function findTOCElement() {
    // just-the-docs maakt vaak een nav met #toc, of een .toc
    return (
      document.querySelector("#toc") ||
      document.querySelector(".toc") ||
      document.querySelector('nav[aria-label="Table of contents"]') ||
      document.querySelector(".table-of-contents") ||
      null
    );
  }

  function wrapTOCAndChart(toc, chart) {
    if (!toc || !chart) return;
    if (toc.parentElement && toc.parentElement.classList.contains("pub-toc-wrap")) return;

    const wrap = document.createElement("div");
    wrap.className = "pub-toc-wrap";
    toc.parentNode.insertBefore(wrap, toc);
    wrap.appendChild(toc);
    wrap.appendChild(chart);
  }

  function init() {
    const chart = document.getElementById("pub-year-chart");
    if (!chart) return;

    // Pak h2/h3/h4 voor de zekerheid
    const headings = Array.from(document.querySelectorAll("h2, h3, h4")).filter(isYearHeading);

    // Als er geen headings zijn: laat *iets* zien (handig voor debug)
    if (!headings.length) {
      chart.innerHTML =
        "<h4>Publications per year</h4><div style='opacity:.7;font-size:.9rem;'>No year headings found.</div>";
      return;
    }

    const data = headings.map((h, i) => {
      const next = headings[i + 1] || null;
      return { year: (h.textContent || "").trim(), count: countListItemsBetween(h, next) };
    });

    buildChart(data);

    const toc = findTOCElement();
    wrapTOCAndChart(toc, chart);
  }

  document.addEventListener("DOMContentLoaded", function () {
    init();
    // Sommige TOC scripts renderen later, daarom nog 2 keer proberen
    setTimeout(init, 250);
    setTimeout(init, 1000);
  });
})();
