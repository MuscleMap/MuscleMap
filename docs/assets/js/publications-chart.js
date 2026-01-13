(function () {
  function isYearHeading(el) {
    if (!el) return false;
    const t = (el.textContent || "").trim();
    return /^\d{4}$/.test(t);
  }

  function ensureIdForYearHeading(h) {
    // maak stabiele id: year-2026
    const year = (h.textContent || "").trim();
    if (!h.id) h.id = `year-${year}`;
    return h.id;
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

  function setActiveYear(year) {
    document.querySelectorAll(".pub-year-row").forEach((row) => {
      row.classList.toggle("is-active", row.dataset.year === year);
    });
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
            <div class="pub-year-row" role="link" tabindex="0" data-year="${d.year}" data-target="${d.targetId}">
              <div class="pub-year-label">${d.year}</div>
              <div class="pub-year-bar"><span style="width:${pct}%"></span></div>
              <div class="pub-year-value">${d.count}</div>
            </div>
          `;
        })
        .join("");

    // click + keyboard
    chart.querySelectorAll(".pub-year-row").forEach((row) => {
      const go = () => {
        const targetId = row.dataset.target;
        const year = row.dataset.year;
        const target = document.getElementById(targetId);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
          setActiveYear(year);
          // update URL hash (handig voor delen/back button)
          history.replaceState(null, "", `#${targetId}`);
        }
      };

      row.addEventListener("click", go);
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          go();
        }
      });
    });
  }

  function getYearHeadings() {
    // Pak h2/h3/h4 voor de zekerheid
    const headings = Array.from(document.querySelectorAll("h2, h3, h4")).filter(isYearHeading);

    // Zorg dat elk jaar een id heeft zodat we erheen kunnen scrollen
    headings.forEach(ensureIdForYearHeading);

    return headings;
  }

  function init() {
    const chart = document.getElementById("pub-year-chart");
    if (!chart) return;

    const headings = getYearHeadings();
    if (!headings.length) {
      chart.innerHTML =
        "<h4>Publications per year</h4><div style='opacity:.7;font-size:.9rem;'>No year headings found.</div>";
      return;
    }

    const data = headings.map((h, i) => {
      const next = headings[i + 1] || null;
      const year = (h.textContent || "").trim();
      const count = countListItemsBetween(h, next);
      return { year, count, targetId: h.id };
    });

    buildChart(data);

    // active state op basis van URL hash (als je direct naar #year-2024 gaat)
    const hash = (window.location.hash || "").replace("#", "");
    const activeFromHash = hash.startsWith("year-") ? hash.replace("year-", "") : null;

    if (activeFromHash) setActiveYear(activeFromHash);
    else setActiveYear(data[0].year); // default: eerste jaar

    // highlight aanpassen tijdens scroll (optional maar nice)
    const observer = new IntersectionObserver(
      (entries) => {
        // pak de meest zichtbare year heading
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible && visible.target) {
          const y = (visible.target.textContent || "").trim();
          if (/^\d{4}$/.test(y)) setActiveYear(y);
        }
      },
      { root: null, threshold: [0.25, 0.5, 0.75] }
    );

    headings.forEach((h) => observer.observe(h));
  }

  document.addEventListener("DOMContentLoaded", function () {
    init();
    // nog een paar keer: sommige themes renderen content later
    setTimeout(init, 250);
    setTimeout(init, 1000);
  });
})();