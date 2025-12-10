const BIB_URL = window.MUSCLEMAP_BIB_URL;

let bibCache = null;

async function loadBibFile() {
  if (bibCache !== null) return bibCache;

  const response = await fetch(BIB_URL);
  if (!response.ok) {
    console.error("Kan BibTeX-bestand niet laden:", response.status);
    alert("Kon het BibTeX-bestand niet laden.");
    return "";
  }
  bibCache = await response.text();
  return bibCache;
}

function extractBibEntry(bibText, key) {
  const escapedKey = key.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&");
  const pattern = new RegExp(
    "@[^{]+\\{\\s*" + escapedKey + "\\s*,[\\s\\S]*?(?=\\n@|$)",
    "m"
  );
  const match = bibText.match(pattern);
  return match ? match[0].trim() + "\n" : null;
}

window.downloadBibtex = async function (key) {
  const bibText = await loadBibFile();
  if (!bibText) return;

  const entry = extractBibEntry(bibText, key);
  if (!entry) {
    alert("BibTeX entry niet gevonden voor key: " + key);
    return;
  }

  const blob = new Blob([entry], { type: "text/plain" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = key + ".bib";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};
