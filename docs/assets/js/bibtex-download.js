// bibtex-download.js

// Jekyll zet window.MUSCLEMAP_BIB_URL vanuit publications.md
const BIB_URL = window.MUSCLEMAP_BIB_URL || '/MuscleMap/musclemap_publications.bib';

let bibCache = null;

async function loadBibFile() {
  if (bibCache !== null) return bibCache;

  try {
    console.log('BibTeX: probeer te laden van', BIB_URL);

    // cache uitzetten zodat je altijd de nieuwste versie krijgt
    const response = await fetch(BIB_URL, { cache: 'no-store' });

    console.log('BibTeX: response status', response.status);

    // Ook bij 404 etc. lezen we de text; dan geeft de parser later netjes "entry niet gevonden"
    bibCache = await response.text();
    return bibCache;
  } catch (e) {
    console.error('Fout bij laden BibTeX-bestand:', e);
    alert('Kon het BibTeX-bestand niet laden: ' + e);
    return '';
  }
}

function extractBibEntry(bibText, key) {
  // normaliseer Windows \r\n naar \n
  const norm = bibText.replace(/\r\n/g, "\n");

  // splitsen op "\n@" (nieuw entry begint altijd met @article / @inproceedings / etc.)
  const chunks = norm.split(/\n@/);

  for (let i = 0; i < chunks.length; i++) {
    // eerste chunk begint meestal direct met "@", de rest niet
    let entry = i === 0 ? chunks[i] : "@" + chunks[i];

    // simpele check: bevat deze entry "{Key," ?
    if (entry.includes("{" + key + ",")) {
      return entry.trim() + "\n";
    }
  }

  // niets gevonden
  return null;
}

window.downloadBibtex = async function (key) {
  const bibText = await loadBibFile();
  if (!bibText) return;

  const entry = extractBibEntry(bibText, key);
  if (!entry) {
    alert('BibTeX entry niet gevonden voor key: ' + key);
    return;
  }

  const blob = new Blob([entry], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = key + '.bib';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};
