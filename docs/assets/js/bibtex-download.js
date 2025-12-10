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
  // escape speciale regex-tekens in de key
  const escapedKey = key.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

  const pattern = new RegExp(
    '@[^{]+\\{\\s*' + escapedKey + '\\s*,[\\s\\S]*?(?=\\n@|$)',
    'm'
  );

  const match = bibText.match(pattern);
  return match ? match[0].trim() + '\n' : null;
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
