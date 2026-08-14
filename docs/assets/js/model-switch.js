/* CT/MRI-schakelaar, paginabreed.
 *
 * Elk element dat bij een modaliteit hoort krijgt data-model="ct" of "mri" en de
 * klasse is-on wanneer het aan staat.  Een klik op een knop in .mm-modelbtns zet
 * de modaliteit voor de HELE pagina: zowel de 3D-vooraanzichten als de axiale
 * overlay-GIFs schakelen dus samen, waar ze ook staan.
 *
 * De listener hangt aan het document, dus het werkt voor elk aantal schakelaars
 * en ook als een blok in de pagina tussen commentaar staat.
 */
(function () {
  function apply(want) {
    var items = document.querySelectorAll('[data-model]');
    Array.prototype.forEach.call(items, function (el) {
      var on = el.getAttribute('data-model') === want;
      el.classList.toggle('is-on', on);
      if (el.tagName === 'BUTTON') {
        el.setAttribute('aria-pressed', on ? 'true' : 'false');
      }
    });
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.mm-modelbtns button[data-model]') : null;
    if (!btn) return;
    apply(btn.getAttribute('data-model'));
  });
})();
