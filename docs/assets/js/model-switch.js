/* CT/MRI-schakelaar voor de hero-afbeelding.
 *
 * Werkt op elke .mm-modelswitch op de pagina.  Binnen zo'n blok krijgen de
 * afbeeldingen en de knoppen hetzelfde data-model ("ct" / "mri"); een klik zet
 * de klasse is-on op alles wat bij de gekozen modaliteit hoort en haalt hem
 * overal anders weg.  Staat er geen .mm-modelswitch (bijvoorbeeld omdat het
 * blok in de pagina tussen commentaar staat), dan doet dit script niets.
 */
(function () {
  var boxes = document.querySelectorAll('.mm-modelswitch');

  Array.prototype.forEach.call(boxes, function (box) {
    box.addEventListener('click', function (e) {
      var btn = e.target.closest('.mm-modelbtns button');
      if (!btn) return;
      var want = btn.getAttribute('data-model');
      var items = box.querySelectorAll('[data-model]');
      Array.prototype.forEach.call(items, function (el) {
        var on = el.getAttribute('data-model') === want;
        el.classList.toggle('is-on', on);
        if (el.tagName === 'BUTTON') el.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    });
  });
})();
