(function () {
  "use strict";

  // Open the list that holds the fragment target, so an error link lands on a visible entry.
  function reveal() {
    var hash = window.location.hash.slice(1);
    var target = null;
    try {
      target = hash ? document.getElementById(decodeURIComponent(hash)) : null;
    } catch (error) {
      return;
    }
    if (!target) {
      return;
    }
    var box = target.closest("details.fpembed-errors");
    if (box) {
      box.open = true;
      target.scrollIntoView();
    }
  }

  function openAll() {
    var lists = document.querySelectorAll("details.fpembed-errors");
    for (var index = 0; index < lists.length; index += 1) {
      lists[index].open = true;
    }
  }

  window.addEventListener("hashchange", reveal);
  window.addEventListener("beforeprint", openAll);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", reveal);
  } else {
    reveal();
  }
})();
