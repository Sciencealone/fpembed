(function () {
  "use strict";

  // The public PostHog write key is injected at build time and read from the
  // page head. It is never stored in this file or in the repository.
  var posthogKey = "";
  var POSTHOG_HOST = "https://eu.i.posthog.com";
  var POSTHOG_ASSET_HOST = "https://eu-assets.i.posthog.com/static/array.js";
  var UI_HOST = "https://eu.posthog.com";
  var STORAGE_KEY = "fpembed-consent";
  var BANNER_ID = "fpembed-consent";
  var FOOTER_LINK_ID = "fpembed-cookie-settings";
  var CONSENT_TEXT =
    "This site uses analytics to measure which documentation pages help readers. " +
    "Accept enables cookies. Reject keeps analytics anonymous.";

  // Current choice for this page. Null means no choice has been made.
  var choice = null;
  var banner = null;

  // Read the build-time key from the meta tag that layout.html emits.
  function readKeyFromMeta() {
    var meta = document.querySelector('meta[name="fpembed-posthog-key"]');
    if (!meta) {
      return "";
    }
    return (meta.getAttribute("content") || "").trim();
  }

  // Only a PostHog public project key is accepted.
  function hasValidKey(key) {
    return /^phc_[A-Za-z0-9]+$/.test(key);
  }

  function hasDoNotTrack() {
    return (
      navigator.doNotTrack === "1" ||
      window.doNotTrack === "1" ||
      navigator.msDoNotTrack === "1"
    );
  }

  // Storage can be unavailable in private mode or behind a policy. The page
  // still works: the choice applies until the tab is closed.
  function readChoice() {
    try {
      var value = window.localStorage.getItem(STORAGE_KEY);
      if (value === "accepted" || value === "rejected") {
        return value;
      }
    } catch (error) {
      return null;
    }
    return null;
  }

  function storeChoice(value) {
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
    } catch (error) {
      return;
    }
  }

  function clearChoice() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      return;
    }
  }

  function hasSdk() {
    return window.posthog && typeof window.posthog === "object";
  }

  function callSdk(name) {
    if (hasSdk() && typeof window.posthog[name] === "function") {
      window.posthog[name]();
    }
  }

  function applySdkState(state) {
    if (state === "accepted") {
      callSdk("opt_in_capturing");
    } else if (state === "rejected") {
      callSdk("opt_out_capturing");
    }
  }

  function buildBanner() {
    var container = document.createElement("div");
    container.id = BANNER_ID;
    container.setAttribute("role", "dialog");
    container.setAttribute("aria-labelledby", BANNER_ID + "-title");
    container.setAttribute("aria-live", "polite");
    container.tabIndex = -1;
    container.innerHTML =
      '<div class="fpembed-consent-inner">' +
      '<p id="' +
      BANNER_ID +
      '-title">' +
      CONSENT_TEXT +
      "</p>" +
      '<div class="fpembed-consent-actions">' +
      '<button type="button" data-consent="accept">Accept</button>' +
      '<button type="button" data-consent="reject">Reject</button>' +
      '<a href="privacy.html">Privacy</a>' +
      "</div>" +
      "</div>";
    return container;
  }

  function showBanner() {
    if (!banner) {
      return;
    }
    document.body.classList.add("fpembed-consent-open");
    banner.focus();
  }

  function hideBanner() {
    document.body.classList.remove("fpembed-consent-open");
  }

  function choose(state) {
    choice = state;
    storeChoice(state);
    applySdkState(state);
    hideBanner();
  }

  function withdraw() {
    choice = null;
    clearChoice();
    if (hasSdk() && typeof window.posthog.reset === "function") {
      window.posthog.reset();
    }
    callSdk("opt_out_capturing");
    showBanner();
  }

  function bindBanner() {
    var buttons = banner.querySelectorAll("[data-consent]");
    for (var index = 0; index < buttons.length; index += 1) {
      buttons[index].addEventListener("click", function (event) {
        choose(event.currentTarget.getAttribute("data-consent"));
      });
    }
  }

  function bindFooterLink() {
    var link = document.getElementById(FOOTER_LINK_ID);
    if (!link) {
      return;
    }
    link.addEventListener("click", function (event) {
      event.preventDefault();
      withdraw();
    });
  }

  // Without analytics there is no consent to manage, so the footer control is
  // hidden.
  function hideFooterLink() {
    var link = document.getElementById(FOOTER_LINK_ID);
    if (link) {
      link.hidden = true;
    }
  }

  function loadPostHog() {
    var script = document.createElement("script");
    script.async = true;
    script.src = POSTHOG_ASSET_HOST;
    script.onload = function () {
      if (!hasSdk() || typeof window.posthog.init !== "function") {
        return;
      }
      window.posthog.init(posthogKey, {
        api_host: POSTHOG_HOST,
        ui_host: UI_HOST,
        cookieless_mode: "on_reject",
        person_profiles: "never",
        autocapture: false,
        disable_session_recording: true,
        respect_dnt: true,
        capture_pageview: true,
        persistence: "localStorage+cookie",
      });
      applySdkState(choice);
    };
    // A blocked script leaves the page and the banner in place.
    script.onerror = function () {
      return;
    };
    document.head.appendChild(script);
  }

  function init() {
    posthogKey = readKeyFromMeta();

    if (!hasValidKey(posthogKey)) {
      hideFooterLink();
      return;
    }

    banner = buildBanner();
    document.body.appendChild(banner);
    bindBanner();
    bindFooterLink();

    if (hasDoNotTrack()) {
      hideBanner();
      return;
    }

    choice = readChoice();
    if (choice === "accepted" || choice === "rejected") {
      hideBanner();
    } else {
      showBanner();
    }

    loadPostHog();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
