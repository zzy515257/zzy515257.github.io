(function () {
  'use strict';

  function findIframeBySource(win) {
    var iframes = document.querySelectorAll('iframe');
    for (var i = 0; i < iframes.length; i++) {
      try {
        if (iframes[i].contentWindow === win) {
          return iframes[i];
        }
      } catch (e) {}
    }
    return null;
  }

  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || data.type !== 'blog-widget-resize') return;
    var h = Number(data.height);
    if (!isFinite(h) || h < 200 || h > 3000) return;

    var iframe = findIframeBySource(event.source);
    if (iframe) {
      iframe.style.height = Math.ceil(h) + 'px';
    }
  });

  function pingWidgets() {
    var iframes = document.querySelectorAll('iframe.widget-meb, iframe.blog-widget');
    iframes.forEach(function (iframe) {
      try {
        if (iframe.contentWindow) {
          iframe.contentWindow.postMessage({ type: 'blog-widget-ping' }, '*');
        }
      } catch (e) {}
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', pingWidgets);
  } else {
    pingWidgets();
  }
  window.addEventListener('load', pingWidgets);
  document.addEventListener('swup:contentReplaced', function () {
    setTimeout(pingWidgets, 80);
  });
  document.addEventListener('swup:pageView', function () {
    setTimeout(pingWidgets, 80);
  });
})();
