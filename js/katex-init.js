(function () {
  'use strict';

  function renderAllMath() {
    if (typeof katex === 'undefined') {
      return false;
    }

    // 1. Render block math from <script type="math/tex"> tags
    var scripts = document.querySelectorAll('script[type^="math/tex"]');
    scripts.forEach(function (el) {
      var isDisplay = el.type.indexOf('mode=display') > -1;
      var tex = (el.textContent || el.innerText || '').trim();
      if (!tex) return;
      var wrapper = document.createElement(isDisplay ? 'div' : 'span');
      if (isDisplay) {
        wrapper.className = 'katex-display my-4 text-center overflow-x-auto';
      }
      try {
        katex.render(tex, wrapper, {
          displayMode: isDisplay,
          throwOnError: false,
          strict: false,
          trust: true
        });
        el.parentNode.replaceChild(wrapper, el);
      } catch (err) {
        console.warn('KaTeX render error on script tag:', err);
      }
    });

    // 2. Render inline and block math from delimiters ($...$, $$...$$, etc.)
    if (typeof renderMathInElement === 'function') {
      var targets = document.querySelectorAll('.article-content-container, .toc-content-container');
      if (targets.length === 0) {
        var fallback = document.querySelectorAll('.post-page-container, .article-content, .markdown-body, main, body');
        if (fallback.length > 0) {
          targets = [fallback[0]];
        }
      }

      targets.forEach(function (target) {
        try {
          renderMathInElement(target, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false },
              { left: '\\(', right: '\\)', display: false },
              { left: '\\[', right: '\\]', display: true }
            ],
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true
          });
        } catch (err) {
          console.warn('renderMathInElement error:', err);
        }
      });
    }

    return true;
  }

  // Poll until KaTeX is ready, then render
  function triggerRender(maxRetries) {
    var retries = 0;
    maxRetries = maxRetries || 60;
    var timer = setInterval(function () {
      retries++;
      if (renderAllMath() || retries >= maxRetries) {
        clearInterval(timer);
      }
    }, 50);
  }

  // Initial page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      triggerRender();
    });
  } else {
    triggerRender();
  }
  window.addEventListener('load', function () {
    triggerRender();
  });

  // Swup SPA integration for Redefine theme
  function initSwupIntegration(swup) {
    if (!swup || !swup.hooks) return;
    try {
      swup.hooks.on('page:view', function () {
        setTimeout(triggerRender, 50);
      });
      swup.hooks.on('content:replace', function () {
        setTimeout(triggerRender, 50);
      });
    } catch (e) {}
  }

  if (window.swup && window.swup.hooks) {
    initSwupIntegration(window.swup);
  } else {
    window.addEventListener('redefine:swup:ready', function (e) {
      var swupInstance = window.swup || e?.detail?.swup;
      initSwupIntegration(swupInstance);
    });
  }

  // Also support standard SPA events just in case
  document.addEventListener('swup:contentReplaced', function () {
    setTimeout(triggerRender, 50);
  });
  document.addEventListener('swup:pageView', function () {
    setTimeout(triggerRender, 50);
  });
  window.addEventListener('popstate', function () {
    setTimeout(triggerRender, 50);
  });

  // MutationObserver fallback to catch any DOM replacement in SPA
  var mutationTimer = null;
  var observer = new MutationObserver(function () {
    if (mutationTimer) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(function () {
      renderAllMath();
    }, 100);
  });

  var observeTarget = document.querySelector('main') || document.body;
  if (observeTarget) {
    observer.observe(observeTarget, { childList: true, subtree: true });
  }
})();
