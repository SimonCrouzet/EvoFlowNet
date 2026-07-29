// MathJax configuration for the equations carried in the API docstrings.
// Arithmatex (generic mode) emits them as `<script type="math/tex">`-free
// spans and divs tagged `arithmatex`, so MathJax is pointed at those.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material's instant navigation swaps the page body without a reload, so the
// equations on the newly loaded page have to be typeset again.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
