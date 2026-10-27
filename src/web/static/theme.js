// Restore before the stylesheet paints; storage may be unavailable in private contexts.
(() => {
  const root = document.documentElement;
  const key = 'lora-workbench-theme';
  let theme = 'dark';
  try { if (localStorage.getItem(key) === 'light') theme = 'light'; } catch (_) {}
  root.dataset.theme = theme;

  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('btnTheme');
    const updateLabel = () => {
      const next = root.dataset.theme === 'light' ? 'Dark' : 'Light';
      button.textContent = next + ' mode';
      button.setAttribute('aria-label', 'Switch to ' + next.toLowerCase() + ' mode');
    };
    updateLabel();
    button.addEventListener('click', () => {
      root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
      try { localStorage.setItem(key, root.dataset.theme); } catch (_) {}
      updateLabel();
    });
  });
})();
