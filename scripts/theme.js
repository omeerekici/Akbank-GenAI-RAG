// @ts-nocheck
function toggleTheme() {
    const main = document.querySelector('.main');
    if (!main) return;

    if (main.classList.contains('dark')) {
        main.classList.remove('dark');
        main.classList.add('light');
    } else {
        main.classList.remove('light');
        main.classList.add('dark');
    }
}
