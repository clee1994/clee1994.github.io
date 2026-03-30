const navLinks = [
  { href: '/', label: 'About' },
  { href: '/blog.html', label: 'Blog' },
];

const nav = document.createElement('nav');
const path = location.pathname;

navLinks.forEach(({ href, label }) => {
  const a = document.createElement('a');
  a.href = href;
  a.textContent = label;
  const base = href.replace('.html', '');
  if (path === href || path === base || (href !== '/' && (path.endsWith(href) || path.endsWith(base)))) {
    a.classList.add('active');
  }
  nav.appendChild(a);
});

document.body.prepend(nav);
