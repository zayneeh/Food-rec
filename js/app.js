// Menu toggle for mobile
const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');

menuToggle.addEventListener('click', () => {
  const isOpen = navLinks.style.display === 'flex';
  navLinks.style.display = isOpen ? 'none' : 'flex';
  navLinks.style.position = 'absolute';
  navLinks.style.top = '100%';
  navLinks.style.left = '0';
  navLinks.style.right = '0';
  navLinks.style.background = 'var(--primary-green)';
  navLinks.style.flexDirection = 'column';
  navLinks.style.padding = '1rem';
});

// Recipe Recommender Modal

async function fetchAIRecs(kind) {
  const res = await fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mood: kind })
  });
  const data = await res.json();
  let items = [];
  try { items = JSON.parse(data.recommendations); } catch {}
  const html = items.length
    ? `<ul style="margin-top:0.5rem; padding-left:1.2rem;">
         ${items.map(i => `<li><strong>${i.name}</strong> — ${i.why} ${i.source ? `<em>(source: ${i.source})</em>` : ''}</li>`).join('')}
       </ul>`
    : `<p>No matches yet—try another mood.</p>`;
  document.getElementById('recoResults').innerHTML = html;
}


// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Article cards click handler
document.querySelectorAll('.article-card').forEach(card => {
  card.addEventListener('click', () => {
    // Placeholder: navigate to article page in real implementation
    console.log('Article clicked');
  });
});

// Animate items on scroll
const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animation = 'fadeInUp 0.6s ease both';
    }
  });
}, observerOptions);

document.querySelectorAll('.category-item, .article-card').forEach(item => {
  observer.observe(item);
});

// Newsletter form handler
const form = document.querySelector('.newsletter-form');
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    alert('Thank you for subscribing! You will receive our latest updates on Nigerian food history.');
    e.target.reset();
  });
}

// Search functionality
const searchBtn = document.querySelector('.search-box button');
if (searchBtn) {
  searchBtn.addEventListener('click', () => {
    const searchTerm = document.querySelector('.search-box input').value;
    if (searchTerm) {
      alert(`Searching for: ${searchTerm}`);
      // TODO: hook up real search logic or navigate to results page
    }
  });
}

// ---- Dynamic Articles ----
async function loadArticles() {
  try {
    const res = await fetch('data/articles.json');
    const articles = await res.json();

    const grid = document.querySelector('.featured-grid');
    if (!grid) return;

    grid.innerHTML = articles.map(a => {
      const imageClass =
        a.imageVariant === 'green' ? 'article-image image-green' :
        a.imageVariant === 'brown' ? 'article-image image-brown' :
        'article-image';

      const date = new Date(a.publishedAt).toLocaleString('en-NG', {
        year: 'numeric', month: 'short', day: 'numeric'
      });

      return `
        <article class="article-card" data-url="${a.url}">
          <div class="${imageClass}"></div>
          <div class="article-content">
            <span class="article-category">${a.category}</span>
            <h3 class="article-title">${a.title}</h3>
            <p class="article-excerpt">${a.excerpt}</p>
            <div class="article-meta">
              <span>${a.readMins} min read</span>
              <span>${date}</span>
            </div>
          </div>
        </article>
      `;
    }).join('');

    // Click-through to article pages
    grid.querySelectorAll('.article-card').forEach(card => {
      card.addEventListener('click', () => {
        const url = card.getAttribute('data-url');
        if (url) location.href = url;
      });
    });
  } catch (e) {
    console.error('Failed to load articles', e);
  }
}
loadArticles();

