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

<div class="modal" id="recommenderModal">
  <div class="modal-content">
    <span class="modal-close" onclick="closeRecommender()">&times;</span>
    <h2>Discover Your Next Dish</h2>

    <h4 style="margin-top:1rem;">By Ingredients</h4>
    <p class="caption">e.g., rice, tomato, pepper, onions</p>
    <input id="reco-ingredients" style="width:100%;padding:0.6rem;margin-bottom:0.5rem;" placeholder="Type ingredients, comma-separated"/>
    <label>Match threshold (0.5–1.0)</label>
    <input id="thres-ingredients" type="number" min="0.5" max="1" step="0.05" value="0.7" style="width:100%;margin:0.25rem 0 1rem;"/><br/>
    <button id="btn-ingredients" style="width:100%;">Find Recipes by Ingredients</button>

    <h4 style="margin-top:1.5rem;">By Food Name</h4>
    <input id="reco-name" style="width:100%;padding:0.6rem;margin-bottom:0.5rem;" placeholder="Jollof Rice"/>
    <button id="btn-name" style="width:100%;">Find Recipes by Name</button>

    <h4 style="margin-top:1.5rem;">Talk to Me</h4>
    <p class="caption">e.g., what can I make with turkey and rice?</p>
    <input id="reco-prompt" style="width:100%;padding:0.6rem;margin-bottom:0.5rem;"/>
    <label>Semantic threshold (0.3–0.9)</label>
    <input id="thres-semantic" type="number" min="0.3" max="0.9" step="0.05" value="0.6" style="width:100%;margin:0.25rem 0 1rem;"/><br/>
    <button id="btn-prompt" style="width:100%;">Get Suggestions</button>

    <div id="recoResults" style="margin-top:1rem;"></div>
  </div>
</div>

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

