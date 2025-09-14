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
function openRecommender() {
  document.getElementById('recommenderModal').style.display = 'flex';
}
function closeRecommender() {
  document.getElementById('recommenderModal').style.display = 'none';
}
window.openRecommender = openRecommender;
window.closeRecommender = closeRecommender;

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
