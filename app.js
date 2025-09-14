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

// ===== FOOD RECOMMENDER MODAL FUNCTIONALITY =====

// Modal control functions
function openRecommender() {
  document.getElementById('recommenderModal').style.display = 'flex';
  showStep('reco-step-0');
}

function closeRecommender() {
  document.getElementById('recommenderModal').style.display = 'none';
  clearResults();
}

function showStep(stepId) {
  // Hide all steps
  document.querySelectorAll('.reco-step').forEach(step => {
    step.style.display = 'none';
  });
  // Show target step
  document.getElementById(stepId).style.display = 'block';
}

function clearResults() {
  document.getElementById('recoResults').innerHTML = '';
  document.getElementById('recoResultsName').innerHTML = '';
  document.getElementById('chatWindow').innerHTML = '';
}

// API call function
async function askAPI(question) {
  try {
    const res = await fetch(window.RECO_API_BASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    
    if (!res.ok) {
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }
    
    return await res.json();
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
}

// Step navigation event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Step 0 -> Choose mode
  document.getElementById('startChatBtn')?.addEventListener('click', () => {
    showStep('reco-step-chat');
  });
  
  document.getElementById('startRecoBtn')?.addEventListener('click', () => {
    showStep('reco-step-1');
  });

  // Back buttons
  document.getElementById('backTo0')?.addEventListener('click', () => {
    showStep('reco-step-0');
  });
  
  document.getElementById('backTo0b')?.addEventListener('click', () => {
    showStep('reco-step-0');
  });
  
  document.getElementById('backTo1a')?.addEventListener('click', () => {
    showStep('reco-step-1');
  });
  
  document.getElementById('backTo1b')?.addEventListener('click', () => {
    showStep('reco-step-1');
  });

  // Step 1 -> Choose recommendation type
  document.getElementById('byIngredientsBtn')?.addEventListener('click', () => {
    showStep('reco-step-ingredients');
  });
  
  document.getElementById('byNameBtn')?.addEventListener('click', () => {
    showStep('reco-step-name');
  });

  // Chat functionality
  document.getElementById('chatSendBtn')?.addEventListener('click', handleChatSend);
  document.getElementById('chatInput')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatSend();
  });

  // Ingredients search
  document.getElementById('btn-ingredients')?.addEventListener('click', handleIngredientsSearch);

  // Name search
  document.getElementById('btn-name')?.addEventListener('click', handleNameSearch);
});

// Chat handler
async function handleChatSend() {
  const input = document.getElementById('chatInput');
  const chatWindow = document.getElementById('chatWindow');
  const question = input.value.trim();
  
  if (!question) return;
  
  // Add user message
  appendChatMessage('user', question);
  input.value = '';
  
  // Add loading message
  const loadingId = Date.now();
  appendChatMessage('bot', 'Thinking...', loadingId);
  
  try {
    const data = await askAPI(question);
    
    // Remove loading message
    document.getElementById(`msg-${loadingId}`)?.remove();
    
    // Add bot response
    appendChatMessage('bot', data.answer);
    
    // Add sources if available
    if (data.sources && data.sources.length > 0) {
      const sourcesHtml = renderSources(data.sources);
      appendChatMessage('sources', sourcesHtml);
    }
  } catch (error) {
    // Remove loading message
    document.getElementById(`msg-${loadingId}`)?.remove();
    appendChatMessage('error', `⚠️ ${error.message}`);
  }
}

function appendChatMessage(type, content, id) {
  const chatWindow = document.getElementById('chatWindow');
  const messageId = id ? `msg-${id}` : '';
  
  const messageClass = type === 'user' ? 'chat-user' : 
                      type === 'error' ? 'chat-error' : 
                      type === 'sources' ? 'chat-sources' : 'chat-bot';
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${messageClass}`;
  if (messageId) messageDiv.id = messageId;
  messageDiv.innerHTML = content;
  
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Ingredients search handler
async function handleIngredientsSearch() {
  const ingredients = document.getElementById('reco-ingredients').value.trim();
  const threshold = document.getElementById('thres-ingredients').value;
  const resultsDiv = document.getElementById('recoResults');
  
  if (!ingredients) {
    resultsDiv.innerHTML = '<p class="error">Please enter some ingredients.</p>';
    return;
  }
  
  resultsDiv.innerHTML = '<p>Searching for recipes...</p>';
  
  const question = `What Nigerian dishes can I make with these ingredients: ${ingredients}? Please suggest recipes that use most of these ingredients.`;
  
  try {
    const data = await askAPI(question);
    displayResults(resultsDiv, data, `Results for ingredients: ${ingredients}`);
  } catch (error) {
    resultsDiv.innerHTML = `<p class="error">⚠️ ${error.message}</p>`;
  }
}

// Name search handler
async function handleNameSearch() {
  const dishName = document.getElementById('reco-name').value.trim();
  const resultsDiv = document.getElementById('recoResultsName');
  
  if (!dishName) {
    resultsDiv.innerHTML = '<p class="error">Please enter a dish name.</p>';
    return;
  }
  
  resultsDiv.innerHTML = '<p>Searching for recipe...</p>';
  
  const question = `Tell me about ${dishName}. How do I make it? What are the ingredients and cooking instructions?`;
  
  try {
    const data = await askAPI(question);
    displayResults(resultsDiv, data, `Recipe for: ${dishName}`);
  } catch (error) {
    resultsDiv.innerHTML = `<p class="error">⚠️ ${error.message}</p>`;
  }
}

// Display results helper
function displayResults(container, data, title) {
  const sourcesHtml = renderSources(data.sources);
  
  container.innerHTML = `
    <div class="reco-result">
      <h4>${title}</h4>
      <div class="answer-content">${escapeHtml(data.answer)}</div>
      ${sourcesHtml}
    </div>
  `;
}

// Render sources helper
function renderSources(sources = []) {
  if (!sources.length) return '';
  
  return `
    <details class="sources-details">
      <summary>Sources (${sources.length})</summary>
      <ul class="sources-list">
        ${sources.map(s => `
          <li class="source-item">
            <strong>${escapeHtml(s.source)}</strong>
            ${s.page != null ? `<span class="source-page">(row ${s.page})</span>` : ''}
            <div class="source-snippet">${escapeHtml(s.snippet)}</div>
          </li>
        `).join('')}
      </ul>
    </details>
  `;
}

// HTML escape utility
function escapeHtml(text) {
  if (!text) return '';
  return text.replace(/[&<>"']/g, (match) => {
    const escapeMap = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    };
    return escapeMap[match];
  });
}

// Close modal when clicking outside
window.addEventListener('click', (e) => {
  const modal = document.getElementById('recommenderModal');
  if (e.target === modal) {
    closeRecommender();
  }
});

// ===== OTHER WEBSITE FUNCTIONALITY =====

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