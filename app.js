// ===== CONFIGURATION =====
// SET YOUR API URL HERE - this is likely the main problem!
const API_BASE_URL = 'https://your-render-app-url.onrender.com/ask';
// Replace 'your-render-app-url' with your actual Render deployment URL

// ===== BASIC MENU TOGGLE =====
const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');

if (menuToggle && navLinks) {
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
}

// ===== MODAL FUNCTIONS =====
function openRecommender() {
  const modal = document.getElementById('recommenderModal');
  if (modal) {
    modal.style.display = 'flex';
    showStep('reco-step-0');
  }
}

function closeRecommender() {
  const modal = document.getElementById('recommenderModal');
  if (modal) {
    modal.style.display = 'none';
    clearResults();
  }
}

function showStep(stepId) {
  // Hide all steps
  document.querySelectorAll('.reco-step').forEach(step => {
    step.style.display = 'none';
  });
  // Show target step
  const targetStep = document.getElementById(stepId);
  if (targetStep) {
    targetStep.style.display = 'block';
  }
}

function clearResults() {
  const elements = ['recoResults', 'recoResultsName', 'chatWindow'];
  elements.forEach(id => {
    const element = document.getElementById(id);
    if (element) element.innerHTML = '';
  });
}

// ===== SIMPLIFIED API CALL =====
async function askAPI(question) {
  console.log('🔍 Making API call...');
  console.log('📍 URL:', API_BASE_URL);
  console.log('❓ Question:', question);
  
  // Check if URL is configured
  if (API_BASE_URL.includes('your-render-app-url')) {
    throw new Error('❌ API URL not configured! Please set your actual Render URL in app.js');
  }
  
  try {
    const response = await fetch(API_BASE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: question })
    });
    
    console.log('📡 Response status:', response.status);
    console.log('📡 Response OK:', response.ok);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Error response:', errorText);
      throw new Error(`API returned ${response.status}: ${errorText || response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Success! Data received:', data);
    return data;
    
  } catch (error) {
    console.error('💥 API Error Details:', error);
    
    // More specific error messages
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('❌ Cannot connect to API. Check your internet connection and API URL.');
    }
    if (error.message.includes('CORS')) {
      throw new Error('❌ CORS error. The API server may not allow requests from this domain.');
    }
    
    throw error;
  }
}

// ===== CHAT FUNCTIONALITY =====
function handleChatSend() {
  const input = document.getElementById('chatInput');
  const chatWindow = document.getElementById('chatWindow');
  
  if (!input || !chatWindow) {
    console.error('❌ Chat elements not found');
    return;
  }
  
  const question = input.value.trim();
  if (!question) {
    showMessage(chatWindow, 'Please enter a question!', 'error');
    return;
  }
  
  // Show user message
  showMessage(chatWindow, question, 'user');
  input.value = '';
  
  // Show loading
  const loadingId = 'loading-' + Date.now();
  showMessage(chatWindow, '🤔 Thinking...', 'loading', loadingId);
  
  // Call API
  askAPI(question)
    .then(data => {
      // Remove loading
      removeMessage(loadingId);
      
      // Show answer
      if (data.answer) {
        showMessage(chatWindow, data.answer, 'bot');
      }
      
      // Show sources if available
      if (data.sources && data.sources.length > 0) {
        showSources(chatWindow, data.sources);
      }
    })
    .catch(error => {
      // Remove loading
      removeMessage(loadingId);
      
      // Show error
      showMessage(chatWindow, error.message, 'error');
    });
}

// ===== MESSAGE HELPERS =====
function showMessage(container, content, type, id = null) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message chat-${type}`;
  if (id) messageDiv.id = id;
  
  // Simple HTML escaping for security
  const safeContent = content.replace(/[<>&"']/g, (c) => ({
    '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
  })[c]);
  
  messageDiv.innerHTML = safeContent;
  container.appendChild(messageDiv);
  container.scrollTop = container.scrollHeight;
}

function removeMessage(id) {
  const element = document.getElementById(id);
  if (element) element.remove();
}

function showSources(container, sources) {
  const sourcesDiv = document.createElement('div');
  sourcesDiv.className = 'chat-message chat-sources';
  
  let sourcesHtml = '<details><summary>📚 Sources (' + sources.length + ')</summary><ul>';
  sources.forEach(source => {
    const safeSource = (source.source || 'Unknown').replace(/[<>&"']/g, (c) => ({
      '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
    })[c]);
    const safeSnippet = (source.snippet || '').replace(/[<>&"']/g, (c) => ({
      '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
    })[c]);
    
    sourcesHtml += '<li><strong>' + safeSource + '</strong>';
    if (source.page != null) sourcesHtml += ' (row ' + source.page + ')';
    sourcesHtml += '<br><small>' + safeSnippet + '</small></li>';
  });
  sourcesHtml += '</ul></details>';
  
  sourcesDiv.innerHTML = sourcesHtml;
  container.appendChild(sourcesDiv);
  container.scrollTop = container.scrollHeight;
}

// ===== INGREDIENTS SEARCH =====
function handleIngredientsSearch() {
  const ingredientsInput = document.getElementById('reco-ingredients');
  const resultsDiv = document.getElementById('recoResults');
  
  if (!ingredientsInput || !resultsDiv) {
    console.error('❌ Ingredients search elements not found');
    return;
  }
  
  const ingredients = ingredientsInput.value.trim();
  if (!ingredients) {
    resultsDiv.innerHTML = '<p class="error">⚠️ Please enter some ingredients.</p>';
    return;
  }
  
  resultsDiv.innerHTML = '<p>🔍 Searching for recipes...</p>';
  
  const question = `What Nigerian dishes can I make with these ingredients: ${ingredients}? Please suggest recipes that use most of these ingredients and include cooking instructions.`;
  
  askAPI(question)
    .then(data => {
      showResults(resultsDiv, data, `Results for: ${ingredients}`);
    })
    .catch(error => {
      resultsDiv.innerHTML = `<p class="error">❌ ${error.message}</p>`;
    });
}

// ===== NAME SEARCH =====
function handleNameSearch() {
  const nameInput = document.getElementById('reco-name');
  const resultsDiv = document.getElementById('recoResultsName');
  
  if (!nameInput || !resultsDiv) {
    console.error('❌ Name search elements not found');
    return;
  }
  
  const dishName = nameInput.value.trim();
  if (!dishName) {
    resultsDiv.innerHTML = '<p class="error">⚠️ Please enter a dish name.</p>';
    return;
  }
  
  resultsDiv.innerHTML = '<p>🔍 Searching for recipe...</p>';
  
  const question = `Tell me about ${dishName}. How do I make it? What are the ingredients and complete cooking instructions?`;
  
  askAPI(question)
    .then(data => {
      showResults(resultsDiv, data, `Recipe for: ${dishName}`);
    })
    .catch(error => {
      resultsDiv.innerHTML = `<p class="error">❌ ${error.message}</p>`;
    });
}

// ===== RESULTS DISPLAY =====
function showResults(container, data, title) {
  let html = `<div class="reco-result">
    <h4>${title}</h4>
    <div class="answer-content">${data.answer || 'No answer received'}</div>`;
  
  if (data.sources && data.sources.length > 0) {
    html += '<details class="sources-details"><summary>📚 Sources (' + data.sources.length + ')</summary><ul>';
    data.sources.forEach(source => {
      html += '<li><strong>' + (source.source || 'Unknown') + '</strong>';
      if (source.page != null) html += ' (row ' + source.page + ')';
      html += '<br><small>' + (source.snippet || '') + '</small></li>';
    });
    html += '</ul></details>';
  }
  
  html += '</div>';
  container.innerHTML = html;
}

// ===== EVENT LISTENERS =====
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 App initializing...');
  console.log('🔗 API URL configured:', API_BASE_URL);
  
  // Modal controls
  const startChatBtn = document.getElementById('startChatBtn');
  const startRecoBtn = document.getElementById('startRecoBtn');
  const chatSendBtn = document.getElementById('chatSendBtn');
  const chatInput = document.getElementById('chatInput');
  const ingredientsBtn = document.getElementById('btn-ingredients');
  const nameBtn = document.getElementById('btn-name');
  
  // Back buttons
  const backButtons = [
    { id: 'backTo0', step: 'reco-step-0' },
    { id: 'backTo0b', step: 'reco-step-0' },
    { id: 'backTo1a', step: 'reco-step-1' },
    { id: 'backTo1b', step: 'reco-step-1' }
  ];
  
  // Step navigation buttons
  const stepButtons = [
    { id: 'byIngredientsBtn', step: 'reco-step-ingredients' },
    { id: 'byNameBtn', step: 'reco-step-name' }
  ];
  
  // Add event listeners with error checking
  if (startChatBtn) {
    startChatBtn.addEventListener('click', () => showStep('reco-step-chat'));
  }
  
  if (startRecoBtn) {
    startRecoBtn.addEventListener('click', () => showStep('reco-step-1'));
  }
  
  if (chatSendBtn) {
    chatSendBtn.addEventListener('click', handleChatSend);
  }
  
  if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleChatSend();
    });
  }
  
  if (ingredientsBtn) {
    ingredientsBtn.addEventListener('click', handleIngredientsSearch);
  }
  
  if (nameBtn) {
    nameBtn.addEventListener('click', handleNameSearch);
  }
  
  // Back buttons
  backButtons.forEach(btn => {
    const element = document.getElementById(btn.id);
    if (element) {
      element.addEventListener('click', () => showStep(btn.step));
    }
  });
  
  // Step buttons
  stepButtons.forEach(btn => {
    const element = document.getElementById(btn.id);
    if (element) {
      element.addEventListener('click', () => showStep(btn.step));
    }
  });
  
  console.log('✅ Event listeners attached');
});

// ===== MODAL CLOSE ON OUTSIDE CLICK =====
window.addEventListener('click', (e) => {
  const modal = document.getElementById('recommenderModal');
  if (e.target === modal) {
    closeRecommender();
  }
});

// ===== OTHER FEATURES (simplified) =====
// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Newsletter form
const form = document.querySelector('.newsletter-form');
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    alert('Thank you for subscribing! You will receive our latest updates on Nigerian food history.');
    e.target.reset();
  });
}

console.log('📱 App.js loaded successfully!');