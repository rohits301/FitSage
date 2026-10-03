const form = document.querySelector('#question-form');
const question = document.querySelector('#question');
const counter = document.querySelector('#counter');
const submit = document.querySelector('#submit');
const result = document.querySelector('#result');
const answer = document.querySelector('#answer-text');
const sources = document.querySelector('#sources');
const safety = document.querySelector('#safety-note');

question.addEventListener('input', () => { counter.textContent = `${question.value.length} / 500`; });
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  question.value = button.dataset.question;
  question.dispatchEvent(new Event('input'));
  question.focus();
}));

function escapeHtml(value) {
  const div = document.createElement('div'); div.textContent = value; return div.innerHTML;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true; submit.innerHTML = 'Finding evidence…';
  try {
    const response = await fetch('/api/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:question.value.trim()}) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Unable to retrieve an answer.');
    answer.textContent = payload.answer;
    safety.textContent = payload.safety_note;
    sources.innerHTML = payload.sources.map((source) => `<a class="source" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"><div class="source-top"><span>${escapeHtml(source.organization)}</span><span>Open source ↗</span></div><h3>${escapeHtml(source.title)}</h3><p>${escapeHtml(source.excerpt)}</p></a>`).join('');
    result.classList.remove('hidden'); result.scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) {
    answer.textContent = error.message; sources.innerHTML = ''; safety.textContent = '';
    result.classList.remove('hidden'); result.scrollIntoView({behavior:'smooth', block:'start'});
  } finally { submit.disabled = false; submit.innerHTML = 'Ask FitSage <span>→</span>'; }
});

