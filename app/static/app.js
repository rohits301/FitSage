const form = document.querySelector('#question-form');
const question = document.querySelector('#question');
const counter = document.querySelector('#counter');
const submit = document.querySelector('#submit');
const result = document.querySelector('#result');
const answer = document.querySelector('#answer-text');
const heading = document.querySelector('#answer-heading');
const modeTag = document.querySelector('#answer-mode');
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

const KIND_NOTE = { table: 'table extract, flattened as the page renders it', list: 'list extract, flattened as the page renders it' };

function renderSource(source, index) {
  const note = KIND_NOTE[source.kind] ? ` · ${KIND_NOTE[source.kind]}` : '';
  return `<a class="source" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">
    <div class="source-top"><span>${index === 0 ? 'Answer passage' : 'Related'} · ${escapeHtml(source.organization)}</span><span>Open source ↗</span></div>
    <h3>${escapeHtml(source.title)}</h3>
    <p class="section">${escapeHtml(source.section)}${escapeHtml(note)} · match ${Math.round(source.confidence * 100)}%</p>
    <p>“${escapeHtml(source.excerpt)}”</p></a>`;
}

function show(payload) {
  const declined = payload.mode === 'declined';
  heading.textContent = declined ? 'No answer from the evidence' : 'Evidence-backed answer';
  modeTag.textContent = payload.mode.toUpperCase();
  result.classList.toggle('declined', declined);
  answer.textContent = payload.answer;
  safety.textContent = payload.safety_note || '';
  sources.innerHTML = payload.sources.map(renderSource).join('');
  result.classList.remove('hidden'); result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true; submit.innerHTML = 'Finding evidence…';
  try {
    const response = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: question.value.trim() }) });
    const payload = await response.json();
    if (!response.ok) throw new Error('Please enter a question of 3–500 characters.');
    show(payload);
  } catch (error) {
    show({ mode: 'declined', answer: error.message, sources: [], safety_note: '' });
  } finally { submit.disabled = false; submit.innerHTML = 'Ask FitSage <span>→</span>'; }
});

fetch('/api/health').then((r) => r.json()).then((h) => {
  document.querySelector('#mode-label').textContent = h.generation === 'llm' ? 'LLM mode' : 'Extractive mode';
  document.querySelector('#corpus-note').textContent = `Corpus: ${h.passages} passages from ${h.sources} sources`;
}).catch(() => {});
