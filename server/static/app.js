/**
 * Neo4j Assistant — Web Client (Minimal ChatGPT Style)
 *
 * Логика: Toggle-микрофон (с паузой при воспроизведении), текстовый ввод, отправка на сервер,
 * потоковое воспроизведение PCM int16 24kHz ответа через Web Audio API с поддержкой Barge-in.
 */

const ASSISTANT_ICON = '<img src="icon.svg" alt="" width="32" height="32">';

// ===== State =====
let isRecording = false;
let isProcessing = false;
let currentUIState = 'idle';

// Медиа-рекордер для записи
let mediaRecorder = null;
let audioChunks = [];

// Аудио-состояние для потокового проигрывания
let audioCtx = null;
let nextStartTime = 0;
let activeSources = [];
const SAMPLE_RATE = 24000;
let currentAbortController = null;

// ===== DOM =====
const chatMessages = document.getElementById('chat-messages');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const stopIcon = document.getElementById('stop-icon');
const pauseIcon = document.getElementById('pause-icon');
const statusText = document.getElementById('db-status-text');
const connectionDot = document.getElementById('connection-dot');
const clearBtn = document.getElementById('clear-btn');

// ===== Audio Utilities =====

/**
 * Кодирует Float32Array в WAV Blob (IEEE float32, mono).
 * Совместимо с серверным sf.read(..., dtype="float32").
 */
function encodeWAV(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 4);
    const view = new DataView(buffer);

    function writeStr(offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    // RIFF header
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 4, true);
    writeStr(8, 'WAVE');

    // fmt chunk
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // chunk size
    view.setUint16(20, 3, true);           // audio format: 3 = IEEE float
    view.setUint16(22, 1, true);           // channels: 1 (mono)
    view.setUint32(24, sampleRate, true);  // sample rate
    view.setUint32(28, sampleRate * 4, true); // byte rate
    view.setUint16(32, 4, true);           // block align
    view.setUint16(34, 32, true);          // bits per sample

    // data chunk
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 4, true);

    for (let i = 0; i < numSamples; i++) {
        view.setFloat32(44 + i * 4, samples[i], true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Инициализирует или возобновляет AudioContext.
 */
async function initAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }
    return audioCtx;
}

/**
 * Воспроизводит один PCM int16 24kHz mono чанк с планированием времени в AudioContext.
 */
async function playChunk(pcmBytes) {
    if (!pcmBytes || pcmBytes.byteLength === 0) return;

    const ctx = await initAudioContext();
    
    const int16 = new Int16Array(pcmBytes);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
    }

    const audioBuffer = ctx.createBuffer(1, float32.length, SAMPLE_RATE);
    audioBuffer.copyToChannel(float32, 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    const currentTime = ctx.currentTime;
    let playTime = nextStartTime;

    // Сглаживание сетевого джиттера при старте новой очереди
    if (playTime < currentTime) {
        playTime = currentTime + 0.05; // 50ms буфер
    }

    source.start(playTime);
    activeSources.push(source);

    nextStartTime = playTime + audioBuffer.duration;

    source.onended = () => {
        const idx = activeSources.indexOf(source);
        if (idx !== -1) {
            activeSources.splice(idx, 1);
        }
    };
}

/**
 * Останавливает текущее воспроизведение, отменяет активный сетевой запрос и очищает очередь.
 */
function stopPlayback() {
    if (currentAbortController) {
        try {
            currentAbortController.abort();
        } catch (_) {}
        currentAbortController = null;
    }
    activeSources.forEach(source => {
        try { source.stop(); } catch (_) {}
    });
    activeSources = [];
    nextStartTime = 0;
    if (audioCtx) {
        try { audioCtx.close(); } catch (_) {}
        audioCtx = null;
    }
}

// ===== Microphone (Toggle & Pause logic) =====

async function toggleMic() {
    // В процессе непосредственного сетевого запроса/анализа не даем кликнуть
    if (currentUIState === 'processing') return;

    if (currentUIState === 'recording') {
        stopRecording();
    } else if (currentUIState === 'playing') {
        // Нажали ПАУЗУ во время проигрывания: останавливаем звук и переходим в idle
        stopPlayback();
        setUIState('idle');
    } else {
        // Обычный клик в состоянии idle: запускаем запись
        await startRecording();
    }
}

async function startRecording() {
    try {
        stopPlayback();

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            // Остановить треки микрофона
            stream.getTracks().forEach((t) => t.stop());

            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            if (blob.size === 0) {
                setUIState('idle');
                return;
            }
            await processAudioBlob(blob);
        };

        mediaRecorder.start();
        isRecording = true;
        setUIState('recording');
    } catch (err) {
        console.error('Microphone error:', err);
        addMessage('system', '⚠️ Нет доступа к микрофону. Разрешите доступ в настройках браузера.');
        setUIState('idle');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        isRecording = false;
        setUIState('processing');
    }
}

/**
 * Конвертирует записанный blob (webm/opus) → ресемплирует до 16кГц → WAV и отправляет на /process.
 */
async function processAudioBlob(blob) {
    setUIState('processing');
    showThinking();

    try {
        // Декодируем webm → PCM float32
        const arrayBuffer = await blob.arrayBuffer();
        const decodeCtx = new AudioContext();
        const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
        const float32 = audioBuffer.getChannelData(0);
        const originalSampleRate = audioBuffer.sampleRate;
        await decodeCtx.close();

        // Ресемплируем в 16000 Гц для лучшей совместимости с Whisper
        const targetRate = 16000;
        let resampledData = float32;

        if (originalSampleRate !== targetRate) {
            const offlineCtx = new OfflineAudioContext(
                1,
                Math.ceil(float32.length * targetRate / originalSampleRate),
                targetRate
            );
            const bufferSource = offlineCtx.createBufferSource();
            const srcBuffer = offlineCtx.createBuffer(1, float32.length, originalSampleRate);
            srcBuffer.copyToChannel(float32, 0);
            bufferSource.buffer = srcBuffer;
            bufferSource.connect(offlineCtx.destination);
            bufferSource.start();
            const rendered = await offlineCtx.startRendering();
            resampledData = rendered.getChannelData(0);
        }

        // Кодируем в WAV
        const wavBlob = encodeWAV(resampledData, targetRate);

        // Перед отправкой нового запроса прерываем предыдущее проигрывание
        stopPlayback();
        currentAbortController = new AbortController();

        // === Шаг 1: STT — получаем распознанный текст мгновенно ===
        const sttResponse = await fetch('/stt', {
            method: 'POST',
            headers: { 'Content-Type': 'audio/wav' },
            body: wavBlob,
            signal: currentAbortController.signal
        });

        removeThinking();

        if (!sttResponse.ok) {
            let errMsg = 'Ошибка сервера';
            try {
                const errData = await sttResponse.json();
                errMsg = errData.error || errMsg;
            } catch (_) {}
            addMessage('system', `⚠️ ${errMsg}`);
            setUIState('idle');
            return;
        }

        const sttData = await sttResponse.json();
        const recognizedText = sttData.text;

        // Показываем текст пользователя сразу
        addMessage('user', recognizedText);

        // === Шаг 2: LLM + TTS — отправляем текст на /process_text ===
        showThinking();

        const response = await fetch('/process_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: recognizedText }),
            signal: currentAbortController.signal
        });

        removeThinking();

        if (!response.ok) {
            let errMsg = 'Ошибка сервера';
            try {
                const errData = await response.json();
                errMsg = errData.error || errMsg;
            } catch (_) {}
            addMessage('system', `⚠️ ${errMsg}`);
            setUIState('idle');
            return;
        }

        await consumeProcessTextResponse(response);
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Fetch aborted.');
            return;
        }
        console.error('Process audio error:', err);
        removeThinking();
        addMessage('system', '⚠️ Ошибка соединения с сервером');
        setUIState('idle');
    }
}

// ===== Text Input =====

async function sendText() {
    const text = textInput.value.trim();
    if (!text || isProcessing) return;

    addMessage('user', text);
    textInput.value = '';
    setUIState('processing');

    // Перед отправкой нового запроса прерываем предыдущее воспроизведение
    stopPlayback();
    currentAbortController = new AbortController();

    try {
        await consumeProcessTextStream(text, currentAbortController.signal);
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Fetch aborted.');
            return;
        }
        console.error('Send text error:', err);
        addMessage('system', '⚠️ Ошибка соединения с сервером');
        setUIState('idle');
    }
}

/**
 * SSE: thinking / tool_call / tool_result / content / done / error
 */
async function consumeProcessTextStream(text, signal) {
    const myController = currentAbortController;
    const response = await fetch('/process_text_stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
        },
        body: JSON.stringify({ text }),
        signal,
    });

    if (!response.ok) {
        let errMsg = 'Ошибка сервера';
        try {
            const errData = await response.json();
            errMsg = errData.error || errMsg;
        } catch (_) {}
        addMessage('system', `⚠️ ${errMsg}`);
        setUIState('idle');
        return;
    }

    const shell = createAssistantStreamShell();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const scheduleMarkdown = throttle(() => {
        shell.answerEl.innerHTML = renderMarkdown(shell.answerText);
        scrollChatIfPinned();
    }, 50);

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let sep;
            while ((sep = buffer.indexOf('\n\n')) !== -1) {
                const rawEvent = buffer.slice(0, sep);
                buffer = buffer.slice(sep + 2);
                const parsed = parseSseEvent(rawEvent);
                if (!parsed) continue;

                const { event, data } = parsed;

                if (event === 'thinking') {
                    const delta = data.delta || '';
                    if (!delta) continue;
                    ensureProcessTrace(shell);
                    ensureThinkingBlock(shell);
                    shell.thinkingText += delta;
                    shell.thinkingBody.textContent = shell.thinkingText;
                    shell.thinkingBody.scrollTop = shell.thinkingBody.scrollHeight;
                    if (!shell.answerStarted) {
                        setProcessTraceLabel(shell, 'Working…');
                    }
                } else if (event === 'tool_call') {
                    ensureProcessTrace(shell);
                    beginToolStep(shell);
                    upsertToolCard(shell, data, 'running');
                    if (!shell.answerStarted) {
                        setProcessTraceLabel(shell, 'Working…');
                    }
                } else if (event === 'tool_result') {
                    ensureProcessTrace(shell);
                    upsertToolCard(
                        shell,
                        data,
                        data.ok === false ? 'error' : 'done'
                    );
                } else if (event === 'content') {
                    const delta = data.delta || '';
                    if (!delta) continue;
                    collapseProcessTraceForAnswer(shell);
                    shell.answerText += delta;
                    shell.answerEl.classList.add('streaming');
                    scheduleMarkdown();
                } else if (event === 'graph_highlight') {
                    const runId = data.graph_run_id ? String(data.graph_run_id) : '';
                    if (!runId) continue;
                    const tokens = Array.isArray(data.tokens)
                        ? data.tokens.map((t, i) => ({
                              id: t.id || `ext_${i}`,
                              text: String(t.text || t.edgeId || '').trim(),
                              color: t.color || GRAPH_TOKEN_PALETTE[i % GRAPH_TOKEN_PALETTE.length],
                              edgeId: t.edgeId || null,
                          })).filter((t) => t.edgeId || t.text.length >= 2)
                        : [];
                    applyExternalGraphSpec(runId, {
                        tokens,
                        note: data.note || '',
                    });
                } else if (event === 'done') {
                    shell.answerEl.classList.remove('streaming');
                    if (shell.processTrace && !shell.answerStarted) {
                        // No content streamed (e.g. empty) — still collapse with timer.
                        collapseProcessTraceForAnswer(shell);
                    } else if (shell.processTrace && shell.answerStarted) {
                        // Refresh elapsed at end.
                        const secs = Math.max(
                            1,
                            Math.round((Date.now() - shell.startedAt) / 1000)
                        );
                        setProcessTraceLabel(shell, `Worked for ${secs}s`);
                    }
                    if (data.final_content != null && data.final_content !== '') {
                        shell.answerText = data.final_content;
                    }
                    shell.answerEl.innerHTML = renderMarkdown(shell.answerText);
                    addCopyButton(shell.contentWrapper, shell.answerText);
                    addGraphButton(shell.contentWrapper, graphMetaFromDone(data));
                    scrollChatIfPinned();
                } else if (event === 'error') {
                    shell.answerEl.classList.remove('streaming');
                    if (shell.processTrace) {
                        collapseProcessTraceForAnswer(shell);
                    }
                    addMessage('system', `⚠️ ${data.message || 'Ошибка стрима'}`);
                }
            }
        }
    } finally {
        shell.answerEl.classList.remove('streaming');
        if (currentAbortController === myController) {
            setUIState('idle');
            currentAbortController = null;
        }
    }
}

function parseSseEvent(raw) {
    let event = 'message';
    const dataLines = [];
    for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) {
            event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
        }
    }
    if (!dataLines.length) return null;
    try {
        return { event, data: JSON.parse(dataLines.join('\n')) };
    } catch (_) {
        return { event, data: { delta: dataLines.join('\n') } };
    }
}

/** Scroll chat only if the user is already near the bottom (don't yank while reading). */
function scrollChatIfPinned() {
    const el = chatMessages;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (dist < 120) {
        el.scrollTop = el.scrollHeight;
    }
}

function createAssistantStreamShell() {
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="assistant-avatar">${ASSISTANT_ICON}</div>
        <div class="message-content-wrapper">
            <div class="message-header">
                <span class="message-label">Ассистент</span>
            </div>
            <div class="message-text markdown-body"></div>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const contentWrapper = div.querySelector('.message-content-wrapper');
    return {
        root: div,
        contentWrapper,
        answerEl: contentWrapper.querySelector('.message-text'),
        answerText: '',
        thinkingText: '',
        thinkingDetails: null,
        thinkingBody: null,
        thinkingCount: 0,
        needNewThinking: true,
        processTrace: null,
        processBody: null,
        processLabel: null,
        toolCards: {},
        startedAt: Date.now(),
        answerStarted: false,
    };
}

function ensureProcessTrace(shell) {
    if (shell.processTrace) return;
    const details = document.createElement('details');
    details.className = 'process-trace';
    // Collapsed by default; never force-open on new steps.
    details.open = false;
    details.innerHTML = `
        <summary class="process-trace-summary">
            <span class="chevron"></span>
            <span class="process-trace-label">Working…</span>
        </summary>
        <div class="process-trace-body"></div>`;
    shell.contentWrapper.insertBefore(details, shell.answerEl);
    shell.processTrace = details;
    shell.processBody = details.querySelector('.process-trace-body');
    shell.processLabel = details.querySelector('.process-trace-label');
}

function setProcessTraceLabel(shell, text) {
    if (shell.processLabel) {
        shell.processLabel.textContent = text;
    }
}

function collapseProcessTraceForAnswer(shell) {
    if (!shell.processTrace || shell.answerStarted) return;
    shell.answerStarted = true;
    const secs = Math.max(1, Math.round((Date.now() - shell.startedAt) / 1000));
    setProcessTraceLabel(shell, `Worked for ${secs}s`);
    // Stay collapsed (or keep whatever the user chose).
    shell.processTrace.open = false;
}

/** After a tool step, the next reasoning goes into a fresh Thinking block. */
function beginToolStep(shell) {
    if (shell.thinkingDetails) {
        shell.thinkingDetails.open = false;
    }
    shell.needNewThinking = true;
    shell.thinkingDetails = null;
    shell.thinkingBody = null;
    shell.thinkingText = '';
}

function ensureThinkingBlock(shell) {
    ensureProcessTrace(shell);
    if (!shell.needNewThinking && shell.thinkingDetails) return;

    shell.thinkingCount += 1;
    const details = document.createElement('details');
    details.className = 'stream-thinking';
    details.open = true;
    const label =
        shell.thinkingCount === 1 ? 'Thinking' : `Thinking ${shell.thinkingCount}`;
    details.innerHTML = `
        <summary><span class="chevron"></span><span>${label}</span></summary>
        <pre class="stream-thinking-body"></pre>`;
    shell.processBody.appendChild(details);
    shell.thinkingDetails = details;
    shell.thinkingBody = details.querySelector('.stream-thinking-body');
    shell.thinkingText = '';
    shell.needNewThinking = false;
}

function upsertToolCard(shell, data, status) {
    ensureProcessTrace(shell);
    const id = data.id || data.name || 'tool';

    let card = shell.toolCards[id];
    if (!card) {
        // Close current thinking; next think after this tool is a new block.
        beginToolStep(shell);
        card = document.createElement('details');
        card.className = 'tool-card';
        card.open = true; // expand while running
        card.dataset.toolId = id;
        card.innerHTML = `
            <summary class="tool-card-header">
                <span class="tool-card-summary-left">
                    <span class="chevron"></span>
                    <span class="tool-card-name"></span>
                </span>
                <span class="tool-card-status"></span>
            </summary>
            <div class="tool-card-body">
                <pre class="tool-card-args"></pre>
                <pre class="tool-card-result" style="display:none"></pre>
            </div>`;
        shell.processBody.appendChild(card);
        shell.toolCards[id] = card;
    }

    card.querySelector('.tool-card-name').textContent = data.name || id;
    const statusEl = card.querySelector('.tool-card-status');
    statusEl.className = `tool-card-status ${status}`;
    statusEl.textContent =
        status === 'running' ? 'running' : status === 'error' ? 'error' : 'done';

    if (data.arguments !== undefined) {
        const argsEl = card.querySelector('.tool-card-args');
        argsEl.textContent =
            typeof data.arguments === 'string'
                ? data.arguments
                : JSON.stringify(data.arguments, null, 2);
    }

    const resultText = data.result != null ? data.result : data.preview;
    if (resultText !== undefined && resultText !== null) {
        const resultEl = card.querySelector('.tool-card-result');
        resultEl.style.display = 'block';
        resultEl.textContent = resultText;
        resultEl.scrollTop = 0;
    }

    // Collapse when finished so the trace stays compact; user can re-open.
    if (status === 'done' || status === 'error') {
        card.open = false;
    }
}

function throttle(fn, ms) {
    let last = 0;
    let timer = null;
    return () => {
        const now = Date.now();
        const run = () => {
            last = Date.now();
            timer = null;
            fn();
        };
        if (now - last >= ms) {
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
            run();
        } else if (!timer) {
            timer = setTimeout(run, ms - (now - last));
        }
    };
}

/**
 * Обрабатывает ответ /process_text:
 * - JSON (audio_enabled=false) — только текст
 * - audio/pcm — заголовок LLM-Response + потоковое воспроизведение
 */
async function consumeProcessTextResponse(response) {
    const contentType = response.headers.get('Content-Type') || '';
    const myController = currentAbortController;

    if (contentType.includes('application/json')) {
        const data = await response.json();
        const llmResponse = data.answer || '';
        if (llmResponse) addMessage('assistant', llmResponse);
        if (currentAbortController === myController) {
            setUIState('idle');
            currentAbortController = null;
        }
        return;
    }

    const llmResponse = safeDecodeHeader(response.headers.get('LLM-Response'));
    if (llmResponse) addMessage('assistant', llmResponse);

    setUIState('playing');

    const reader = response.body.getReader();
    let leftoverBytes = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        let data = value;
        if (leftoverBytes) {
            const combined = new Uint8Array(leftoverBytes.length + data.length);
            combined.set(leftoverBytes);
            combined.set(data, leftoverBytes.length);
            data = combined;
            leftoverBytes = null;
        }

        if (data.length % 2 !== 0) {
            leftoverBytes = data.slice(data.length - 1);
            data = data.slice(0, data.length - 1);
        }

        if (data.length > 0) {
            const pcmChunk = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
            await playChunk(pcmChunk);
        }
    }

    if (audioCtx && nextStartTime > audioCtx.currentTime) {
        const delay = (nextStartTime - audioCtx.currentTime) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    if (currentAbortController === myController) {
        setUIState('idle');
        currentAbortController = null;
    }
}

// ===== Chain Graph Modal =====

const GRAPH_NODE_COLORS = {
    Metabolite: '#c990c0',
    Microbe: '#569480',
    StarterCulture: '#4c8dff',
    EnvironmentCondition: '#f0a85e',
};
const GRAPH_DEFAULT_NODE_COLOR = '#a5abb6';
const GRAPH_TOKEN_PALETTE = [
    '#f59e0b',
    '#34d399',
    '#60a5fa',
    '#f472b6',
    '#a78bfa',
    '#fb7185',
];
const graphPayloadCache = new Map();
const pendingHighlights = new Map();
let graphModal = null;
let graphTokenSeq = 0;

function graphMetaFromDone(data) {
    const runId = data && data.graph_run_id ? String(data.graph_run_id) : '';
    const chainCount = Number(data && data.graph_chain_count) || 0;
    if (!runId || chainCount < 1) return null;
    return { runId, chainCount };
}

async function fetchGraphViz(runId) {
    if (graphPayloadCache.has(runId)) {
        return graphPayloadCache.get(runId);
    }

    const resp = await fetch('/graph_viz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph_run_id: runId }),
    });
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        throw new Error(payload.error || 'Не удалось загрузить граф');
    }
    graphPayloadCache.set(runId, payload);
    return payload;
}

function addGraphButton(contentWrapper, graphMeta) {
    if (!graphMeta || !graphMeta.runId) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'show-graph-btn';
    btn.dataset.graphRunId = graphMeta.runId;
    btn.innerHTML = `
        <span class="spinner-small" style="display:none"></span>
        <span class="show-graph-label">Показать граф${graphMeta.chainCount > 1 ? ` (${graphMeta.chainCount})` : ''}</span>`;
    btn.addEventListener('click', () => openGraphModal(graphMeta, btn));
    contentWrapper.appendChild(btn);
}

// ----- Pure edge search engine -----

function normalizeQuery(s) {
    return String(s || '').trim().toLowerCase();
}

function buildEdgeSearchIndex(payload) {
    const nodeCaptions = new Map();
    for (const view of (payload && payload.views) || []) {
        for (const node of view.nodes || []) {
            if (!nodeCaptions.has(node.id)) {
                nodeCaptions.set(node.id, node.caption || node.label || node.id);
            }
        }
    }
    for (const node of ((payload && payload.all && payload.all.nodes) || [])) {
        if (!nodeCaptions.has(node.id)) {
            nodeCaptions.set(node.id, node.caption || node.label || node.id);
        }
    }

    const byId = new Map();
    const consider = (edge) => {
        if (!edge || !edge.id) return;
        const props = edge.properties || {};
        const fromCaption = nodeCaptions.get(edge.from) || edge.from || '';
        const toCaption = nodeCaptions.get(edge.to) || edge.to || '';
        const evidence = String(props.evidence || '');
        const sourceFile = String(props.source_file || '');
        const label = String(edge.label || '');
        const chainIds = Array.isArray(edge.chain_ids) ? edge.chain_ids.slice() : [];
        const existing = byId.get(edge.id);
        if (existing) {
            for (const cid of chainIds) {
                if (!existing.chainIds.includes(cid)) existing.chainIds.push(cid);
            }
            return;
        }
        const haystack = normalizeQuery(
            [evidence, label, fromCaption, toCaption, sourceFile].join(' ')
        );
        byId.set(edge.id, {
            edgeId: edge.id,
            haystack,
            chainIds,
            label,
            fromCaption,
            toCaption,
            evidenceSnippet: evidence.length > 90 ? evidence.slice(0, 87) + '…' : evidence,
            from: edge.from,
            to: edge.to,
        });
    };

    for (const edge of ((payload && payload.all && payload.all.edges) || [])) {
        consider(edge);
    }
    for (const view of (payload && payload.views) || []) {
        for (const edge of view.edges || []) consider(edge);
    }
    return Array.from(byId.values());
}

function emptyMatchBucket() {
    return { edgeIds: new Set(), nodeIds: new Set(), perToken: new Map() };
}

function matchTokens(payload, tokens) {
    const index = buildEdgeSearchIndex(payload);
    const byEdgeId = new Map(index.map((item) => [item.edgeId, item]));
    const perView = new Map();
    const all = emptyMatchBucket();

    for (const view of (payload && payload.views) || []) {
        perView.set(view.id, emptyMatchBucket());
    }

    const active = (tokens || []).filter((t) => {
        if (t.edgeId) return true;
        return normalizeQuery(t.text).length >= 2;
    });
    if (!active.length) {
        return { perView, all, active: false };
    }

    const addMatch = (token, item) => {
        if (!item) return;
        all.edgeIds.add(item.edgeId);
        if (item.from) all.nodeIds.add(item.from);
        if (item.to) all.nodeIds.add(item.to);
        if (!all.perToken.has(token.id)) all.perToken.set(token.id, new Set());
        all.perToken.get(token.id).add(item.edgeId);

        for (const chainId of item.chainIds) {
            let bucket = perView.get(chainId);
            if (!bucket) {
                bucket = emptyMatchBucket();
                perView.set(chainId, bucket);
            }
            bucket.edgeIds.add(item.edgeId);
            if (item.from) bucket.nodeIds.add(item.from);
            if (item.to) bucket.nodeIds.add(item.to);
            if (!bucket.perToken.has(token.id)) bucket.perToken.set(token.id, new Set());
            bucket.perToken.get(token.id).add(item.edgeId);
        }
    };

    for (const token of active) {
        if (token.edgeId) {
            addMatch(token, byEdgeId.get(token.edgeId));
            continue;
        }
        const q = normalizeQuery(token.text);
        for (const item of index) {
            if (!item.haystack.includes(q)) continue;
            addMatch(token, item);
        }
    }

    return { perView, all, active: true };
}

function suggestEdges(index, query, limit) {
    const q = normalizeQuery(query);
    if (q.length < 2) return [];
    const max = limit || 8;
    const out = [];
    for (const item of index) {
        if (!item.haystack.includes(q)) continue;
        out.push(item);
        if (out.length >= max) break;
    }
    return out;
}

function nextTokenColor(tokens) {
    return GRAPH_TOKEN_PALETTE[tokens.length % GRAPH_TOKEN_PALETTE.length];
}

function edgeTokenLabel(item) {
    if (!item) return 'edge';
    const title = `${item.fromCaption} -[${item.label}]-> ${item.toCaption}`;
    return title.length > 48 ? title.slice(0, 45) + '…' : title;
}

function makeToken(opts, tokens) {
    graphTokenSeq += 1;
    const text = String((opts && opts.text) || '').trim();
    return {
        id: `tok_${graphTokenSeq}`,
        text,
        color: nextTokenColor(tokens || []),
        edgeId: (opts && opts.edgeId) || null,
    };
}

function visibleViewsForModal(modal) {
    const views = (modal.payload && modal.payload.views) || [];
    if (!modal.matches || !modal.matches.active) return views;
    return views.filter((view) => {
        const bucket = modal.matches.perView.get(view.id);
        return bucket && bucket.edgeIds.size > 0;
    });
}

function currentMatchBucket(modal) {
    if (!modal.matches || !modal.matches.active) return null;
    if (modal.viewIndex === -1) return modal.matches.all;
    const views = (modal.payload && modal.payload.views) || [];
    const view = views[modal.viewIndex];
    if (!view) return modal.matches.all;
    return modal.matches.perView.get(view.id) || emptyMatchBucket();
}

function firstTokenColorForEdge(modal, edgeId) {
    const tokens = (modal.spec && modal.spec.tokens) || [];
    const bucket = currentMatchBucket(modal);
    if (!bucket) return GRAPH_TOKEN_PALETTE[0];
    for (const token of tokens) {
        const set = bucket.perToken.get(token.id);
        if (set && set.has(edgeId)) return token.color;
    }
    return GRAPH_TOKEN_PALETTE[0];
}

function ensureGraphModal() {
    if (graphModal) return graphModal;

    const overlay = document.createElement('div');
    overlay.className = 'graph-modal-overlay';
    overlay.innerHTML = `
        <div class="graph-modal" role="dialog" aria-modal="true" aria-label="Граф цепочек">
            <div class="graph-modal-header">
                <div>
                    <div class="graph-modal-title">Граф цепочек</div>
                    <div class="graph-modal-subtitle"></div>
                </div>
                <button class="graph-modal-close" type="button" aria-label="Закрыть">×</button>
            </div>
            <div class="graph-modal-search">
                <div class="graph-modal-chips"></div>
                <input class="graph-modal-search-input" type="search" placeholder="Поиск по evidence / рёбрам…" autocomplete="off" spellcheck="false">
                <div class="graph-modal-suggest" style="display:none"></div>
            </div>
            <div class="graph-modal-note" style="display:none"></div>
            <div class="graph-modal-toolbar">
                <button class="graph-modal-nav" type="button" data-nav="prev" aria-label="Предыдущая цепь">‹</button>
                <button class="graph-modal-all" type="button">Весь граф</button>
                <button class="graph-modal-nav" type="button" data-nav="next" aria-label="Следующая цепь">›</button>
            </div>
            <div class="graph-modal-canvas-wrap">
                <div class="graph-modal-canvas"></div>
                <div class="graph-modal-empty" style="display:none"></div>
            </div>
            <div class="graph-modal-details node-details-panel" style="display:none"></div>
        </div>`;

    document.body.appendChild(overlay);

    graphModal = {
        overlay,
        canvas: overlay.querySelector('.graph-modal-canvas'),
        empty: overlay.querySelector('.graph-modal-empty'),
        details: overlay.querySelector('.graph-modal-details'),
        subtitle: overlay.querySelector('.graph-modal-subtitle'),
        allBtn: overlay.querySelector('.graph-modal-all'),
        navBtns: overlay.querySelectorAll('.graph-modal-nav'),
        chipsEl: overlay.querySelector('.graph-modal-chips'),
        searchInput: overlay.querySelector('.graph-modal-search-input'),
        suggestEl: overlay.querySelector('.graph-modal-suggest'),
        noteEl: overlay.querySelector('.graph-modal-note'),
        network: null,
        payload: null,
        runId: null,
        viewIndex: -1,
        searchIndex: [],
        spec: { tokens: [], note: '' },
        matches: null,
        suggestTimer: null,
        focusEdgeId: null,
    };

    overlay.addEventListener('click', (event) => {
        if (event.target === overlay) closeGraphModal();
    });
    overlay.querySelector('.graph-modal-close').addEventListener('click', closeGraphModal);
    graphModal.allBtn.addEventListener('click', () => {
        if (!graphModal.payload) return;
        graphModal.viewIndex = -1;
        renderGraphView();
    });
    graphModal.navBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            stepGraphView(btn.dataset.nav === 'next' ? 1 : -1);
        });
    });

    graphModal.searchInput.addEventListener('input', () => {
        clearTimeout(graphModal.suggestTimer);
        graphModal.suggestTimer = setTimeout(() => renderGraphSuggestions(), 120);
    });
    graphModal.searchInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            const text = graphModal.searchInput.value.trim();
            if (text.length >= 2) {
                pinGraphTextToken(text);
                graphModal.searchInput.value = '';
                hideGraphSuggestions();
            }
        } else if (event.key === 'Backspace' && !graphModal.searchInput.value) {
            const tokens = graphModal.spec.tokens;
            if (tokens.length) {
                event.preventDefault();
                removeGraphToken(tokens[tokens.length - 1].id);
            }
        } else if (event.key === 'Escape') {
            if (graphModal.suggestEl.style.display !== 'none') {
                event.preventDefault();
                event.stopPropagation();
                hideGraphSuggestions();
            }
        }
    });
    graphModal.searchInput.addEventListener('blur', () => {
        setTimeout(() => hideGraphSuggestions(), 150);
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && graphModal.overlay.classList.contains('open')) {
            if (graphModal.suggestEl.style.display !== 'none') {
                hideGraphSuggestions();
                return;
            }
            closeGraphModal();
        }
    });

    return graphModal;
}

function renderGraphChips() {
    const modal = ensureGraphModal();
    modal.chipsEl.innerHTML = '';
    for (const token of modal.spec.tokens) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'graph-modal-chip';
        chip.innerHTML = `
            <span class="graph-modal-chip-dot" style="background:${token.color}"></span>
            <span class="graph-modal-chip-text">${escapeHtml(token.text)}</span>
            <span class="graph-modal-chip-x" aria-hidden="true">×</span>`;
        chip.title = 'Убрать фильтр';
        chip.addEventListener('click', () => removeGraphToken(token.id));
        modal.chipsEl.appendChild(chip);
    }
}

function renderGraphNote() {
    const modal = ensureGraphModal();
    const note = (modal.spec && modal.spec.note) || '';
    if (!note) {
        modal.noteEl.style.display = 'none';
        modal.noteEl.textContent = '';
        return;
    }
    modal.noteEl.style.display = 'block';
    modal.noteEl.textContent = note;
}

function hideGraphSuggestions() {
    const modal = ensureGraphModal();
    modal.suggestEl.style.display = 'none';
    modal.suggestEl.innerHTML = '';
}

function renderGraphSuggestions() {
    const modal = ensureGraphModal();
    const query = modal.searchInput.value.trim();
    if (!modal.payload || normalizeQuery(query).length < 2) {
        hideGraphSuggestions();
        return;
    }
    const items = suggestEdges(modal.searchIndex, query, 8);
    if (!items.length) {
        modal.suggestEl.style.display = 'block';
        modal.suggestEl.innerHTML = '<div class="graph-modal-suggest-empty">Нет совпадений</div>';
        return;
    }
    modal.suggestEl.style.display = 'block';
    modal.suggestEl.innerHTML = items.map((item) => {
        const chains = (item.chainIds || []).join(', ');
        return `
            <button type="button" class="graph-modal-suggest-item" data-edge-id="${escapeHtml(item.edgeId)}">
                <div class="graph-modal-suggest-title">${escapeHtml(item.fromCaption)} -[${escapeHtml(item.label)}]-> ${escapeHtml(item.toCaption)}</div>
                <div class="graph-modal-suggest-meta">${escapeHtml(item.evidenceSnippet || '')}${chains ? ` · ${escapeHtml(chains)}` : ''}</div>
            </button>`;
    }).join('');
    modal.suggestEl.querySelectorAll('.graph-modal-suggest-item').forEach((btn) => {
        btn.addEventListener('mousedown', (event) => {
            event.preventDefault();
            const edgeId = btn.dataset.edgeId;
            const item = modal.searchIndex.find((x) => x.edgeId === edgeId);
            pinGraphEdgeToken(item || { edgeId, fromCaption: '?', label: 'REL', toCaption: '?' });
            modal.searchInput.value = '';
            hideGraphSuggestions();
        });
    });
}

function pinGraphTextToken(text) {
    const modal = ensureGraphModal();
    const normalized = normalizeQuery(text);
    if (normalized.length < 2) return;
    const exists = modal.spec.tokens.some(
        (t) => !t.edgeId && normalizeQuery(t.text) === normalized
    );
    if (!exists) {
        modal.spec.tokens.push(makeToken({ text }, modal.spec.tokens));
    }
    applyGraphSpec(modal.spec);
}

function pinGraphEdgeToken(item) {
    const modal = ensureGraphModal();
    if (!item || !item.edgeId) return;
    const exists = modal.spec.tokens.some((t) => t.edgeId === item.edgeId);
    if (!exists) {
        modal.spec.tokens.push(
            makeToken(
                { text: edgeTokenLabel(item), edgeId: item.edgeId },
                modal.spec.tokens
            )
        );
    }
    modal.focusEdgeId = item.edgeId;
    applyGraphSpec(modal.spec);
}

function removeGraphToken(tokenId) {
    const modal = ensureGraphModal();
    modal.spec.tokens = modal.spec.tokens.filter((t) => t.id !== tokenId);
    applyGraphSpec(modal.spec);
}

function applyGraphSpec(spec) {
    const modal = ensureGraphModal();
    modal.spec = {
        tokens: Array.isArray(spec && spec.tokens) ? spec.tokens.slice() : [],
        note: (spec && spec.note) || '',
    };
    renderGraphChips();
    renderGraphNote();

    if (!modal.payload) {
        modal.matches = null;
        return;
    }

    modal.matches = matchTokens(modal.payload, modal.spec.tokens);
    const visible = visibleViewsForModal(modal);

    if (modal.viewIndex !== -1) {
        const current = (modal.payload.views || [])[modal.viewIndex];
        const stillVisible = current && visible.some((v) => v.id === current.id);
        if (!stillVisible) {
            if (visible.length === 1) modal.viewIndex = (modal.payload.views || []).indexOf(visible[0]);
            else if (visible.length) {
                const first = visible[0];
                modal.viewIndex = (modal.payload.views || []).findIndex((v) => v.id === first.id);
            } else {
                modal.viewIndex = -1;
            }
        }
    }

    renderGraphView();
}

function applyExternalGraphSpec(runId, spec) {
    if (!runId || !spec) return;
    pendingHighlights.set(runId, {
        tokens: Array.isArray(spec.tokens) ? spec.tokens.slice() : [],
        note: spec.note || '',
    });
    const modal = ensureGraphModal();
    if (modal.overlay.classList.contains('open') && modal.runId === runId) {
        const pending = pendingHighlights.get(runId);
        pendingHighlights.delete(runId);
        applyGraphSpec(pending);
        return;
    }
    openGraphModal({ runId, chainCount: 0 }, null);
}

async function openGraphModal(graphMeta, btn) {
    const modal = ensureGraphModal();
    const label = btn ? btn.querySelector('.show-graph-label') : null;
    const spinner = btn ? btn.querySelector('.spinner-small') : null;

    if (btn) btn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';
    if (label) label.textContent = 'Загрузка...';

    modal.overlay.classList.add('open');
    document.body.classList.add('graph-modal-open');
    setGraphModalMessage('Загрузка графа...', true);

    try {
        const payload = await fetchGraphViz(graphMeta.runId);
        modal.payload = payload;
        modal.runId = graphMeta.runId;
        modal.searchIndex = buildEdgeSearchIndex(payload);
        modal.viewIndex = payload.views && payload.views.length === 1 ? 0 : -1;

        const pending = pendingHighlights.get(graphMeta.runId);
        if (pending) {
            pendingHighlights.delete(graphMeta.runId);
            applyGraphSpec(pending);
        } else {
            applyGraphSpec(modal.spec);
        }
    } catch (err) {
        console.error('Graph viz error:', err);
        setGraphModalMessage(err.message || 'Не удалось загрузить граф', false);
    } finally {
        if (btn) btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
        if (label) {
            const count = graphMeta.chainCount || ((modal.payload && modal.payload.views) || []).length;
            label.textContent = `Показать граф${count > 1 ? ` (${count})` : ''}`;
        }
    }
}

function closeGraphModal() {
    if (!graphModal) return;
    hideGraphSuggestions();
    graphModal.overlay.classList.remove('open');
    document.body.classList.remove('graph-modal-open');
    if (graphModal.network) {
        graphModal.network.destroy();
        graphModal.network = null;
    }
}

function setGraphModalMessage(text, loading) {
    const modal = ensureGraphModal();
    if (modal.network) {
        modal.network.destroy();
        modal.network = null;
    }
    modal.canvas.innerHTML = '';
    modal.details.style.display = 'none';
    modal.empty.style.display = 'flex';
    modal.empty.innerHTML = loading
        ? '<span class="spinner-small"></span><span>' + escapeHtml(text) + '</span>'
        : escapeHtml(text);
    updateGraphToolbar();
}

function currentGraphView() {
    const modal = ensureGraphModal();
    if (!modal.payload) return null;
    if (modal.viewIndex === -1) return modal.payload.all;
    return (modal.payload.views || [])[modal.viewIndex] || null;
}

function stepGraphView(delta) {
    const modal = ensureGraphModal();
    const visible = visibleViewsForModal(modal);
    if (!visible.length) {
        modal.viewIndex = -1;
        renderGraphView();
        return;
    }

    // Order: all (-1) then each visible view.
    const order = [-1].concat(
        visible.map((v) => (modal.payload.views || []).findIndex((x) => x.id === v.id))
    );
    let pos = order.indexOf(modal.viewIndex);
    if (pos < 0) pos = 0;
    const nextPos = (pos + delta + order.length) % order.length;
    modal.viewIndex = order[nextPos];
    renderGraphView();
}

function updateGraphToolbar() {
    const modal = ensureGraphModal();
    const allViews = (modal.payload && modal.payload.views) || [];
    const visible = visibleViewsForModal(modal);
    const chainCount = visible.length;
    const filtering = Boolean(modal.matches && modal.matches.active);

    if (!modal.payload) {
        modal.subtitle.textContent = '';
        modal.allBtn.classList.remove('active');
        modal.navBtns.forEach((btn) => { btn.disabled = true; });
        return;
    }

    let matchCount = 0;
    if (filtering) {
        const bucket = currentMatchBucket(modal);
        matchCount = bucket ? bucket.edgeIds.size : 0;
    }

    if (modal.viewIndex === -1) {
        const suffix = filtering
            ? ` · совпадений: ${matchCount} · цепей: ${chainCount}/${allViews.length}`
            : ` · ${allViews.length}`;
        modal.subtitle.textContent = `Все цепи${suffix}`;
        modal.allBtn.classList.add('active');
    } else {
        const view = allViews[modal.viewIndex];
        const visiblePos = visible.findIndex((v) => view && v.id === view.id) + 1;
        const posLabel = filtering && visiblePos > 0
            ? `${visiblePos} из ${chainCount}`
            : `${modal.viewIndex + 1} из ${allViews.length}`;
        const matchPart = filtering ? ` · совпадений: ${matchCount}` : '';
        modal.subtitle.textContent = `${view ? view.label : 'Цепь'} ${posLabel}${matchPart} · score ${Number((view && view.score) || 0).toFixed(3)}`;
        modal.allBtn.classList.remove('active');
    }
    modal.navBtns.forEach((btn) => { btn.disabled = allViews.length < 1; });
}

function renderGraphView() {
    const modal = ensureGraphModal();
    const view = currentGraphView();
    updateGraphToolbar();

    if (!view || !view.nodes || !view.nodes.length) {
        setGraphModalMessage(
            (modal.matches && modal.matches.active)
                ? 'Нет цепей с совпадениями'
                : 'Нет данных для визуализации',
            false
        );
        return;
    }

    modal.empty.style.display = 'none';
    modal.details.style.display = 'none';
    modal.canvas.innerHTML = '';

    if (modal.network) {
        modal.network.destroy();
        modal.network = null;
    }

    const bucket = currentMatchBucket(modal);
    const filtering = Boolean(bucket && modal.matches && modal.matches.active);

    const visNodes = new vis.DataSet(view.nodes.map((node) => {
        const color = node.color || GRAPH_NODE_COLORS[node.group] || GRAPH_DEFAULT_NODE_COLOR;
        const caption = node.caption || node.label || node.id;
        const matched = filtering && bucket.nodeIds.has(node.id);
        let borderColor = color;
        let borderWidth = 2;
        if (matched) {
            // Prefer color of first token that touches any incident matched edge.
            borderColor = GRAPH_TOKEN_PALETTE[0];
            for (const edge of view.edges || []) {
                if ((edge.from === node.id || edge.to === node.id) && bucket.edgeIds.has(edge.id)) {
                    borderColor = firstTokenColorForEdge(modal, edge.id);
                    break;
                }
            }
            borderWidth = 4;
        } else if (filtering) {
            borderWidth = 1;
        }
        return {
            id: node.id,
            label: caption,
            group: node.group,
            title: `${node.group || 'Node'}: ${caption}`,
            color: {
                background: color,
                border: borderColor,
                highlight: { background: color, border: '#ffffff' },
            },
            font: {
                color: filtering && !matched ? 'rgba(255,255,255,0.35)' : '#ffffff',
                size: 12,
                face: 'Inter, sans-serif',
                strokeWidth: 3,
                strokeColor: 'rgba(0,0,0,0.65)',
            },
            borderWidth,
            borderWidthSelected: 3,
            size: matched ? 30 : 26,
            opacity: filtering && !matched ? 0.35 : 1,
            shape: 'dot',
            _rawData: node,
        };
    }));

    const visEdges = new vis.DataSet(view.edges.map((edge) => {
        const isSpine = edge.role === 'spine';
        const chains = (edge.chain_ids || []).join(', ');
        const evidence = edge.properties && edge.properties.evidence ? edge.properties.evidence : '';
        const matched = filtering && bucket.edgeIds.has(edge.id);
        const chipColor = matched ? firstTokenColorForEdge(modal, edge.id) : null;

        let edgeColor;
        let width;
        let label = edge.label;
        if (filtering && matched) {
            edgeColor = chipColor;
            width = 4;
        } else if (filtering) {
            edgeColor = 'rgba(255,255,255,0.06)';
            width = 1;
            label = '';
        } else {
            edgeColor = isSpine ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.22)';
            width = isSpine ? 3 : 1.4;
        }

        return {
            id: edge.id,
            from: edge.from,
            to: edge.to,
            label,
            title: `${edge.label} · ${chains}${evidence ? `\n${evidence}` : ''}`,
            dashes: !isSpine && !matched,
            width,
            color: {
                color: edgeColor,
                highlight: matched ? chipColor : 'rgba(255,255,255,0.85)',
                hover: matched ? chipColor : 'rgba(255,255,255,0.55)',
            },
            font: {
                color: matched ? '#e5e7eb' : '#9a9a9a',
                size: 10,
                face: 'Inter, sans-serif',
                strokeWidth: 2,
                strokeColor: 'rgba(0,0,0,0.5)',
                align: 'top',
            },
            arrows: { to: { enabled: true, scaleFactor: 0.55, type: 'arrow' } },
            smooth: { enabled: true, type: 'dynamic' },
            _rawData: edge,
        };
    }));

    modal.network = new vis.Network(modal.canvas, { nodes: visNodes, edges: visEdges }, {
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -90,
                centralGravity: 0.01,
                springLength: 180,
                springConstant: 0.08,
                damping: 0.4,
                avoidOverlap: 0.6,
            },
            stabilization: { iterations: 150, fit: true },
        },
        interaction: {
            hover: true,
            tooltipDelay: 150,
            zoomView: true,
            dragView: true,
            multiselect: false,
        },
        layout: { hierarchical: { enabled: false } },
    });

    modal.network.on('click', (params) => {
        if (params.nodes.length > 0) {
            const node = visNodes.get(params.nodes[0]);
            if (node && node._rawData) showGraphDetails('node', node._rawData);
        } else if (params.edges.length > 0) {
            const edge = visEdges.get(params.edges[0]);
            if (edge && edge._rawData) showGraphDetails('edge', edge._rawData);
        }
    });

    modal.network.once('stabilizationIterationsDone', () => {
        const focusId = modal.focusEdgeId;
        modal.focusEdgeId = null;
        if (focusId) {
            const edge = visEdges.get(focusId);
            if (edge) {
                const focusNode = edge.from || edge.to;
                if (focusNode) {
                    modal.network.focus(focusNode, {
                        scale: 1.25,
                        animation: { duration: 350, easingFunction: 'easeInOutQuad' },
                    });
                }
                modal.network.selectEdges([focusId]);
                if (edge._rawData) showGraphDetails('edge', edge._rawData);
                return;
            }
        }
        modal.network.fit({ animation: { duration: 350, easingFunction: 'easeInOutQuad' } });
    });
}

function graphPropsTable(properties) {
    const rows = Object.entries(properties || {})
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => {
            const rendered = typeof value === 'object' ? JSON.stringify(value) : String(value);
            return `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(rendered)}</td></tr>`;
        })
        .join('');
    return rows ? `<table class="node-props-table">${rows}</table>` : '';
}

function showGraphDetails(kind, data) {
    const modal = ensureGraphModal();
    const details = modal.details;
    const isNode = kind === 'node';
    const color = isNode ? (data.color || GRAPH_DEFAULT_NODE_COLOR) : '#8e8e8e';
    const title = isNode ? (data.caption || data.label || 'Node') : (data.label || 'Relationship');
    const properties = { ...(data.properties || {}) };

    if (!isNode) {
        properties.role = data.role || '';
        properties.chain_ids = (data.chain_ids || []).join(', ');
    }

    details.innerHTML = `
        <div class="node-details-header">
            <span class="node-details-title">${escapeHtml(title)}</span>
            <button class="node-details-close" type="button" aria-label="Закрыть">×</button>
        </div>
        <div class="node-label-badges">
            <span class="node-label-badge" style="border-color: ${color}; background: ${color}22;">${escapeHtml(isNode ? (data.group || 'Node') : (data.label || 'REL'))}</span>
            ${!isNode && data.role ? `<span class="node-label-badge graph-role-badge">${escapeHtml(data.role)}</span>` : ''}
        </div>
        ${graphPropsTable(properties)}`;
    details.style.display = 'block';
    details.querySelector('.node-details-close').addEventListener('click', () => {
        details.style.display = 'none';
    });
}

// ===== UI Helpers =====

function renderMarkdown(text) {
    if (typeof marked === 'undefined') {
        return escapeHtml(text);
    }
    const raw = marked.parse(text, { breaks: true, gfm: true });
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(raw);
    }
    return raw;
}

function addCopyButton(wrapper, plainText) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.title = 'Скопировать ответ';
    btn.setAttribute('aria-label', 'Скопировать ответ');
    btn.innerHTML = `
        <svg class="copy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <svg class="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span class="copy-label">Копировать</span>`;

    btn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(plainText);
            btn.classList.add('copied');
            btn.querySelector('.copy-icon').style.display = 'none';
            btn.querySelector('.check-icon').style.display = 'block';
            btn.querySelector('.copy-label').textContent = 'Скопировано';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.querySelector('.copy-icon').style.display = 'block';
                btn.querySelector('.check-icon').style.display = 'none';
                btn.querySelector('.copy-label').textContent = 'Копировать';
            }, 1600);
        } catch (err) {
            console.error('Clipboard error:', err);
            btn.querySelector('.copy-label').textContent = 'Ошибка';
            setTimeout(() => {
                btn.querySelector('.copy-label').textContent = 'Копировать';
            }, 1600);
        }
    });

    wrapper.appendChild(btn);
}

function addMessage(role, text, graphMeta = null) {
    // Убираем welcome-сообщение при первом реальном сообщении
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (role === 'assistant') {
        div.innerHTML = `
            <div class="assistant-avatar">${ASSISTANT_ICON}</div>
            <div class="message-content-wrapper">
                <div class="message-header">
                    <span class="message-label">Ассистент</span>
                </div>
                <div class="message-text markdown-body">${renderMarkdown(text)}</div>
            </div>`;

        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const contentWrapper = div.querySelector('.message-content-wrapper');
        addCopyButton(contentWrapper, text);
        addGraphButton(contentWrapper, graphMeta);
    } else if (role === 'user') {
        div.innerHTML = `<span class="message-text">${escapeHtml(text)}</span>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } else {
        // System / Errors
        div.innerHTML = `<span class="message-text">${escapeHtml(text)}</span>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function showThinking() {
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = 'thinking';
    div.id = 'thinking-indicator';
    div.innerHTML = `
        <div class="assistant-avatar">${ASSISTANT_ICON}</div>
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeThinking() {
    const el = document.getElementById('thinking-indicator');
    if (el) el.remove();
}

function setUIState(state) {
    currentUIState = state;
    isProcessing = state === 'processing';

    // Mic button
    micBtn.classList.toggle('recording', state === 'recording');
    micBtn.disabled = state === 'processing';

    // Управление иконками: микрофон, квадрат (стоп), пауза
    if (state === 'recording') {
        micIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
    } else if (state === 'playing') {
        micIcon.style.display = 'none';
        stopIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
    } else {
        // idle или processing
        micIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        pauseIcon.style.display = 'none';
    }

    // Text input
    textInput.disabled = isProcessing;
    sendBtn.disabled = isProcessing;

    // Status text
    const statusMap = {
        idle: 'Подключено',
        recording: '🔴 Запись...',
        processing: '⏳ Анализ...',
        playing: '🔊 Озвучивание...',
    };
    statusText.textContent = statusMap[state] || 'Подключено';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function safeDecodeHeader(value) {
    if (!value) return '';
    try {
        return decodeURIComponent(value);
    } catch {
        return value;
    }
}

// ===== Health Check =====

async function checkHealth() {
    try {
        const resp = await fetch('/health', { signal: AbortSignal.timeout(3000) });
        const ok = resp.ok;
        connectionDot.classList.toggle('offline', !ok);
        connectionDot.title = ok ? 'Сервер подключен' : 'Сервер недоступен';
        statusText.textContent = ok ? (currentUIState === 'idle' ? 'Подключено' : statusText.textContent) : 'Сервер недоступен';
    } catch {
        connectionDot.classList.add('offline');
        connectionDot.title = 'Сервер недоступен';
        statusText.textContent = 'Сервер недоступен';
    }
}

// ===== Clear Context / Reset History =====

async function clearHistory() {
    if (isProcessing) return;

    // Останавливаем проигрывание и прерываем текущий запрос
    stopPlayback();

    try {
        const response = await fetch('/clear_history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            // Очищаем историю на экране и восстанавливаем приветственный экран
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-logo"><img src="icon.svg" alt="" width="40" height="40"></div>
                    <h2>Чем я могу помочь?</h2>
                    <p>Задайте вопрос голосом или текстом. Я проанализирую базу знаний Neo4j и предоставлю структурированный ответ с голосовой озвучкой.</p>
                </div>`;
            setUIState('idle');
        } else {
            addMessage('system', '⚠️ Не удалось сбросить историю на сервере.');
        }
    } catch (err) {
        console.error('Clear history error:', err);
        addMessage('system', '⚠️ Ошибка соединения с сервером при попытке сбросить историю.');
    }
}

// ===== Init =====

// Подключение кнопок управления микрофоном и текстом
micBtn.addEventListener('click', toggleMic);
sendBtn.addEventListener('click', sendText);
clearBtn.addEventListener('click', clearHistory);
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendText();
    }
});

// Проверяем здоровье сервера при загрузке и периодически
checkHealth();
setInterval(checkHealth, 30000);
