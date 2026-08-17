/**
 * Neo4j Assistant — Web Client
 *
 * Текстовый чат по SSE. Аудио (STT/TTS) опционально и скрыто, пока audio_enabled=false.
 */

const ASSISTANT_ICON = '<img src="icon.svg" alt="" width="32" height="32">';
const WELCOME_COPY =
    'Задайте вопрос по базе знаний Neo4j — отвечу с опорой на граф и источники.';
const TOOL_LABELS = {
    ask_subgraph: 'Поиск в графе',
};

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
const sendIcon = document.getElementById('send-icon');
const stopGenIcon = document.getElementById('stop-gen-icon');
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const stopIcon = document.getElementById('stop-icon');
const pauseIcon = document.getElementById('pause-icon');
const statusText = document.getElementById('db-status-text');
const connectionDot = document.getElementById('connection-dot');
const clearBtn = document.getElementById('clear-btn');
const effortPicker = document.getElementById('effort-picker');
const effortBtn = document.getElementById('effort-btn');
const effortMenu = document.getElementById('effort-menu');
const effortLabel = document.getElementById('effort-label');
const depthPicker = document.getElementById('depth-picker');
const depthBtn = document.getElementById('depth-btn');
const depthMenu = document.getElementById('depth-menu');
const depthLabel = document.getElementById('depth-label');

const EFFORT_STORAGE_KEY = 'reasoning_effort';
const EFFORT_OPTIONS = {
    low: { label: 'Низкий' },
    medium: { label: 'Средний' },
    xhigh: { label: 'Максимум' },
};
const DEPTH_STORAGE_KEY = 'search_depth';
const DEPTH_OPTIONS = {
    low: { label: 'Узко' },
    medium: { label: 'Обычно' },
    high: { label: 'Широко' },
};
let currentSearchDepth = 'medium';
let currentReasoningEffort = 'xhigh';
let reasoningEffortEnabled = true;
let audioEnabled = false;
let activeStreamShell = null;

function getSessionId() {
    if (!window.__assistantSessionId) {
        window.__assistantSessionId = (crypto.randomUUID && crypto.randomUUID()) ||
            'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
                const r = (Math.random() * 16) | 0;
                const v = c === 'x' ? r : (r & 0x3) | 0x8;
                return v.toString(16);
            });
    }
    return window.__assistantSessionId;
}

function withSessionHeaders(headers) {
    return Object.assign({ 'X-Session-Id': getSessionId() }, headers || {});
}

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
function abortActiveRequest() {
    if (!currentAbortController) return;
    try {
        currentAbortController.abort();
    } catch (_) {}
}

function stopPlayback() {
    abortActiveRequest();
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

        addMessage('user', recognizedText);

        // Same SSE as text input so done.graph_run_id can show the graph button.
        await consumeProcessTextStream(recognizedText, currentAbortController.signal);
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

function onComposerSubmit() {
    if (currentUIState === 'processing') {
        abortActiveRequest();
        return;
    }
    sendText();
}

async function sendText() {
    const text = textInput.value.trim();
    if (!text || isProcessing) return;

    addMessage('user', text);
    textInput.value = '';
    resizeTextInput();
    setUIState('processing');

    activeSources.forEach((source) => {
        try { source.stop(); } catch (_) {}
    });
    activeSources = [];
    nextStartTime = 0;
    if (audioCtx) {
        try { audioCtx.close(); } catch (_) {}
        audioCtx = null;
    }
    currentAbortController = new AbortController();

    try {
        await consumeProcessTextStream(text, currentAbortController.signal);
    } catch (err) {
        if (err.name === 'AbortError') return;
        console.error('Send text error:', err);
        setUIState('idle');
    }
}

/**
 * SSE: thinking / tool_call / tool_result / content / content_rewind / done / error
 */
async function consumeProcessTextStream(text, signal) {
    const myController = currentAbortController;
    const shell = createAssistantStreamShell();
    startProcessTraceTimer(shell);

    const scheduleMarkdown = throttle(() => {
        renderStreamingAnswer(shell, { streaming: true });
        scrollChatIfPinned();
    }, 50);

    const handleEvent = (event, data) => {
        if (shell.finished) return;

        if (event === 'thinking') {
            const delta = data.delta || '';
            if (!delta) return;
            ensureProcessTrace(shell);
            ensureThinkingBlock(shell);
            shell.thinkingText += delta;
            shell.thinkingBody.textContent = shell.thinkingText;
            shell.thinkingBody.scrollTop = shell.thinkingBody.scrollHeight;
            if (shell.runningTools === 0) shell.tracePhase = 'think';
            refreshProcessTraceLabel(shell);
        } else if (event === 'tool_call') {
            ensureProcessTrace(shell);
            beginToolStep(shell);
            shell.runningTools += 1;
            shell.tracePhase = 'tool';
            upsertToolCard(shell, data, 'running');
            refreshProcessTraceLabel(shell);
        } else if (event === 'tool_result') {
            ensureProcessTrace(shell);
            shell.runningTools = Math.max(0, shell.runningTools - 1);
            if (shell.runningTools === 0) shell.tracePhase = 'think';
            upsertToolCard(shell, data, data.ok === false ? 'error' : 'done');
            refreshProcessTraceLabel(shell);
        } else if (event === 'content') {
            const delta = data.delta || '';
            if (!delta) return;
            shell.answerText += delta;
            shell.answerEl.classList.add('streaming');
            scheduleMarkdown();
        } else if (event === 'content_rewind') {
            const rewind = data.text || '';
            if (rewind && shell.answerText.endsWith(rewind)) {
                shell.answerText = shell.answerText.slice(0, -rewind.length);
            } else if (rewind) {
                const idx = shell.answerText.lastIndexOf(rewind);
                if (idx !== -1) {
                    shell.answerText =
                        shell.answerText.slice(0, idx) +
                        shell.answerText.slice(idx + rewind.length);
                }
            }
            if (!shell.answerText) {
                shell.answerEl.classList.remove('streaming');
                shell.answerEl.innerHTML = '';
            } else {
                scheduleMarkdown();
            }
        } else if (event === 'graph_highlight') {
            const runId = data.graph_run_id ? String(data.graph_run_id) : '';
            if (!runId) return;
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
            finishStreamShell(shell, {
                status: 'done',
                finalContent: data.final_content,
                graphMeta: graphMetaFromDone(data),
            });
            scrollChatIfPinned();
        } else if (event === 'error') {
            finishStreamShell(shell, {
                status: 'error',
                message: data.message || 'Ошибка стрима',
            });
        }
    };

    try {
        const response = await fetch('/process_text_stream', {
            method: 'POST',
            headers: withSessionHeaders({
                'Content-Type': 'application/json',
                Accept: 'text/event-stream',
            }),
            body: JSON.stringify(processTextPayload(text)),
            signal,
        });

        if (!response.ok) {
            let errMsg = 'Ошибка сервера';
            try {
                const errData = await response.json();
                errMsg = errData.error || errMsg;
            } catch (_) {}
            finishStreamShell(shell, { status: 'error', message: errMsg });
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                buffer += decoder.decode();
                break;
            }
            buffer += decoder.decode(value, { stream: true });

            let sep;
            while ((sep = buffer.indexOf('\n\n')) !== -1) {
                const rawEvent = buffer.slice(0, sep);
                buffer = buffer.slice(sep + 2);
                const parsed = parseSseEvent(rawEvent);
                if (!parsed) continue;
                handleEvent(parsed.event, parsed.data);
            }
        }
        if (buffer.trim()) {
            const parsed = parseSseEvent(buffer);
            if (parsed) handleEvent(parsed.event, parsed.data);
        }
        if (!shell.finished) {
            finishStreamShell(shell, {
                status: shell.answerText.trim() ? 'done' : 'error',
                message: 'Поток оборвался',
            });
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            if (!shell.finished) {
                finishStreamShell(shell, { status: 'aborted' });
            }
            throw err;
        }
        if (!shell.finished) {
            finishStreamShell(shell, {
                status: 'error',
                message: 'Ошибка соединения с сервером',
            });
        }
        throw err;
    } finally {
        stopProcessTraceTimer(shell);
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

function welcomeMarkup() {
    return `
                <div class="welcome-message">
                    <div class="welcome-logo"><img src="icon.svg" alt="" width="40" height="40"></div>
                    <h2>Чем я могу помочь?</h2>
                    <p>${WELCOME_COPY}</p>
                </div>`;
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
            <div class="message-text markdown-body" aria-live="polite"></div>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const contentWrapper = div.querySelector('.message-content-wrapper');
    const shell = {
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
        finished: false,
        tracePhase: 'think',
        traceDone: false,
        userOpenedTrace: false,
        timerId: null,
        runningTools: 0,
    };
    activeStreamShell = shell;
    ensureProcessTrace(shell);
    return shell;
}

function ensureProcessTrace(shell) {
    if (shell.processTrace) return;
    const details = document.createElement('details');
    details.className = 'process-trace';
    details.open = false;
    details.innerHTML = `
        <summary class="process-trace-summary">
            <span class="chevron"></span>
            <span class="process-trace-pulse" aria-hidden="true"></span>
            <span class="process-trace-label">Думаю…</span>
        </summary>
        <div class="process-trace-body"></div>`;
    details.addEventListener('toggle', () => {
        if (details.open) shell.userOpenedTrace = true;
    });
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

function elapsedSeconds(shell) {
    return Math.max(0, Math.round((Date.now() - shell.startedAt) / 1000));
}

function refreshProcessTraceLabel(shell) {
    if (!shell.processLabel || shell.traceDone) return;
    const secs = elapsedSeconds(shell);
    const verb =
        shell.runningTools > 0 || shell.tracePhase === 'tool'
            ? 'Ищу в графе'
            : 'Думаю';
    shell.processLabel.textContent = secs > 0 ? `${verb} · ${secs} с` : `${verb}…`;
}

function startProcessTraceTimer(shell) {
    refreshProcessTraceLabel(shell);
    if (shell.timerId) return;
    shell.timerId = setInterval(() => refreshProcessTraceLabel(shell), 1000);
}

function stopProcessTraceTimer(shell) {
    if (shell.timerId) {
        clearInterval(shell.timerId);
        shell.timerId = null;
    }
}

function densifyCitations(text) {
    const sessionToDisplay = new Map();
    const displayId = (sid) => {
        if (!sessionToDisplay.has(sid)) {
            sessionToDisplay.set(sid, sessionToDisplay.size + 1);
        }
        return sessionToDisplay.get(sid);
    };
    let rendered = String(text || '').replace(
        /\(\s*source\s*:\s*(\d+(?:\s*,\s*\d+)*)\s*\)/gi,
        (_, ids) =>
            ids
                .split(',')
                .map((part) => {
                    const n = parseInt(part.trim(), 10);
                    return Number.isFinite(n) ? `[${displayId(n)}]` : '';
                })
                .join('')
    );
    rendered = rendered.replace(/(?<!\w)source\s*:?\s*(\d+)(?!\w)/gi, (_, n) => {
        const sid = parseInt(n, 10);
        return Number.isFinite(sid) ? `[${displayId(sid)}]` : '';
    });
    return rendered;
}

function holdIncompleteFence(text) {
    const src = String(text || '');
    let count = 0;
    let last = -1;
    let i = 0;
    while ((i = src.indexOf('```', i)) !== -1) {
        count += 1;
        last = i;
        i += 3;
    }
    if (count % 2 === 1 && last >= 0) return src.slice(0, last);
    return src;
}

function renderStreamingAnswer(shell, { streaming = false, cited = false } = {}) {
    let text = shell.answerText || '';
    if (!cited) {
        text = holdIncompleteFence(densifyCitations(text));
    }
    shell.answerEl.innerHTML = text ? renderMarkdown(text) : '';
    if (streaming && shell.answerText) {
        const caret = document.createElement('span');
        caret.className = 'stream-caret';
        caret.setAttribute('aria-hidden', 'true');
        shell.answerEl.appendChild(caret);
    }
}

function finishStreamShell(shell, { status, message, finalContent, graphMeta } = {}) {
    if (!shell || shell.finished) return;
    shell.finished = true;
    stopProcessTraceTimer(shell);
    shell.answerEl.classList.remove('streaming');
    const secs = Math.max(1, elapsedSeconds(shell));

    if (shell.processTrace) {
        const hasDetails = shell.processBody && shell.processBody.children.length > 0;
        if (status === 'done' && !hasDetails) {
            shell.processTrace.remove();
            shell.processTrace = null;
        } else {
            shell.traceDone = true;
            shell.processTrace.classList.toggle('is-done', status === 'done');
            shell.processTrace.classList.toggle('is-error', status === 'error');
            shell.processTrace.classList.toggle('is-aborted', status === 'aborted');
            if (status === 'done') {
                setProcessTraceLabel(shell, `Готово за ${secs} с`);
            } else if (status === 'aborted') {
                setProcessTraceLabel(shell, `Остановлено · ${secs} с`);
            } else {
                setProcessTraceLabel(shell, `Ошибка · ${secs} с`);
            }
            if (!shell.userOpenedTrace) shell.processTrace.open = false;
        }
    }

    if (status === 'aborted') {
        if (!shell.answerText.trim()) {
            shell.answerEl.innerHTML = '<p class="stream-status-note">Остановлено</p>';
        } else {
            renderStreamingAnswer(shell, { streaming: false });
            addCopyButton(shell.contentWrapper, shell.answerText);
        }
    } else if (status === 'error') {
        const note = message || 'Ошибка стрима';
        if (!shell.answerText.trim()) {
            shell.answerEl.innerHTML = `<p class="stream-status-note stream-status-error">${escapeHtml(note)}</p>`;
        } else {
            renderStreamingAnswer(shell, { streaming: false });
            addCopyButton(shell.contentWrapper, shell.answerText);
            const p = document.createElement('p');
            p.className = 'stream-status-note stream-status-error';
            p.textContent = note;
            shell.contentWrapper.appendChild(p);
        }
    } else {
        if (finalContent != null && finalContent !== '') {
            shell.answerText = finalContent;
        }
        renderStreamingAnswer(shell, { streaming: false, cited: true });
        addCopyButton(shell.contentWrapper, shell.answerText);
        addGraphButton(shell.contentWrapper, graphMeta);
    }

    if (activeStreamShell === shell) activeStreamShell = null;
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
        shell.thinkingCount === 1
            ? 'Рассуждение'
            : `Рассуждение ${shell.thinkingCount}`;
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
        beginToolStep(shell);
        card = document.createElement('details');
        card.className = 'tool-card';
        card.open = false;
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

    const displayName = TOOL_LABELS[data.name] || data.name || id;
    card.querySelector('.tool-card-name').textContent = displayName;
    const statusEl = card.querySelector('.tool-card-status');
    statusEl.className = `tool-card-status ${status}`;
    statusEl.textContent =
        status === 'running' ? 'идёт' : status === 'error' ? 'ошибка' : 'готово';

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

    if (status === 'running') {
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
const GRAPH_LEGEND = [
    { group: 'Microbe', label: 'Microbe' },
    { group: 'Metabolite', label: 'Metabolite' },
    { group: 'StarterCulture', label: 'StarterCulture' },
    { group: 'EnvironmentCondition', label: 'EnvironmentCondition' },
];
const GRAPH_TOKEN_PALETTE = [
    '#f59e0b',
    '#34d399',
    '#60a5fa',
    '#f472b6',
    '#a78bfa',
    '#fb7185',
];
const GRAPH_NETWORK_OPTIONS = {
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
        stabilization: { iterations: 120, fit: true },
    },
    interaction: {
        hover: true,
        tooltipDelay: 180,
        zoomView: true,
        dragView: true,
        multiselect: false,
    },
    layout: { hierarchical: { enabled: false } },
    edges: {
        smooth: { enabled: true, type: 'cubicBezier', roundness: 0.35 },
    },
};
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

    const bar = ensureMessageToolbar(contentWrapper);
    bar.querySelectorAll('.show-graph-btn').forEach((el) => el.remove());

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'show-graph-btn';
    btn.dataset.graphRunId = graphMeta.runId;
    const count = graphMeta.chainCount > 1 ? ` (${graphMeta.chainCount})` : '';
    btn.innerHTML = `
        <span class="spinner-small" style="display:none"></span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="6.5" cy="7" r="2.4"></circle>
            <circle cx="17.5" cy="6.5" r="2.4"></circle>
            <circle cx="8" cy="17.5" r="2.4"></circle>
            <circle cx="18" cy="16.5" r="2.4"></circle>
            <path d="M8.7 8.4 16 7.6M8.2 15.4 7.4 9.4M16.2 8.7 16.8 14.2M10.3 17.2 15.7 16.6"></path>
        </svg>
        <span class="show-graph-label">Показать граф${count}</span>`;
    btn.addEventListener('click', () => openGraphModal(graphMeta, btn));
    bar.appendChild(btn);
}

function normalizeQuery(s) {
    return String(s || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function nodeCaption(node) {
    const name = String((node && node.caption) || '').trim();
    if (name) return name;
    const group = String((node && node.group) || '').trim();
    if (group && group !== 'Unknown') return `Без имени · ${group}`;
    return 'Без имени';
}

function nodeRef(group, name) {
    const nm = String(name || '').trim();
    const grp = String(group || '').trim();
    if (grp && grp !== 'Unknown' && nm) return `${grp}: ${nm}`;
    return nm || nodeCaption({ caption: nm, group: grp });
}

function chainLabelForId(payload, chainId) {
    const view = ((payload && payload.views) || []).find((v) => v.id === chainId);
    return (view && view.label) || chainId;
}

function buildEdgeSearchIndex(payload) {
    const nodeById = new Map();
    const remember = (node) => {
        if (!node || !node.id || nodeById.has(node.id)) return;
        nodeById.set(node.id, node);
    };
    for (const view of (payload && payload.views) || []) {
        for (const node of view.nodes || []) remember(node);
    }
    for (const node of ((payload && payload.all && payload.all.nodes) || [])) {
        remember(node);
    }

    const chainBits = (chainIds) => {
        const bits = [];
        for (const cid of chainIds || []) {
            bits.push(String(cid));
            const label = chainLabelForId(payload, cid);
            bits.push(label);
            bits.push(normalizeQuery(label));
            const n = String(cid).replace(/^a/i, '');
            if (n) {
                bits.push(`цепь ${n}`);
                bits.push(`chain ${n}`);
            }
        }
        return bits;
    };

    const byId = new Map();
    const consider = (edge) => {
        if (!edge || !edge.id) return;
        const props = edge.properties || {};
        const fromNode = nodeById.get(edge.from);
        const toNode = nodeById.get(edge.to);
        const fromName = String(edge.from_name || (fromNode && fromNode.caption) || '').trim();
        const toName = String(edge.to_name || (toNode && toNode.caption) || '').trim();
        const fromGroup = String(edge.from_group || (fromNode && fromNode.group) || '').trim();
        const toGroup = String(edge.to_group || (toNode && toNode.group) || '').trim();
        const fromCaption = nodeCaption({ caption: fromName, group: fromGroup });
        const toCaption = nodeCaption({ caption: toName, group: toGroup });
        const evidence = String(props.evidence || '');
        const sourceFile = String(props.source_file || '');
        const label = String(edge.label || '');
        const chainIds = Array.isArray(edge.chain_ids) ? edge.chain_ids.slice() : [];
        const haystack = normalizeQuery(
            [
                evidence,
                label,
                fromName,
                toName,
                fromCaption,
                toCaption,
                fromGroup,
                toGroup,
                sourceFile,
                nodeRef(fromGroup, fromName),
                nodeRef(toGroup, toName),
                fromGroup && fromName ? `${fromGroup} ${fromName}` : '',
                toGroup && toName ? `${toGroup} ${toName}` : '',
                edge.hub_name || '',
                ...chainBits(chainIds),
            ].join(' ')
        );

        const existing = byId.get(edge.id);
        if (existing) {
            for (const cid of chainIds) {
                if (!existing.chainIds.includes(cid)) existing.chainIds.push(cid);
            }
            if (evidence && evidence.length > String(existing.evidence || '').length) {
                existing.evidence = evidence;
            }
            if (fromName && !existing.fromName) existing.fromName = fromName;
            if (toName && !existing.toName) existing.toName = toName;
            existing.fromCaption = existing.fromName
                ? nodeCaption({ caption: existing.fromName, group: existing.fromGroup })
                : existing.fromCaption;
            existing.toCaption = existing.toName
                ? nodeCaption({ caption: existing.toName, group: existing.toGroup })
                : existing.toCaption;
            existing.haystack = normalizeQuery(`${existing.haystack} ${haystack}`);
            return;
        }
        byId.set(edge.id, {
            edgeId: edge.id,
            haystack,
            chainIds,
            label,
            fromCaption,
            toCaption,
            fromName,
            toName,
            fromGroup,
            toGroup,
            evidence,
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

    const textTokens = active.filter((t) => !t.edgeId);
    const edgeTokens = active.filter((t) => t.edgeId);

    for (const token of edgeTokens) {
        addMatch(token, byEdgeId.get(token.edgeId));
    }
    if (textTokens.length) {
        for (const item of index) {
            const matching = textTokens.filter((t) =>
                item.haystack.includes(normalizeQuery(t.text))
            );
            if (matching.length !== textTokens.length) continue;
            for (const token of matching) addMatch(token, item);
        }
    }

    return { perView, all, active: true };
}

function suggestScore(item, q) {
    const from = normalizeQuery(item.fromCaption);
    const to = normalizeQuery(item.toCaption);
    const fromName = normalizeQuery(item.fromName);
    const toName = normalizeQuery(item.toName);
    const ev = normalizeQuery(item.evidence);
    const label = normalizeQuery(item.label);
    const names = [from, to, fromName, toName];
    if (names.some((n) => n === q)) return 0;
    if (names.some((n) => n.startsWith(q))) return 1;
    if (names.some((n) => n.includes(q))) return 2;
    if (ev.startsWith(q)) return 3;
    if (label === q || label.startsWith(q)) return 4;
    return 5;
}

function suggestEdges(index, query, limit) {
    const q = normalizeQuery(query);
    if (q.length < 2) return [];
    const max = limit || 8;
    return index
        .filter((item) => item.haystack.includes(q))
        .sort((a, b) => {
            const d = suggestScore(a, q) - suggestScore(b, q);
            if (d !== 0) return d;
            return String(a.fromCaption).localeCompare(String(b.fromCaption));
        })
        .slice(0, max);
}

function nextTokenColor(tokens) {
    return GRAPH_TOKEN_PALETTE[tokens.length % GRAPH_TOKEN_PALETTE.length];
}

function edgeTriple(item) {
    if (!item) return 'ребро';
    return `${item.fromCaption} —${item.label}→ ${item.toCaption}`;
}

function edgeTokenLabel(item) {
    const full = edgeTriple(item);
    const short = item && item.fromCaption ? item.fromCaption : (item && item.label) || 'ребро';
    return short.length > 28 ? `${short.slice(0, 26)}…` : short;
}

function makeToken(opts, tokens) {
    graphTokenSeq += 1;
    const text = String((opts && opts.text) || '').trim();
    return {
        id: `tok_${graphTokenSeq}`,
        text,
        title: (opts && opts.title) || text,
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

function unionViews(views) {
    const nodes = new Map();
    const edges = new Map();
    for (const view of views || []) {
        for (const node of view.nodes || []) {
            if (!nodes.has(node.id)) nodes.set(node.id, node);
        }
        for (const edge of view.edges || []) {
            if (!edges.has(edge.id)) edges.set(edge.id, edge);
        }
    }
    return { nodes: Array.from(nodes.values()), edges: Array.from(edges.values()) };
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

function graphTopologyKey(modal) {
    if (modal.viewIndex >= 0) return `v:${modal.viewIndex}`;
    if (modal.matches && modal.matches.active) {
        return `all:${visibleViewsForModal(modal).map((v) => v.id).join(',')}`;
    }
    return 'all';
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
                <input class="graph-modal-search-input" type="search" role="combobox" aria-autocomplete="list" aria-expanded="false" placeholder="Поиск по цитатам и рёбрам…" autocomplete="off" spellcheck="false">
                <div class="graph-modal-suggest" role="listbox" style="display:none"></div>
            </div>
            <div class="graph-modal-note" style="display:none"></div>
            <div class="graph-modal-toolbar">
                <button class="graph-modal-nav" type="button" data-nav="prev" aria-label="Предыдущая цепь">‹</button>
                <button class="graph-modal-all" type="button">Все</button>
                <button class="graph-modal-nav" type="button" data-nav="next" aria-label="Следующая цепь">›</button>
            </div>
            <div class="graph-modal-body">
                <div class="graph-modal-canvas-wrap">
                    <div class="graph-modal-canvas"></div>
                    <div class="graph-modal-empty" style="display:none"></div>
                    <div class="graph-modal-legend"></div>
                </div>
                <aside class="graph-modal-inspector"></aside>
            </div>
        </div>`;

    document.body.appendChild(overlay);

    graphModal = {
        overlay,
        canvas: overlay.querySelector('.graph-modal-canvas'),
        empty: overlay.querySelector('.graph-modal-empty'),
        details: overlay.querySelector('.graph-modal-inspector'),
        legend: overlay.querySelector('.graph-modal-legend'),
        subtitle: overlay.querySelector('.graph-modal-subtitle'),
        allBtn: overlay.querySelector('.graph-modal-all'),
        navBtns: overlay.querySelectorAll('.graph-modal-nav'),
        chipsEl: overlay.querySelector('.graph-modal-chips'),
        searchInput: overlay.querySelector('.graph-modal-search-input'),
        suggestEl: overlay.querySelector('.graph-modal-suggest'),
        noteEl: overlay.querySelector('.graph-modal-note'),
        network: null,
        visNodes: null,
        visEdges: null,
        payload: null,
        runId: null,
        viewIndex: -1,
        searchIndex: [],
        spec: { tokens: [], note: '' },
        matches: null,
        suggestTimer: null,
        suggestItems: [],
        suggestIndex: 0,
        focusEdgeId: null,
        selected: null,
        renderedKey: '',
    };

    renderGraphLegend(graphModal);
    showInspectorEmpty();

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
        graphModal.suggestTimer = setTimeout(() => renderGraphSuggestions(), 80);
    });
    graphModal.searchInput.addEventListener('keydown', (event) => {
        const open = graphModal.suggestEl.style.display !== 'none';
        const items = graphModal.suggestItems || [];
        if (event.key === 'ArrowDown' && open && items.length) {
            event.preventDefault();
            graphModal.suggestIndex = Math.min(graphModal.suggestIndex + 1, items.length - 1);
            highlightGraphSuggestion();
        } else if (event.key === 'ArrowUp' && open && items.length) {
            event.preventDefault();
            graphModal.suggestIndex = Math.max(graphModal.suggestIndex - 1, 0);
            highlightGraphSuggestion();
        } else if (event.key === 'Enter') {
            event.preventDefault();
            if (open && items.length) {
                pinGraphEdgeToken(items[graphModal.suggestIndex] || items[0]);
                graphModal.searchInput.value = '';
                hideGraphSuggestions();
                return;
            }
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
            if (open) {
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

function renderGraphLegend(modal) {
    modal.legend.innerHTML = GRAPH_LEGEND.map((item) => `
        <span class="graph-modal-legend-item">
            <span class="graph-modal-legend-dot" style="background:${GRAPH_NODE_COLORS[item.group]}"></span>
            ${escapeHtml(item.label)}
        </span>`).join('');
}

function renderGraphChips() {
    const modal = ensureGraphModal();
    modal.chipsEl.innerHTML = '';
    for (const token of modal.spec.tokens) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'graph-modal-chip';
        chip.title = token.title || token.text || 'Убрать фильтр';
        chip.innerHTML = `
            <span class="graph-modal-chip-dot" style="background:${token.color}"></span>
            <span class="graph-modal-chip-text">${escapeHtml(token.text)}</span>
            <span class="graph-modal-chip-x" aria-hidden="true">×</span>`;
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
    modal.suggestItems = [];
    modal.suggestIndex = 0;
    modal.searchInput.setAttribute('aria-expanded', 'false');
}

function highlightGraphSuggestion() {
    const modal = ensureGraphModal();
    const buttons = modal.suggestEl.querySelectorAll('.graph-modal-suggest-item');
    buttons.forEach((btn, i) => {
        btn.classList.toggle('active', i === modal.suggestIndex);
    });
    const active = buttons[modal.suggestIndex];
    if (active && active.scrollIntoView) {
        active.scrollIntoView({ block: 'nearest' });
    }
}

function renderGraphSuggestions() {
    const modal = ensureGraphModal();
    const query = modal.searchInput.value.trim();
    if (!modal.payload || normalizeQuery(query).length < 2) {
        hideGraphSuggestions();
        return;
    }
    const items = suggestEdges(modal.searchIndex, query, 8);
    modal.suggestItems = items;
    modal.suggestIndex = 0;
    modal.suggestEl.style.display = 'block';
    modal.searchInput.setAttribute('aria-expanded', 'true');
    if (!items.length) {
        modal.suggestEl.innerHTML = '<div class="graph-modal-suggest-empty">Нет совпадений — Enter добавит текстовый фильтр</div>';
        return;
    }
    modal.suggestEl.innerHTML = items.map((item, i) => {
        const chains = (item.chainIds || [])
            .map((cid) => chainLabelForId(modal.payload, cid))
            .join(', ');
        return `
            <button type="button" class="graph-modal-suggest-item${i === 0 ? ' active' : ''}" data-edge-id="${escapeHtml(item.edgeId)}" role="option">
                <div class="graph-modal-suggest-title">${escapeHtml(edgeTriple(item))}</div>
                <div class="graph-modal-suggest-meta">${escapeHtml(item.evidence || '')}${chains ? ` · ${escapeHtml(chains)}` : ''}</div>
            </button>`;
    }).join('');
    modal.suggestEl.querySelectorAll('.graph-modal-suggest-item').forEach((btn, i) => {
        btn.addEventListener('mouseenter', () => {
            modal.suggestIndex = i;
            highlightGraphSuggestion();
        });
        btn.addEventListener('mousedown', (event) => {
            event.preventDefault();
            pinGraphEdgeToken(items[i]);
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
                { text: edgeTokenLabel(item), title: edgeTriple(item), edgeId: item.edgeId },
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

function resetGraphSpec(modal) {
    modal.spec = { tokens: [], note: '' };
    modal.matches = null;
    modal.selected = null;
    modal.focusEdgeId = null;
    modal.suggestItems = [];
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
        const stillVisible = current && visible.includes(current);
        if (!stillVisible) {
            modal.viewIndex = visible.length
                ? (modal.payload.views || []).indexOf(visible[0])
                : -1;
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

    if (modal.runId && modal.runId !== graphMeta.runId) {
        resetGraphSpec(modal);
        renderGraphChips();
        renderGraphNote();
    }

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
        modal.payload = null;
        modal.searchIndex = [];
        modal.renderedKey = '';
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
    destroyGraphNetwork(graphModal);
}

function destroyGraphNetwork(modal) {
    if (modal.network) {
        modal.network.destroy();
        modal.network = null;
    }
    modal.visNodes = null;
    modal.visEdges = null;
    modal.renderedKey = '';
}

function setGraphModalMessage(text, loading) {
    const modal = ensureGraphModal();
    destroyGraphNetwork(modal);
    modal.canvas.innerHTML = '';
    modal.empty.style.display = 'flex';
    modal.legend.style.display = 'none';
    modal.empty.innerHTML = loading
        ? '<span class="spinner-small"></span><span>' + escapeHtml(text) + '</span>'
        : escapeHtml(text);
    showInspectorEmpty();
    updateGraphToolbar();
}

function currentGraphView() {
    const modal = ensureGraphModal();
    if (!modal.payload) return null;
    const filtering = Boolean(modal.matches && modal.matches.active);
    if (filtering) {
        const visible = visibleViewsForModal(modal);
        if (!visible.length) return { nodes: [], edges: [] };
        if (modal.viewIndex === -1) return unionViews(visible);
        return (modal.payload.views || [])[modal.viewIndex] || { nodes: [], edges: [] };
    }
    if (modal.viewIndex === -1) return modal.payload.all;
    return (modal.payload.views || [])[modal.viewIndex] || null;
}

function stepGraphView(delta) {
    const modal = ensureGraphModal();
    const views = (modal.payload && modal.payload.views) || [];
    const visible = visibleViewsForModal(modal);
    if (!visible.length) {
        modal.viewIndex = -1;
        renderGraphView();
        return;
    }

    const indices = visible
        .map((v) => views.indexOf(v))
        .filter((i) => i >= 0);
    if (!indices.length) {
        modal.viewIndex = -1;
        renderGraphView();
        return;
    }

    let pos = indices.indexOf(modal.viewIndex);
    if (pos < 0) {
        pos = delta > 0 ? -1 : 0;
    }
    const nextPos = (pos + delta + indices.length) % indices.length;
    modal.viewIndex = indices[nextPos];
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
            ? ` · ${chainCount} из ${allViews.length} · совпадений: ${matchCount}`
            : (allViews.length ? ` · ${allViews.length}` : '');
        modal.subtitle.textContent = `Все${suffix}`;
        modal.allBtn.classList.add('active');
    } else {
        const view = allViews[modal.viewIndex];
        const label = (view && view.label) || 'Цепь';
        const visiblePos = visible.indexOf(view) + 1;
        const matchPart = filtering ? ` · совпадений: ${matchCount}` : '';
        const scorePart = ` · score ${Number((view && view.score) || 0).toFixed(3)}`;
        let title = label;
        if (chainCount > 1 && visiblePos > 0) {
            title = (visiblePos === modal.viewIndex + 1)
                ? `${label} / ${chainCount}`
                : `${label} · ${visiblePos} / ${chainCount}`;
        }
        modal.subtitle.textContent = `${title}${matchPart}${scorePart}`;
        modal.allBtn.classList.remove('active');
    }
    modal.navBtns.forEach((btn) => { btn.disabled = visible.length < 2; });
}

function visNodeRecord(modal, node, view, filtering, bucket) {
    const color = node.color || GRAPH_NODE_COLORS[node.group] || GRAPH_DEFAULT_NODE_COLOR;
    const caption = nodeCaption(node);
    const matched = filtering && bucket.nodeIds.has(node.id);
    let borderColor = color;
    let borderWidth = 2;
    if (matched) {
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
    const group = node.group && node.group !== 'Unknown' ? node.group : '';
    return {
        id: node.id,
        label: caption,
        group: node.group,
        title: group ? `${group}: ${caption}` : caption,
        color: {
            background: color,
            border: borderColor,
            highlight: { background: color, border: '#ffffff' },
        },
        font: {
            color: filtering && !matched ? 'rgba(255,255,255,0.35)' : '#ffffff',
            size: 13,
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
}

function visEdgeRecord(modal, edge, view, filtering, bucket) {
    const isSpine = edge.role === 'spine';
    const fromCap = nodeCaption({
        caption: edge.from_name,
        group: edge.from_group,
    });
    const toCap = nodeCaption({
        caption: edge.to_name,
        group: edge.to_group,
    });
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
        edgeColor = 'rgba(255,255,255,0.08)';
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
        title: `${fromCap} —${edge.label}→ ${toCap}${evidence ? `\n${evidence}` : ''}`,
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
        _rawData: edge,
    };
}

function bindGraphNetworkEvents(modal) {
    modal.network.on('click', (params) => {
        if (params.nodes.length > 0) {
            const node = modal.visNodes.get(params.nodes[0]);
            if (node && node._rawData) showInspector('node', node._rawData);
        } else if (params.edges.length > 0) {
            const edge = modal.visEdges.get(params.edges[0]);
            if (edge && edge._rawData) showInspector('edge', edge._rawData);
        } else {
            modal.network.unselectAll();
            showInspectorEmpty();
        }
    });
}

function freezeGraphPhysics(modal) {
    if (!modal.network) return;
    modal.network.setOptions({ physics: { enabled: false } });
}

function afterGraphStabilize(modal) {
    freezeGraphPhysics(modal);
    const focusId = modal.focusEdgeId;
    modal.focusEdgeId = null;
    if (focusId && modal.visEdges && modal.visEdges.get(focusId)) {
        focusGraphEdge(modal, focusId);
        return;
    }
    restoreGraphSelection(modal);
    modal.network.fit({ animation: { duration: 280, easingFunction: 'easeInOutQuad' } });
}

function focusGraphEdge(modal, edgeId) {
    if (!modal.network || !modal.visEdges) return;
    const edge = modal.visEdges.get(edgeId);
    if (!edge) return;
    modal.network.selectEdges([edgeId]);
    const focusNode = edge.from || edge.to;
    if (focusNode) {
        modal.network.focus(focusNode, {
            scale: 1.2,
            animation: { duration: 280, easingFunction: 'easeInOutQuad' },
        });
    }
    if (edge._rawData) showInspector('edge', edge._rawData);
}

function restoreGraphSelection(modal) {
    const selected = modal.selected;
    if (!selected || !modal.visNodes || !modal.visEdges) {
        showInspectorEmpty();
        return;
    }
    if (selected.kind === 'node') {
        const node = modal.visNodes.get(selected.id);
        if (node && node._rawData) {
            modal.network.selectNodes([selected.id]);
            showInspector('node', node._rawData);
            return;
        }
    }
    if (selected.kind === 'edge') {
        const edge = modal.visEdges.get(selected.id);
        if (edge && edge._rawData) {
            modal.network.selectEdges([selected.id]);
            showInspector('edge', edge._rawData);
            return;
        }
    }
    showInspectorEmpty();
}

function renderGraphView() {
    const modal = ensureGraphModal();
    const view = currentGraphView();
    updateGraphToolbar();

    if (!view || !view.nodes || !view.nodes.length) {
        setGraphModalMessage(
            (modal.matches && modal.matches.active)
                ? 'Нет совпадений'
                : 'Нет данных для визуализации',
            false
        );
        return;
    }

    modal.empty.style.display = 'none';
    modal.legend.style.display = 'flex';

    const bucket = currentMatchBucket(modal) || emptyMatchBucket();
    const filtering = Boolean(modal.matches && modal.matches.active);
    const topologyKey = graphTopologyKey(modal);
    const nodeRecords = view.nodes.map((node) => visNodeRecord(modal, node, view, filtering, bucket));
    const edgeRecords = view.edges.map((edge) => visEdgeRecord(modal, edge, view, filtering, bucket));

    const canUpdateInPlace = Boolean(
        modal.network &&
        modal.visNodes &&
        modal.visEdges &&
        modal.renderedKey === topologyKey
    );

    if (canUpdateInPlace) {
        modal.visNodes.update(nodeRecords);
        modal.visEdges.update(edgeRecords);
        if (modal.focusEdgeId) {
            const id = modal.focusEdgeId;
            modal.focusEdgeId = null;
            focusGraphEdge(modal, id);
        } else {
            restoreGraphSelection(modal);
        }
        return;
    }

    if (modal.network) {
        modal.network.destroy();
        modal.network = null;
    }
    modal.canvas.innerHTML = '';

    modal.visNodes = new vis.DataSet(nodeRecords);
    modal.visEdges = new vis.DataSet(edgeRecords);
    modal.renderedKey = topologyKey;
    modal.network = new vis.Network(
        modal.canvas,
        { nodes: modal.visNodes, edges: modal.visEdges },
        {
            ...GRAPH_NETWORK_OPTIONS,
            physics: { ...GRAPH_NETWORK_OPTIONS.physics, enabled: true },
        }
    );
    bindGraphNetworkEvents(modal);
    modal.network.once('stabilizationIterationsDone', () => afterGraphStabilize(modal));
}

function showInspectorEmpty() {
    const modal = ensureGraphModal();
    modal.selected = null;
    modal.details.innerHTML = '<div class="graph-inspector-empty">Выберите узел или ребро</div>';
}

function incidentEdgeCount(view, nodeId) {
    return (view.edges || []).filter((e) => e.from === nodeId || e.to === nodeId).length;
}

function formatConfidence(value) {
    if (value === null || value === undefined || value === '') return '—';
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(2) : '—';
}

function formatChainList(payload, chainIds) {
    return (chainIds || []).map((cid) => chainLabelForId(payload, cid)).join(', ');
}

function showInspector(kind, data) {
    const modal = ensureGraphModal();
    const view = currentGraphView() || { nodes: [], edges: [] };
    modal.selected = { kind, id: data.id };

    if (kind === 'node') {
        const caption = nodeCaption(data);
        const group = data.group && data.group !== 'Unknown' ? data.group : '';
        const color = data.color || GRAPH_NODE_COLORS[group] || GRAPH_DEFAULT_NODE_COLOR;
        const community = data.properties && data.properties.leiden_community;
        const degree = incidentEdgeCount(view, data.id);
        modal.details.innerHTML = `
            <div class="graph-inspector-kicker">Узел</div>
            <div class="graph-inspector-title">${escapeHtml(caption)}</div>
            <div class="graph-inspector-badges">
                ${group ? `<span class="graph-inspector-badge" style="border-color:${color}">${escapeHtml(group)}</span>` : ''}
            </div>
            <dl class="graph-inspector-meta">
                <div class="graph-inspector-row"><dt>Рёбра</dt><dd>${degree}</dd></div>
                ${community !== undefined && community !== null && community !== ''
                    ? `<div class="graph-inspector-row"><dt>Community</dt><dd>${escapeHtml(String(community))}</dd></div>`
                    : ''}
            </dl>`;
        return;
    }

    const fromCap = nodeCaption({ caption: data.from_name, group: data.from_group });
    const toCap = nodeCaption({ caption: data.to_name, group: data.to_group });
    const fromRef = nodeRef(data.from_group, data.from_name || fromCap);
    const toRef = nodeRef(data.to_group, data.to_name || toCap);
    const evidence = data.properties && data.properties.evidence ? String(data.properties.evidence) : '';
    const sourceFile = data.properties && data.properties.source_file ? String(data.properties.source_file) : '';
    const confidence = data.properties ? data.properties.confidence : null;
    const role = data.role === 'spine' ? 'остов' : data.role === 'fan' ? 'луч' : (data.role || '');
    const chains = formatChainList(modal.payload, data.chain_ids);
    const hub = data.hub_name ? String(data.hub_name) : '';

    modal.details.innerHTML = `
        <div class="graph-inspector-kicker">Ребро</div>
        <div class="graph-inspector-title">${escapeHtml(data.label || 'RELATED')}</div>
        <div class="graph-inspector-triple">${escapeHtml(fromRef)} —${escapeHtml(data.label || '')}→ ${escapeHtml(toRef)}</div>
        <div class="graph-inspector-badges">
            ${role ? `<span class="graph-inspector-badge">${escapeHtml(role)}</span>` : ''}
        </div>
        ${evidence ? `<div class="graph-inspector-quote">${escapeHtml(evidence)}</div>` : ''}
        ${evidence ? `<button type="button" class="graph-inspector-copy">Копировать цитату</button>` : ''}
        <dl class="graph-inspector-meta">
            ${sourceFile ? `<div class="graph-inspector-row"><dt>Источник</dt><dd>${escapeHtml(sourceFile)}</dd></div>` : ''}
            <div class="graph-inspector-row"><dt>Conf</dt><dd>${escapeHtml(formatConfidence(confidence))}</dd></div>
            ${chains ? `<div class="graph-inspector-row"><dt>Цепи</dt><dd>${escapeHtml(chains)}</dd></div>` : ''}
            ${hub ? `<div class="graph-inspector-row"><dt>Хаб</dt><dd>${escapeHtml(hub)}</dd></div>` : ''}
        </dl>`;

    const copyBtn = modal.details.querySelector('.graph-inspector-copy');
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(evidence);
                copyBtn.textContent = 'Скопировано';
                setTimeout(() => { copyBtn.textContent = 'Копировать цитату'; }, 1400);
            } catch (_) {
                copyBtn.textContent = 'Ошибка';
                setTimeout(() => { copyBtn.textContent = 'Копировать цитату'; }, 1400);
            }
        });
    }
}

// ===== UI Helpers =====

function renderMarkdown(text) {
    if (typeof marked === 'undefined') {
        return escapeHtml(text);
    }
    const raw = marked.parse(text, { breaks: true, gfm: true });
    const html = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(raw) : raw;
    return wrapSourcesBlock(wrapMarkdownTables(html));
}

function wrapMarkdownTables(html) {
    const holder = document.createElement('div');
    holder.innerHTML = html;
    holder.querySelectorAll('table').forEach((table) => {
        if (table.parentElement && table.parentElement.classList.contains('md-table-scroll')) {
            return;
        }
        const wrap = document.createElement('div');
        wrap.className = 'md-table-scroll';
        table.parentNode.insertBefore(wrap, table);
        wrap.appendChild(table);
    });
    return holder.innerHTML;
}

function wrapSourcesBlock(html) {
    const holder = document.createElement('div');
    holder.innerHTML = html;
    const headings = [...holder.querySelectorAll('h3')];
    const src = headings.find((h) => h.textContent.trim() === 'Источники');
    if (!src) return holder.innerHTML;
    const wrap = document.createElement('div');
    wrap.className = 'md-sources';
    src.parentNode.insertBefore(wrap, src);
    wrap.appendChild(src);
    let next = wrap.nextSibling;
    while (next) {
        const tag = next.nodeType === 1 ? next.tagName : '';
        if (tag === 'H1' || tag === 'H2' || tag === 'H3') break;
        const keep = next.nextSibling;
        wrap.appendChild(next);
        next = keep;
    }
    return holder.innerHTML;
}

function ensureMessageToolbar(wrapper) {
    let bar = wrapper.querySelector('.message-toolbar');
    if (bar) return bar;
    bar = document.createElement('div');
    bar.className = 'message-toolbar';
    wrapper.appendChild(bar);
    return bar;
}

function addCopyButton(wrapper, plainText) {
    const bar = ensureMessageToolbar(wrapper);
    bar.querySelectorAll('.copy-btn').forEach((el) => el.remove());
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

    bar.appendChild(btn);
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
    const stopping = state === 'processing';

    micBtn.classList.toggle('recording', state === 'recording');
    micBtn.disabled = stopping || !audioEnabled;

    if (state === 'recording') {
        micIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
    } else if (state === 'playing') {
        micIcon.style.display = 'none';
        stopIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
    } else {
        micIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        pauseIcon.style.display = 'none';
    }

    textInput.disabled = false;
    sendBtn.disabled = false;
    sendBtn.classList.toggle('stopping', stopping);
    if (sendIcon) sendIcon.style.display = stopping ? 'none' : 'block';
    if (stopGenIcon) stopGenIcon.style.display = stopping ? 'block' : 'none';
    sendBtn.title = stopping ? 'Остановить' : 'Отправить';
    sendBtn.setAttribute('aria-label', stopping ? 'Остановить' : 'Отправить');

    const statusMap = {
        idle: 'Подключено',
        recording: 'Запись…',
        processing: 'Анализ…',
        playing: 'Озвучивание…',
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
    if (activeStreamShell) {
        activeStreamShell.finished = true;
        stopProcessTraceTimer(activeStreamShell);
        activeStreamShell = null;
    }
    abortActiveRequest();
    stopPlayback();

    try {
        const response = await fetch('/clear_history', {
            method: 'POST',
            headers: withSessionHeaders({ 'Content-Type': 'application/json' })
        });

        if (response.ok) {
            chatMessages.innerHTML = welcomeMarkup();
            setUIState('idle');
        } else {
            addMessage('system', '⚠️ Не удалось сбросить историю на сервере.');
        }
    } catch (err) {
        console.error('Clear history error:', err);
        addMessage('system', '⚠️ Ошибка соединения с сервером при попытке сбросить историю.');
    }
}

// ===== Reasoning effort / search depth pickers =====

function processTextPayload(text) {
    const payload = { text, search_depth: currentSearchDepth };
    if (reasoningEffortEnabled) {
        payload.reasoning_effort = currentReasoningEffort;
    }
    return payload;
}

function isEffortMenuOpen() {
    return effortBtn.getAttribute('aria-expanded') === 'true';
}

function closeEffortMenu() {
    effortBtn.setAttribute('aria-expanded', 'false');
    effortMenu.hidden = true;
}

function openEffortMenu() {
    effortBtn.setAttribute('aria-expanded', 'true');
    effortMenu.hidden = false;
}

function toggleEffortMenu() {
    if (isEffortMenuOpen()) closeEffortMenu();
    else openEffortMenu();
}

function setReasoningEffort(effort, persist = true) {
    if (!EFFORT_OPTIONS[effort]) return;
    currentReasoningEffort = effort;
    effortLabel.textContent = EFFORT_OPTIONS[effort].label;
    effortBtn.title = `Глубина рассуждения: ${EFFORT_OPTIONS[effort].label}`;
    effortBtn.setAttribute(
        'aria-label',
        `Глубина рассуждения: ${EFFORT_OPTIONS[effort].label}`
    );
    effortMenu.querySelectorAll('.effort-option').forEach((btn) => {
        btn.setAttribute('aria-selected', btn.dataset.effort === effort ? 'true' : 'false');
    });
    if (persist) {
        try {
            localStorage.setItem(EFFORT_STORAGE_KEY, effort);
        } catch (_) {}
    }
}

function loadStoredEffort() {
    try {
        const stored = localStorage.getItem(EFFORT_STORAGE_KEY);
        if (stored && EFFORT_OPTIONS[stored]) return stored;
    } catch (_) {}
    return null;
}

function isDepthMenuOpen() {
    return depthBtn.getAttribute('aria-expanded') === 'true';
}

function closeDepthMenu() {
    depthBtn.setAttribute('aria-expanded', 'false');
    depthMenu.hidden = true;
}

function openDepthMenu() {
    depthBtn.setAttribute('aria-expanded', 'true');
    depthMenu.hidden = false;
}

function toggleDepthMenu() {
    if (isDepthMenuOpen()) closeDepthMenu();
    else openDepthMenu();
}

function setSearchDepth(depth, persist = true) {
    if (!DEPTH_OPTIONS[depth]) return;
    currentSearchDepth = depth;
    depthLabel.textContent = DEPTH_OPTIONS[depth].label;
    depthBtn.title = `Глубина поиска: ${DEPTH_OPTIONS[depth].label}`;
    depthBtn.setAttribute('aria-label', `Глубина поиска: ${DEPTH_OPTIONS[depth].label}`);
    depthMenu.querySelectorAll('.effort-option').forEach((btn) => {
        btn.setAttribute('aria-selected', btn.dataset.depth === depth ? 'true' : 'false');
    });
    if (persist) {
        try {
            localStorage.setItem(DEPTH_STORAGE_KEY, depth);
        } catch (_) {}
    }
}

function loadStoredDepth() {
    try {
        const stored = localStorage.getItem(DEPTH_STORAGE_KEY);
        if (stored && DEPTH_OPTIONS[stored]) return stored;
    } catch (_) {}
    return null;
}

async function initComposerControls() {
    const storedEffort = loadStoredEffort();
    const storedDepth = loadStoredDepth();
    let serverDefault = 'xhigh';
    let serverDepth = 'medium';
    let thinkEnabled = true;
    try {
        const resp = await fetch('/ui_config', { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
            const data = await resp.json();
            thinkEnabled = data.think !== false;
            audioEnabled = data.audio_enabled === true;
            if (data.reasoning_effort && EFFORT_OPTIONS[data.reasoning_effort]) {
                serverDefault = data.reasoning_effort;
            }
            if (data.search_depth && DEPTH_OPTIONS[data.search_depth]) {
                serverDepth = data.search_depth;
            }
        }
    } catch (_) {}

    reasoningEffortEnabled = thinkEnabled;
    if (effortPicker) {
        effortPicker.hidden = !thinkEnabled;
    }
    if (micBtn) {
        micBtn.hidden = !audioEnabled;
    }
    setReasoningEffort(storedEffort || serverDefault, false);
    setSearchDepth(storedDepth || serverDepth, false);
}

// ===== Init =====

// Подключение кнопок управления микрофоном и текстом
micBtn.addEventListener('click', toggleMic);
sendBtn.addEventListener('click', onComposerSubmit);
clearBtn.addEventListener('click', clearHistory);
effortBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    closeDepthMenu();
    toggleEffortMenu();
});
effortMenu.addEventListener('click', (e) => {
    const option = e.target.closest('.effort-option');
    if (!option) return;
    setReasoningEffort(option.dataset.effort);
    closeEffortMenu();
});
depthBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    closeEffortMenu();
    toggleDepthMenu();
});
depthMenu.addEventListener('click', (e) => {
    const option = e.target.closest('.effort-option');
    if (!option) return;
    setSearchDepth(option.dataset.depth);
    closeDepthMenu();
});
document.addEventListener('click', (e) => {
    if (!effortPicker.contains(e.target)) closeEffortMenu();
    if (!depthPicker.contains(e.target)) closeDepthMenu();
});
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (isEffortMenuOpen()) {
        closeEffortMenu();
        effortBtn.focus();
    }
    if (isDepthMenuOpen()) {
        closeDepthMenu();
        depthBtn.focus();
    }
});
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendText();
    }
});
textInput.addEventListener('input', resizeTextInput);

function resizeTextInput() {
    textInput.style.height = 'auto';
    const max = parseFloat(getComputedStyle(textInput).maxHeight) || 208;
    const next = Math.min(textInput.scrollHeight, max);
    textInput.style.height = `${next}px`;
    textInput.style.overflowY = textInput.scrollHeight > max + 1 ? 'auto' : 'hidden';
}

initComposerControls();
resizeTextInput();

// Проверяем здоровье сервера при загрузке и периодически
checkHealth();
setInterval(checkHealth, 30000);
