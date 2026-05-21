/**
 * Neo4j Voice Assistant — Web Client (Minimal ChatGPT Style)
 *
 * Логика: Toggle-микрофон (с паузой при воспроизведении), текстовый ввод, отправка на сервер,
 * потоковое воспроизведение PCM int16 24kHz ответа через Web Audio API с поддержкой Barge-in.
 */

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

        // Читаем ответ LLM из заголовка
        const llmResponse = safeDecodeHeader(response.headers.get('LLM-Response'));
        if (llmResponse) addMessage('assistant', llmResponse);

        // Воспроизводим аудио-ответ потоково
        setUIState('playing');
        
        const reader = response.body.getReader();
        let leftoverBytes = null;
        const myController = currentAbortController;

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

            // Выравнивание чанка по границе 2 байт (для Int16)
            if (data.length % 2 !== 0) {
                leftoverBytes = data.slice(data.length - 1);
                data = data.slice(0, data.length - 1);
            }

            if (data.length > 0) {
                const pcmChunk = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                await playChunk(pcmChunk);
            }
        }

        // Ожидаем проигрывание всех запланированных чанков
        if (audioCtx && nextStartTime > audioCtx.currentTime) {
            const delay = (nextStartTime - audioCtx.currentTime) * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));
        }

        // Если за это время не пришел новый прерывающий запрос
        if (currentAbortController === myController) {
            setUIState('idle');
            currentAbortController = null;
        }
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
    showThinking();

    // Перед отправкой нового запроса прерываем предыдущее воспроизведение
    stopPlayback();
    currentAbortController = new AbortController();

    try {
        const response = await fetch('/process_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
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

        const llmResponse = safeDecodeHeader(response.headers.get('LLM-Response'));
        if (llmResponse) addMessage('assistant', llmResponse);

        // Воспроизводим аудио-ответ потоково
        setUIState('playing');
        
        const reader = response.body.getReader();
        let leftoverBytes = null;
        const myController = currentAbortController;

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

            // Выравнивание чанка по границе 2 байт (для Int16)
            if (data.length % 2 !== 0) {
                leftoverBytes = data.slice(data.length - 1);
                data = data.slice(0, data.length - 1);
            }

            if (data.length > 0) {
                const pcmChunk = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                await playChunk(pcmChunk);
            }
        }

        // Ожидаем проигрывание всех запланированных чанков
        if (audioCtx && nextStartTime > audioCtx.currentTime) {
            const delay = (nextStartTime - audioCtx.currentTime) * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));
        }

        // Если за это время не пришел новый прерывающий запрос
        if (currentAbortController === myController) {
            setUIState('idle');
            currentAbortController = null;
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Fetch aborted.');
            return;
        }
        console.error('Send text error:', err);
        removeThinking();
        addMessage('system', '⚠️ Ошибка соединения с сервером');
        setUIState('idle');
    }
}

// ===== UI Helpers =====

function addMessage(role, text) {
    // Убираем welcome-сообщение при первом реальном сообщении
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (role === 'assistant') {
        div.innerHTML = `
            <div class="assistant-avatar">🧬</div>
            <div class="message-content-wrapper">
                <span class="message-label">Ассистент</span>
                <span class="message-text">${escapeHtml(text)}</span>
            </div>`;
    } else if (role === 'user') {
        div.innerHTML = `<span class="message-text">${escapeHtml(text)}</span>`;
    } else {
        // System / Errors
        div.innerHTML = `<span class="message-text">${escapeHtml(text)}</span>`;
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showThinking() {
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = 'thinking';
    div.id = 'thinking-indicator';
    div.innerHTML = `
        <div class="assistant-avatar">🧬</div>
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

// ===== Init =====

// Подключение кнопок управления микрофоном и текстом
micBtn.addEventListener('click', toggleMic);
sendBtn.addEventListener('click', sendText);
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendText();
    }
});

// Проверяем здоровье сервера при загрузке и периодически
checkHealth();
setInterval(checkHealth, 30000);
