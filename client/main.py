import asyncio
import logging
import sys
import threading
from urllib.parse import unquote

import httpx
import keyboard

from client.config import ClientConfig, load_config
from client.player import StreamingPlayer
from client.recorder import AudioRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def read_console(text_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """Фоновый поток для чтения текстового ввода из консоли."""
    while True:
        try:
            line = input()
            if line.strip():
                loop.call_soon_threadsafe(text_queue.put_nowait, line.strip())
        except EOFError:
            break


async def send_audio(
    config: ClientConfig, wav_bytes: bytes, player: StreamingPlayer
) -> None:
    """Отправляет аудио на сервер и воспроизводит ответ."""
    try:
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            async with client.stream(
                "POST",
                f"{config.server_url}/process",
                content=wav_bytes,
                headers={"Content-Type": "audio/wav"},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error(
                        f"Сервер вернул ошибку {resp.status_code}: {body.decode()}"
                    )
                    return

                recognized = unquote(resp.headers.get("Recognized-Text", ""))
                llm_response = unquote(resp.headers.get("LLM-Response", ""))

                if recognized:
                    logger.info(f"Ты: {recognized}")
                if llm_response:
                    logger.info(f"Ассистент: {llm_response}")

                await player.play_stream(
                    resp.aiter_bytes(4096),
                    should_stop=lambda: keyboard.is_pressed(config.push_to_talk_key),
                )

    except httpx.ConnectError:
        logger.error(f"Не удалось подключиться к серверу {config.server_url}")
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {e}", exc_info=True)


async def send_text(config: ClientConfig, text: str, player: StreamingPlayer) -> None:
    """Отправляет текст на сервер и воспроизводит ответ."""
    try:
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            async with client.stream(
                "POST",
                f"{config.server_url}/process_text",
                json={"text": text},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error(
                        f"Сервер вернул ошибку {resp.status_code}: {body.decode()}"
                    )
                    return

                llm_response = unquote(resp.headers.get("LLM-Response", ""))
                if llm_response:
                    logger.info(f"Ассистент: {llm_response}")

                await player.play_stream(
                    resp.aiter_bytes(4096),
                    should_stop=lambda: keyboard.is_pressed(config.push_to_talk_key),
                )

    except httpx.ConnectError:
        logger.error(f"Не удалось подключиться к серверу {config.server_url}")
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}", exc_info=True)


async def wait_for_server(
    server_url: str, retries: int = 30, delay: float = 2.0
) -> None:
    """Ожидает готовности сервера перед стартом основного цикла."""
    logger.info(f"Ожидание сервера {server_url}...")
    async with httpx.AsyncClient(timeout=5) as client:
        for attempt in range(1, retries + 1):
            try:
                resp = await client.get(f"{server_url}/health")
                if resp.status_code == 200:
                    logger.info("Сервер готов!")
                    return
            except httpx.RequestError:
                pass
            logger.debug(f"Попытка {attempt}/{retries}...")
            await asyncio.sleep(delay)

    raise ConnectionError(f"Сервер {server_url} не отвечает после {retries} попыток")


async def main() -> None:
    config = load_config()

    if config.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)

    await wait_for_server(config.server_url)

    recorder = AudioRecorder(config.push_to_talk_key, config.sample_rate)
    player = StreamingPlayer(config.playback_sample_rate)

    text_queue: asyncio.Queue[str] = asyncio.Queue()

    if config.enable_text_input:
        loop = asyncio.get_event_loop()
        thread = threading.Thread(
            target=read_console, args=(text_queue, loop), daemon=True
        )
        thread.start()

    msg_parts = []
    if config.enable_voice_input:
        msg_parts.append(f"голоса (Удерживай '{config.push_to_talk_key}')")
    if config.enable_text_input:
        msg_parts.append("текста (пиши и жми Enter)")

    msg = f"Ожидание {' или '.join(msg_parts)}..."
    logger.info(f"Готов. {msg}")

    while True:
        try:
            if config.enable_text_input and not text_queue.empty():
                text = text_queue.get_nowait()
                logger.info(f"Введён текст: {text}")
                logger.info("Отправка запроса...")
                await send_text(config, text, player)
                logger.info(msg)
                continue

            if config.enable_voice_input and keyboard.is_pressed(recorder.ptt_key):
                wav_bytes = await recorder.record()
                if wav_bytes:
                    logger.info("Отправка аудио запроса...")
                    await send_audio(config, wav_bytes, player)
                    logger.info(msg)
                continue

            await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Ошибка в цикле клиента: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
    except ConnectionError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
