import logging
import os

from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> None:
    """
    Настраивает OpenTelemetry + OpenAI auto-instrumentation.

    Вызывать ОДИН РАЗ при старте сервера, ДО создания AsyncOpenAI клиентов.
    Если PHOENIX_COLLECTOR_ENDPOINT не задан — трейсинг не активируется,
    приложение работает как раньше.
    """
    global _initialized
    if _initialized:
        return

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        logger.info("PHOENIX_COLLECTOR_ENDPOINT не задан — трейсинг отключён")
        return

    resource = Resource.create({"service.name": "voice-assistant"})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument(tracer_provider=provider)

    _initialized = True
    logger.info(f"OpenTelemetry трейсинг включён → {endpoint}")


OI_SPAN_KIND = "openinference.span.kind"
OI_INPUT_VALUE = "input.value"
OI_OUTPUT_VALUE = "output.value"


class OISpanKind:
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"


def set_span_ok(span, output_value: str = None) -> None:
    """Отмечает спан успешным и опционально записывает вывод."""
    span.set_status(Status(StatusCode.OK))
    if output_value is not None:
        span.set_attribute(OI_OUTPUT_VALUE, str(output_value))


def set_span_error(span, error_msg: str) -> None:
    """Отмечает спан ошибочным с сообщением."""
    span.set_status(Status(StatusCode.ERROR, error_msg))
    span.set_attribute(OI_OUTPUT_VALUE, str(error_msg))


def get_tracer(name: str) -> trace.Tracer:
    """Возвращает tracer для создания ручных спанов."""
    return trace.get_tracer(name)
