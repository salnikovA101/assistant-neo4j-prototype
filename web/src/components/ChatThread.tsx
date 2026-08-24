import { useEffect, useRef } from "react";
import type { ChatMessage, ChatStep } from "../types";
import { prettyJson, renderMarkdown } from "../format";
import { IconGraph } from "./Icons";

function stepsOf(msg: ChatMessage): ChatStep[] {
  if (msg.steps?.length) return msg.steps;
  const out: ChatStep[] = [];
  if (msg.thinking) out.push({ kind: "think", text: msg.thinking });
  for (const tool of msg.tools || []) out.push({ kind: "tool", ...tool });
  return out;
}

function streamLabel(msg: ChatMessage): string {
  const steps = stepsOf(msg);
  if (steps.some((step) => step.kind === "tool" && step.status === "running")) {
    return "поиск в графе…";
  }
  if (msg.text) return "пишет…";
  return "думает…";
}

function thinkIndex(steps: ChatStep[], index: number): number {
  return steps.slice(0, index + 1).filter((step) => step.kind === "think").length;
}

function TraceBundle({
  steps,
  status,
  hasAnswer,
}: {
  steps: ChatStep[];
  status: ChatMessage["status"];
  hasAnswer: boolean;
}) {
  if (!steps.length) return null;
  const thinkCount = steps.filter((step) => step.kind === "think").length;
  const isLive = status === "streaming" && !hasAnswer;

  return (
    <details className="trace-bundle" open={isLive}>
      <summary>
        Ход работы <span>{steps.length}</span>
      </summary>
      <div className="trace-bundle-content">
        {steps.map((step, index) =>
          step.kind === "think" ? (
            <section key={`think-${index}`} className="trace-entry">
              <p className="trace-entry-title">
                Рассуждение{thinkCount > 1 ? ` ${thinkIndex(steps, index)}` : ""}
              </p>
              <pre>{step.text}</pre>
            </section>
          ) : (
            <section key={step.id} className={`trace-entry trace-entry-${step.status}`}>
              <p className="trace-entry-title">
                Вызов {step.name || "ask_subgraph"} ·{" "}
                {step.status === "running"
                  ? "идёт"
                  : step.status === "error"
                    ? "ошибка"
                    : "готово"}
              </p>
              {step.args != null && step.args !== "" && (
                <div>
                  <p className="tool-kicker">Аргументы</p>
                  <pre>{prettyJson(step.args)}</pre>
                </div>
              )}
              {step.result ? (
                <div>
                  <p className="tool-kicker">Результат</p>
                  <pre>{step.result}</pre>
                </div>
              ) : (
                step.status === "running" && <p className="muted">Ждём ответ графа…</p>
              )}
            </section>
          )
        )}
      </div>
    </details>
  );
}

export function ChatThread({
  messages,
  onOpenGraph,
  openGraphId,
}: {
  messages: ChatMessage[];
  onOpenGraph: (runId: string, chains: number) => void;
  openGraphId?: string | null;
}) {
  const threadRef = useRef<HTMLDivElement>(null);
  const followTailRef = useRef(true);
  const frameRef = useRef<number | null>(null);
  const isStreaming = messages.some((message) => message.status === "streaming");

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread || (!isStreaming && !followTailRef.current)) return;
    if (isStreaming) followTailRef.current = true;
    if (frameRef.current != null) return;

    const easeToTail = () => {
      const target = Math.max(0, thread.scrollHeight - thread.clientHeight);
      const delta = target - thread.scrollTop;
      if (delta <= 1) {
        thread.scrollTop = target;
        frameRef.current = null;
        return;
      }
      thread.scrollTop += Math.max(1, delta * 0.28);
      frameRef.current = requestAnimationFrame(easeToTail);
    };
    frameRef.current = requestAnimationFrame(easeToTail);
  }, [isStreaming, messages]);

  useEffect(
    () => () => {
      if (frameRef.current != null) cancelAnimationFrame(frameRef.current);
    },
    []
  );

  if (!messages.length) return null;

  return (
    <div
      ref={threadRef}
      className="thread"
      onScroll={(event) => {
        if (isStreaming) return;
        const { clientHeight, scrollHeight, scrollTop } = event.currentTarget;
        followTailRef.current = scrollHeight - scrollTop - clientHeight < 56;
      }}
    >
      {messages.map((msg) => {
        const steps = stepsOf(msg);
        return (
          <article key={msg.id} className={`bubble bubble-${msg.role}`}>
            {msg.role === "assistant" && (
              <div className="bubble-kicker">
                <span className="assistant-avatar" aria-hidden="true">N</span>
                <span>Neo4j Assistant</span>
                {msg.status === "streaming" && <span className="stream-state"><i />{streamLabel(msg)}</span>}
                {msg.status === "done" && msg.elapsedSec != null && (
                  <span>готово за {msg.elapsedSec} с</span>
                )}
                {msg.status === "aborted" && <span>остановлено</span>}
                {msg.status === "error" && <span className="is-error">не удалось ответить</span>}
              </div>
            )}
            {msg.role === "assistant" && (
              <TraceBundle steps={steps} status={msg.status} hasAnswer={Boolean(msg.text)} />
            )}
            {msg.text ? (
              <div
                className="md"
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(msg.text, msg.status === "streaming"),
                }}
              />
            ) : (
              msg.status === "streaming" &&
              !steps.some((step) => step.kind === "tool" && step.status === "running") &&
              !steps.some((step) => step.kind === "think") && (
                <p className="muted">Думает…</p>
              )
            )}
            {msg.graphRunId && msg.status === "done" && openGraphId !== msg.graphRunId && (
              <button
                type="button"
                className="graph-open"
                onClick={() => onOpenGraph(msg.graphRunId!, msg.graphChainCount || 1)}
              >
                <IconGraph /> Показать граф
                {msg.graphChainCount && msg.graphChainCount > 1 ? ` (${msg.graphChainCount})` : ""}
              </button>
            )}
          </article>
        );
      })}
    </div>
  );
}
