import { useEffect, useRef, useState } from "react";
import type { ChatMessage, ChatStep, PendingApproval } from "../types";
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

function ApprovalCard({
  approval,
  busy,
  onResolve,
}: {
  approval: PendingApproval;
  busy: boolean;
  onResolve: (action: "approve" | "revise", sqs: string[], feedback?: string) => void;
}) {
  const initial = approval.toolCall.arguments?.subquestions || [];
  const [items, setItems] = useState(initial.map((text) => ({ text, enabled: true })));
  const [feedback, setFeedback] = useState("");
  useEffect(() => {
    const revised = approval.toolCall.arguments?.subquestions || [];
    setItems(revised.map((text) => ({ text, enabled: true })));
    setFeedback("");
  }, [approval.revision, approval.toolCall]);
  return (
    <section className="approval-card">
      <header><strong>План поиска требует подтверждения</strong><span>revision {approval.revision}</span></header>
      <p>Проверьте SQ. По подтверждённым open SQ будет получено не более одного нового UNIT.</p>
      <div className="approval-sqs">
        {items.map((item, index) => (
          <label key={index}>
            <input
              type="checkbox"
              checked={item.enabled}
              onChange={(event) => setItems((prev) => prev.map((value, i) => i === index ? { ...value, enabled: event.target.checked } : value))}
            />
            <input
              value={item.text}
              onChange={(event) => setItems((prev) => prev.map((value, i) => i === index ? { ...value, text: event.target.value } : value))}
            />
          </label>
        ))}
      </div>
      <textarea value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="Что изменить, если план нужно отклонить" />
      <footer>
        <button type="button" className="primary-btn" disabled={busy || !items.some((item) => item.enabled && item.text.trim())} onClick={() => onResolve("approve", items.filter((item) => item.enabled).map((item) => item.text.trim()))}>Подтвердить</button>
        <button type="button" className="ghost-btn danger-btn" disabled={busy} onClick={() => onResolve("revise", [], feedback)}>Отклонить</button>
      </footer>
    </section>
  );
}

export function ChatThread({
  messages,
  onOpenGraph,
  openGraphId,
  pendingApproval,
  approvalBusy,
  onResolveApproval,
  onCheckpoint,
  onFork,
  onSaveCard,
}: {
  messages: ChatMessage[];
  onOpenGraph: (runId: string, chains: number) => void;
  openGraphId?: string | null;
  pendingApproval?: PendingApproval | null;
  approvalBusy?: boolean;
  onResolveApproval: (action: "approve" | "revise", sqs: string[], feedback?: string) => void;
  onCheckpoint: (checkpointId: string) => void;
  onFork: (checkpointId: string) => void;
  onSaveCard: (draftId: string, title: string) => void;
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
            {msg.checkpointId && (
              <div className="checkpoint-actions">
                <button type="button" onClick={() => onCheckpoint(msg.checkpointId!)}>checkpoint</button>
                <button type="button" onClick={() => onFork(msg.checkpointId!)}>fork отсюда</button>
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
            {msg.cardDraft && (
              <section className="chat-card-draft">
                <header>
                  <div>
                    <strong>{msg.cardTemplateName || "Карточка"}</strong>
                    <span>{msg.cardDraft.status === "saved" ? "сохранено в библиотеку" : "DRAFT · проверьте перед сохранением"}</span>
                  </div>
                  {msg.cardDraft.status !== "saved" && (
                    <button
                      type="button"
                      className="primary-btn"
                      onClick={() => onSaveCard(
                        msg.cardDraft!.id,
                        String(msg.cardDraft!.data.title || msg.cardTemplateName || "Карточка")
                      )}
                    >
                      Сохранить
                    </button>
                  )}
                </header>
                <pre>{prettyJson(msg.cardDraft.data)}</pre>
                {msg.cardDraft.gaps.length > 0 && <p>GAPS: {msg.cardDraft.gaps.map(String).join(" · ")}</p>}
              </section>
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
            {pendingApproval?.assistantMessageId === msg.id && pendingApproval.status === "pending" && (
              <ApprovalCard
                approval={pendingApproval}
                busy={Boolean(approvalBusy)}
                onResolve={onResolveApproval}
              />
            )}
          </article>
        );
      })}
      {pendingApproval?.status === "pending" && !messages.some((msg) => msg.id === pendingApproval.assistantMessageId) && (
        <article className="bubble bubble-assistant">
          <ApprovalCard
            approval={pendingApproval}
            busy={Boolean(approvalBusy)}
            onResolve={onResolveApproval}
          />
        </article>
      )}
    </div>
  );
}
