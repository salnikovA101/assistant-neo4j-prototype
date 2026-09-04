import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import type { AgendaItem, CardDraft, CardTemplate, ChatMessage, ChatStep, PendingApproval, TurnFailure } from "../types";
import { prettyJson, renderAnswerMarkdown, renderMarkdown, renderReasoningMarkdown } from "../format";
import { blankData, fallbackSchemaForData } from "../cardModel";
import { IconFork, IconGraph } from "./Icons";
import { CardVisual } from "./CardVisual";

function templateForVersion(templates: CardTemplate[], versionId: string): CardTemplate | undefined {
  return templates.find((template) => template.latestVersion.id === versionId);
}

function userEditedProvenance(provenance: Record<string, unknown>, key: string): Record<string, unknown> {
  const pointer = `/${key.replaceAll("~", "~0").replaceAll("/", "~1")}`;
  const next = Object.fromEntries(Object.entries(provenance).filter(([existing]) => existing !== pointer && !existing.startsWith(`${pointer}/`)));
  next[pointer] = [{ verification: "user-edited" }];
  return next;
}

function GrowingTextarea({
  value,
  onChange,
}: {
  value: string;
  onChange: (event: ChangeEvent<HTMLTextAreaElement>) => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = ref.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [value]);

  return (
    <textarea
      ref={ref}
      rows={1}
      value={value}
      onChange={onChange}
      aria-label="Пункт плана"
    />
  );
}

function InlineCardDraft({ draft, template, templateName, onSave }: {
  draft: CardDraft;
  template?: CardTemplate;
  templateName: string;
  onSave: (draftId: string, title: string, data: Record<string, unknown>, provenance: Record<string, unknown>, gaps: unknown[]) => void;
}) {
  const [data, setData] = useState(draft.data);
  const [provenance, setProvenance] = useState(draft.provenance);
  useEffect(() => { setData(draft.data); setProvenance(draft.provenance); }, [draft]);
  const schema = template?.latestVersion.schema || fallbackSchemaForData(data);
  const title = String(data.title || "").trim();
  const saved = draft.status === "saved";
  return (
    <section className="chat-card-draft">
      <CardVisual
        templateName={template?.name || templateName}
        version={template?.latestVersion.version}
        schema={schema}
        ui={template?.latestVersion.ui || {}}
        data={data}
        provenance={provenance}
        editable={!saved}
        status={saved ? "Сохранено" : "Черновик"}
        onChange={(key, value) => {
          setData((current) => ({ ...current, [key]: value }));
          setProvenance((current) => userEditedProvenance(current, key));
        }}
      />
      {!saved && <div className="chat-card-save"><span>{title ? "Проверьте ответы перед сохранением" : "Добавьте название карточки"}</span><button type="button" className="primary-btn" disabled={!title} onClick={() => onSave(draft.id, title, data, provenance, draft.gaps)}>Сохранить</button></div>}
    </section>
  );
}

function stepsOf(msg: ChatMessage): ChatStep[] {
  if (msg.steps?.length) return msg.steps;
  const out: ChatStep[] = [];
  if (msg.thinking) out.push({ kind: "think", text: msg.thinking });
  for (const tool of msg.tools || []) out.push({ kind: "tool", ...tool });
  return out;
}

function streamLabel(msg: ChatMessage): string {
  if (msg.cardDraft) return "оформляет карточку…";
  const steps = stepsOf(msg);
  const runningTool = steps.find((step) => step.kind === "tool" && step.status === "running");
  if (runningTool?.kind === "tool" && runningTool.name === "get_service_guide") {
    return "открывает помощь…";
  }
  if (runningTool) {
    return "поиск в базе…";
  }
  if (msg.text) return "пишет…";
  return "думает…";
}

function toolLabel(name: string): string {
  return name === "get_service_guide" ? "Помощь сервиса" : "Поиск в базе";
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
  const thinkCount = steps.filter((step) => step.kind === "think").length;
  const isLive = status === "streaming" && !hasAnswer;
  if (!steps.length && !isLive) return null;
  const lastThinkIndex = steps.reduce((last, step, index) => (step.kind === "think" ? index : last), -1);

  return (
    <details className="trace-bundle" open={isLive}>
      <summary>
        Журнал {steps.length > 0 && <span>{steps.length}</span>}
      </summary>
      <div className="trace-bundle-content">
        {!steps.length ? (
          <section className="trace-entry trace-entry-pending" aria-live="polite">
            <p className="trace-entry-title">Подготавливаю ответ</p>
            <p className="muted"><i className="trace-pending-dot" aria-hidden="true" />Собираю контекст и выбираю следующий шаг…</p>
          </section>
        ) : steps.map((step, index) =>
          step.kind === "think" ? (
            <section key={`think-${index}`} className="trace-entry">
              <p className="trace-entry-title">
                Размышление{thinkCount > 1 ? ` ${thinkIndex(steps, index)}` : ""}
              </p>
              <div
                className="trace-thinking md"
                dangerouslySetInnerHTML={{
                  __html: renderReasoningMarkdown(
                    step.text,
                    status === "streaming" && index === lastThinkIndex,
                  ),
                }}
              />
            </section>
          ) : (
            <section key={step.id} className={`trace-entry trace-entry-${step.status}`}>
              <p className="trace-entry-title">
                {toolLabel(step.name)} ·{" "}
                {step.status === "running"
                  ? "идёт"
                  : step.status === "error"
                    ? "ошибка"
                    : "готово"}
              </p>
              {step.name !== "get_service_guide" && step.args != null && step.args !== "" && (
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
                step.status === "running" && (
                  <p className="muted">
                    {step.name === "get_service_guide" ? "Загружаем помощь…" : "Ждём данные из базы…"}
                  </p>
                )
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
  agenda,
  busy,
  onResolve,
}: {
  approval: PendingApproval;
  agenda: AgendaItem[];
  busy: boolean;
  onResolve: (
    action: "approve" | "revise" | "cancel",
    selection: { openSqRefs: string[]; newSubquestions: string[] },
    feedback?: string
  ) => void;
}) {
  const args = approval.toolCall.arguments || {};
  const initialRefs = args.open_sq_refs || args.open_sq_ids || [];
  const initialNew = args.new_subquestions || args.subquestions || [];
  const [selectedRefs, setSelectedRefs] = useState<string[]>(initialRefs);
  const [items, setItems] = useState(initialNew.map((text) => ({ text, enabled: true })));
  const [feedback, setFeedback] = useState("");
  const [revisionOpen, setRevisionOpen] = useState(false);
  const selectedCount = selectedRefs.length + items.filter((item) => item.enabled && item.text.trim()).length;
  const directionLabel = (ref: string) => {
    const itemIndex = agenda.findIndex((value) => value.ref === ref);
    const match = ref.match(/:(\d+)$/);
    return `Пункт ${match ? Number(match[1]) : itemIndex + 1}`;
  };
  useEffect(() => {
    const revisedArgs = approval.toolCall.arguments || {};
    const revised = revisedArgs.new_subquestions || revisedArgs.subquestions || [];
    setSelectedRefs(revisedArgs.open_sq_refs || revisedArgs.open_sq_ids || []);
    setItems(revised.map((text) => ({ text, enabled: true })));
    setFeedback("");
    setRevisionOpen(false);
  }, [approval.revision, approval.toolCall]);
  return (
    <section className="approval-card">
      <header><strong>Что будем искать</strong><span>{selectedCount} из 5</span></header>
      <p>Отметьте пункты для этого поиска.</p>
      <div className="approval-sqs">
        {initialRefs.length > 0 && <h4>Уже в плане</h4>}
        {initialRefs.map((ref) => {
          const item = agenda.find((value) => value.ref === ref);
          return (
            <label key={ref}>
              <input
                type="checkbox"
                checked={selectedRefs.includes(ref)}
                onChange={(event) => setSelectedRefs((prev) => event.target.checked
                  ? [...prev, ref]
                  : prev.filter((value) => value !== ref))}
              />
              <span className="approval-existing"><strong>{directionLabel(ref)}</strong>{item?.text || "Пункт недоступен"}<small>В плане</small></span>
            </label>
          );
        })}
        {items.length > 0 && <h4>Новые пункты</h4>}
        {items.map((item, index) => (
          <label key={index}>
            <input
              type="checkbox"
              checked={item.enabled}
              onChange={(event) => setItems((prev) => prev.map((value, i) => i === index ? { ...value, enabled: event.target.checked } : value))}
            />
            <GrowingTextarea
              value={item.text}
              onChange={(event) => setItems((prev) => prev.map((value, i) => i === index ? { ...value, text: event.target.value } : value))}
            />
          </label>
        ))}
      </div>
      {revisionOpen ? (
        <div className="approval-revision">
          <label>
            <span>Что изменить в плане</span>
            <textarea
              autoFocus
              value={feedback}
              onChange={(event) => setFeedback(event.target.value)}
              placeholder="Например: объединить похожие пункты и добавить поиск по стабильности индикатора"
            />
          </label>
          <footer>
            <button
              type="button"
              className="primary-btn"
              disabled={busy || !feedback.trim()}
              onClick={() => onResolve("revise", { openSqRefs: [], newSubquestions: [] }, feedback.trim())}
            >Отправить</button>
            <button
              type="button"
              className="ghost-btn"
              disabled={busy}
              onClick={() => { setRevisionOpen(false); setFeedback(""); }}
            >Назад</button>
          </footer>
        </div>
      ) : (
        <footer>
          <button
            type="button"
            className="primary-btn"
            disabled={busy || selectedCount === 0 || selectedCount > 5}
            onClick={() => onResolve("approve", {
              openSqRefs: selectedRefs,
              newSubquestions: items.filter((item) => item.enabled).map((item) => item.text.trim()).filter(Boolean),
            })}
          >Запустить поиск</button>
          <button type="button" className="ghost-btn" disabled={busy} onClick={() => setRevisionOpen(true)}>Переделать план</button>
        </footer>
      )}
    </section>
  );
}

export function ChatThread({
  messages,
  cardTemplates,
  onOpenGraph,
  openGraphId,
  pendingApproval,
  agenda,
  approvalBusy,
  onResolveApproval,
  turnFailures = [],
  onDismissFailure,
  selectedMessageIds = [],
  onSaveCard,
  onFork,
  forkingCheckpointId = "",
}: {
  messages: ChatMessage[];
  cardTemplates: CardTemplate[];
  onOpenGraph: (checkpointId: string) => void;
  openGraphId?: string | null;
  pendingApproval?: PendingApproval | null;
  agenda: AgendaItem[];
  approvalBusy?: boolean;
  onResolveApproval: (
    action: "approve" | "revise" | "cancel",
    selection: { openSqRefs: string[]; newSubquestions: string[] },
    feedback?: string
  ) => void;
  turnFailures?: TurnFailure[];
  onDismissFailure?: (createdAt: number) => void;
  selectedMessageIds?: string[];
  onSaveCard: (draftId: string, title: string, data: Record<string, unknown>, provenance: Record<string, unknown>, gaps: unknown[]) => void;
  onFork: (checkpointId: string) => void;
  forkingCheckpointId?: string;
}) {
  const threadRef = useRef<HTMLDivElement>(null);
  const followTailRef = useRef(true);
  const frameRef = useRef<number | null>(null);
  const [openSourcesMessageId, setOpenSourcesMessageId] = useState<string | null>(null);
  const isStreaming = messages.some((message) => message.status === "streaming");

  useEffect(() => {
    if (!selectedMessageIds.length) return;
    const target = document.getElementById(`message-${selectedMessageIds[selectedMessageIds.length - 1]}`);
    target?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [selectedMessageIds]);

  useEffect(() => {
    if (!openSourcesMessageId) return;
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(`sources-${openSourcesMessageId}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [openSourcesMessageId]);

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

  if (!messages.length && !turnFailures.length) return null;

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
      {turnFailures.map((failure) => (
        <div key={failure.createdAt} className="turn-failure-chip" role="status">
          <span>
            {failure.reason === "cancelled"
              ? "Запрос отменён"
              : failure.reason === "aborted"
                ? "Запрос остановлен"
                : "Запрос не выполнен"}
            {failure.message ? ` · ${failure.message}` : ""}
          </span>
          <time dateTime={new Date(failure.createdAt).toISOString()}>
            {new Date(failure.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </time>
          {onDismissFailure && (
            <button type="button" onClick={() => onDismissFailure(failure.createdAt)}>Закрыть</button>
          )}
        </div>
      ))}
      {messages.map((msg) => {
        const steps = stepsOf(msg);
        const renderedAnswer = msg.role === "assistant" && msg.text
          ? renderAnswerMarkdown(msg.text, msg.status === "streaming")
          : null;
        return (
          <article
            id={`message-${msg.id}`}
            key={msg.id}
            className={`bubble bubble-${msg.role} ${selectedMessageIds.includes(msg.id) ? "is-context-step" : ""}`}
          >
            {msg.role === "assistant" && (
              <div className="bubble-kicker">
                <span className="assistant-avatar" aria-hidden="true">N</span>
                <span>Neo4j Assistant</span>
                {(msg.modelLabel || msg.modelId) && (
                  <span>{msg.modelLabel || msg.modelId}</span>
                )}
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
            {msg.role === "user" ? (
              <div className="user-message-content">
                {msg.text && (
                  <div
                    className="md"
                    dangerouslySetInnerHTML={{
                      __html: renderMarkdown(msg.text, false),
                    }}
                  />
                )}
                {msg.cardRequest && (
                  <div className="message-card-alias">
                    <CardVisual templateName={msg.cardRequest.templateName} version={msg.cardRequest.version} schema={msg.cardRequest.schema} ui={msg.cardRequest.ui || {}} data={blankData(msg.cardRequest.schema)} status="Запрос на создание" />
                  </div>
                )}
                {msg.cardReference && (
                  <div className="message-card-alias">{(() => {
                    const template = templateForVersion(cardTemplates, msg.cardReference!.templateVersionId || "");
                    return <CardVisual templateName={template?.name || "Карточка"} version={template?.latestVersion.version} schema={template?.latestVersion.schema || fallbackSchemaForData(msg.cardReference!.data)} ui={template?.latestVersion.ui || {}} data={{ ...msg.cardReference!.data, title: msg.cardReference!.title }} provenance={msg.cardReference!.provenance || {}} status={`Правка ${msg.cardReference!.revision}`} />;
                  })()}</div>
                )}
              </div>
            ) : msg.text ? (
              <div
                className="md"
                dangerouslySetInnerHTML={{
                  __html: renderedAnswer?.bodyHtml || "",
                }}
              />
            ) : (
              null
            )}
            {msg.role === "assistant" && msg.sqStatusWarning && (
              <p className="sq-status-warning">{msg.sqStatusWarning}</p>
            )}
            {msg.cardDraft && (
              <InlineCardDraft draft={msg.cardDraft} template={templateForVersion(cardTemplates, msg.cardDraft.templateVersionId)} templateName={msg.cardTemplateName || "Карточка"} onSave={onSaveCard} />
            )}
            {msg.role === "assistant" && msg.status === "done" && (renderedAnswer?.sourcesHtml || msg.checkpointId) && (
              <div className="assistant-message-actions" aria-label="Действия с ответом">
                {renderedAnswer?.sourcesHtml && (
                  <button
                    type="button"
                    className="source-action"
                    aria-expanded={openSourcesMessageId === msg.id}
                    onClick={() => setOpenSourcesMessageId((current) => current === msg.id ? null : msg.id)}
                  >Источники{renderedAnswer.sourceCount ? ` · ${renderedAnswer.sourceCount}` : ""}</button>
                )}
                {msg.checkpointId && msg.graphChainCount && openGraphId !== msg.checkpointId && (
                  <button
                    type="button"
                    className="graph-open"
                    onClick={() => onOpenGraph(msg.checkpointId!)}
                  >
                    <IconGraph /> Данные ответа
                    {msg.graphChainCount > 1 ? ` (${msg.graphChainCount})` : ""}
                  </button>
                )}
                {msg.checkpointId && <button
                  type="button"
                  className="desktop-fork-action"
                  disabled={Boolean(forkingCheckpointId)}
                  onClick={() => onFork(msg.checkpointId!)}
                ><IconFork /> {forkingCheckpointId === msg.checkpointId ? "Создаю вариант" : "Новый вариант"}</button>}
                {msg.checkpointId && <details className="mobile-message-menu">
                  <summary aria-label="Действия с ответом">•••</summary>
                  <button type="button" disabled={Boolean(forkingCheckpointId)} onClick={() => onFork(msg.checkpointId!)}><IconFork /> Новый вариант</button>
                </details>}
              </div>
            )}
            {msg.role === "assistant" && renderedAnswer?.sourcesHtml && openSourcesMessageId === msg.id && (
              <div id={`sources-${msg.id}`} className="answer-source-panel md" dangerouslySetInnerHTML={{ __html: renderedAnswer.sourcesHtml }} />
            )}
            {pendingApproval?.assistantMessageId === msg.id && pendingApproval.status === "pending" && (
              <ApprovalCard
                approval={pendingApproval}
                agenda={agenda}
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
            agenda={agenda}
            busy={Boolean(approvalBusy)}
            onResolve={onResolveApproval}
          />
        </article>
      )}
    </div>
  );
}
