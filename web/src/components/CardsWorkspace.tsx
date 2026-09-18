import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  archiveCard, archiveCardTemplate, createCardTemplate,
  createCardTemplateVersion, downloadCard, fetchCards, fetchCardTemplates, importCardDraft,
  reviseCard, saveCardDraft, updateCardActionState, updateCardDraft,
} from "../api";
import { blankData, fallbackSchemaForData } from "../cardModel";
import type { CardActionId, CardActionState, CardDraft, CardTemplate, SavedCard } from "../types";
import { CardTemplateBuilder, type TemplateBuilderValue } from "./CardTemplateBuilder";
import { CardVisual } from "./CardVisual";

const plannedCardActions = [
  { id: "digital_experiment", label: "Цифровой эксперимент", description: "Цифровой эксперимент пока недоступен. Здесь появятся результаты проверки карточки." },
  { id: "regulations", label: "Проверить по нормативам", description: "Проверка по нормативным документам в разработке." },
] as const;

const emptyCardActionState: CardActionState = {
  activeAction: null,
  actions: {
    digital_experiment: { status: "not_ready", logs: [] },
    regulations: { status: "not_ready", logs: [] },
  },
};

function PlannedCardActions({ card, onState, onNotice }: {
  card: SavedCard;
  onState: (state: CardActionState) => void;
  onNotice: (text: string) => void;
}) {
  const state = card.actionState || emptyCardActionState;
  const activeAction = state.activeAction;
  const [saving, setSaving] = useState(false);
  const saveInFlight = useRef(false);
  const panelId = useId();
  const selectAction = async (action: CardActionId) => {
    if (saveInFlight.current) return;
    saveInFlight.current = true;
    setSaving(true);
    const nextAction = activeAction === action ? null : action;
    onState({ ...state, activeAction: nextAction });
    try {
      onState(await updateCardActionState(card.id, nextAction));
    } catch (error) {
      onState(state);
      onNotice(error instanceof Error ? error.message : "Не удалось сохранить состояние проверки карточки");
    } finally {
      saveInFlight.current = false;
      setSaving(false);
    }
  };
  return (
    <div className="card-planned-actions" aria-label="Будущие возможности карточки">
      {plannedCardActions.map((action) => (
        <button key={action.id} id={`${panelId}-${action.id}`} type="button" className="card-planned-button" disabled={saving} aria-pressed={activeAction === action.id} aria-expanded={activeAction === action.id} aria-controls={panelId} onClick={() => void selectAction(action.id)}>
          <span>{action.label}</span><span className="card-soon-badge">Скоро</span>
        </button>
      ))}
      <div id={panelId} className="card-planned-panel" role="region" aria-labelledby={activeAction === null ? undefined : `${panelId}-${activeAction}`} hidden={activeAction === null}>
        <p>{plannedCardActions.find((action) => action.id === activeAction)?.description}</p>
      </div>
    </div>
  );
}

function templateForVersion(templates: CardTemplate[], versionId: string): CardTemplate | undefined {
  return templates.find((item) => item.latestVersion.id === versionId);
}

function changedProvenance(provenance: Record<string, unknown>, key: string): Record<string, unknown> {
  const pointer = `/${key.replaceAll("~", "~0").replaceAll("/", "~1")}`;
  const next = Object.fromEntries(Object.entries(provenance).filter(([existing]) => existing !== pointer && !existing.startsWith(`${pointer}/`)));
  next[pointer] = [{ verification: "user-edited" }];
  return next;
}

function DraftReview({ draft, template, queueSize, busy, onSave, onCancel }: {
  draft: CardDraft;
  template?: CardTemplate;
  queueSize: number;
  busy: boolean;
  onCancel: () => void;
  onSave: (data: Record<string, unknown>, provenance: Record<string, unknown>, title: string) => Promise<void>;
}) {
  const [data, setData] = useState(draft.data);
  const [provenance, setProvenance] = useState(draft.provenance);
  useEffect(() => { setData(draft.data); setProvenance(draft.provenance); }, [draft]);
  const schema = template?.latestVersion.schema || fallbackSchemaForData(data);
  const title = String(data.title || "").trim();
  return (
    <section className="card-draft-review">
      <div className="draft-review-kicker">Черновик · не сохранено{queueSize ? ` · ещё ${queueSize} в очереди` : ""}</div>
      <CardVisual
        templateName={template?.name || "Карточка"}
        version={template?.latestVersion.version}
        schema={schema}
        ui={template?.latestVersion.ui || {}}
        data={data}
        provenance={provenance}
        editable
        status="Проверка"
        onChange={(key, value) => {
          setData((current) => ({ ...current, [key]: value }));
          setProvenance((current) => changedProvenance(current, key));
        }}
      />
      {!title && <p className="draft-save-hint">Добавьте название карточки перед сохранением.</p>}
      <footer><button className="primary-btn" disabled={busy || !title} onClick={() => onSave(data, provenance, title)}>Сохранить карточку</button><button type="button" className="ghost-btn" disabled={busy} onClick={onCancel}>Отмена</button></footer>
    </section>
  );
}

function SavedCardEditor({ card, template, busy, onCancel, onSave }: {
  card: SavedCard;
  template?: CardTemplate;
  busy: boolean;
  onCancel: () => void;
  onSave: (data: Record<string, unknown>, editedFields: string[], title: string) => Promise<void>;
}) {
  const [data, setData] = useState<Record<string, unknown>>({ ...card.latestRevision.data, title: card.title });
  const [edited, setEdited] = useState<Set<string>>(new Set());
  const schema = template?.latestVersion.schema || card.template?.schema || fallbackSchemaForData(data);
  const title = String(data.title || "").trim();
  return <article className="library-card is-editing">
    <CardVisual
      templateName={template?.name || card.template?.name || "Карточка"}
      version={template?.latestVersion.version || card.template?.version}
      schema={schema}
      ui={template?.latestVersion.ui || card.template?.ui || {}}
      data={data}
      provenance={card.latestRevision.provenance}
      editable
      status="Редактирование"
      onChange={(key, value) => {
        setData((current) => ({ ...current, [key]: value }));
        setEdited((current) => new Set(current).add(key));
      }}
    />
    {!title && <p className="draft-save-hint">Название обязательно.</p>}
    <footer><button type="button" className="primary-btn" disabled={busy || !title || edited.size === 0} onClick={() => onSave(data, [...edited], title)}>Сохранить изменения</button><button type="button" className="ghost-btn" disabled={busy} onClick={onCancel}>Отмена</button></footer>
  </article>;
}

export function CardsWorkspace({
  checkpointId, branchId, onNotice, chatMode = false,
  onGenerate, onInsert, onClose, initialTab = "library", readOnly = false,
}: {
  checkpointId: string;
  branchId: string;
  onNotice: (text: string) => void;
  chatMode?: boolean;
  onGenerate?: (templateVersionId: string) => void;
  onInsert?: (card: SavedCard) => void;
  onClose?: () => void;
  initialTab?: "templates" | "library";
  readOnly?: boolean;
}) {
  const importInputRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<"templates" | "library">(initialTab);
  const [templates, setTemplates] = useState<CardTemplate[]>([]);
  const [cards, setCards] = useState<SavedCard[]>([]);
  const [draft, setDraft] = useState<CardDraft | null>(null);
  const [draftQueue, setDraftQueue] = useState<CardDraft[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [busy, setBusy] = useState(false);
  const [editor, setEditor] = useState<"create" | "edit" | null>(null);
  const [downloadingCardId, setDownloadingCardId] = useState("");
  const [editingCardId, setEditingCardId] = useState("");
  const [query, setQuery] = useState("");
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const queryNeedle = query.trim().toLocaleLowerCase("ru-RU");
  const visibleCards = cards.filter((card) => `${card.title} ${JSON.stringify(card.latestRevision.data)}`.toLocaleLowerCase("ru-RU").includes(queryNeedle));
  const visibleTemplates = useMemo(() => queryNeedle
    ? templates.filter((item) => `${item.name} ${item.description}`.toLocaleLowerCase("ru-RU").includes(queryNeedle))
    : templates, [templates, queryNeedle]);

  const reload = async () => {
    const [nextTemplates, nextCards] = await Promise.all([fetchCardTemplates(), fetchCards()]);
    setTemplates(nextTemplates); setCards(nextCards);
    setSelectedTemplate((current) => current && nextTemplates.some((item) => item.latestVersion.id === current)
      ? current : "");
  };
  useEffect(() => { void reload().catch((error: Error) => onNotice(error.message)); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setTab(initialTab); }, [initialTab]);
  useEffect(() => {
    if (editor) return;
    const pool = chatMode ? templates : visibleTemplates;
    if (pool.some((item) => item.latestVersion.id === selectedTemplate)) return;
    const next = pool[0]?.latestVersion.id || "";
    if (next !== selectedTemplate) setSelectedTemplate(next);
  }, [chatMode, editor, selectedTemplate, templates, visibleTemplates]);
  const template = useMemo(() => templateForVersion(templates, selectedTemplate), [templates, selectedTemplate]);
  const draftTemplate = draft ? templateForVersion(templates, draft.templateVersionId) : undefined;

  const importFile = async (file: File) => {
    setBusy(true);
    try {
      const uploaded = JSON.parse(await file.text());
      if (uploaded?.format === "neo4j-assistant-card" && ![1, 2].includes(uploaded.formatVersion)) throw new Error("Эта версия файла карточки пока не поддерживается");
      const templateName = typeof uploaded?.template === "string" ? uploaded.template : uploaded?.template?.name;
      if (typeof templateName !== "string" || !templateName.trim()) throw new Error("В JSON не указано название шаблона. Выберите файл, скачанный из карточки.");
      const data = uploaded.data;
      const objects = Array.isArray(data) ? data : [data];
      if (!objects.length || objects.some((item) => !item || typeof item !== "object" || Array.isArray(item))) throw new Error("В JSON нет данных карточки.");
      const availableTemplates = await fetchCardTemplates();
      const normalizeName = (name: string) => name.normalize("NFC").trim().replace(/\s+/g, " ").toLocaleLowerCase("ru-RU");
      const matches = availableTemplates.filter((item) => normalizeName(item.name) === normalizeName(templateName));
      if (!matches.length) throw new Error(`Шаблон «${templateName}» не найден. Добавьте его в Шаблонах и повторите загрузку.`);
      if (matches.length > 1) throw new Error(`Найдено несколько шаблонов «${templateName}». Задайте им разные названия, чтобы шаблон определялся однозначно.`);
      const imported = await importCardDraft(matches[0].latestVersion.id, data, checkpointId || undefined);
      setTemplates(availableTemplates);
      setDraft(imported[0] || null);
      setDraftQueue(imported.slice(1));
      setTab("library");
      onNotice(`Шаблон «${matches[0].name}» выбран автоматически. Проверьте карточку и сохраните её.`);
    } catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось вставить карточку"); }
    finally { setBusy(false); }
  };

  const saveBuilder = async (value: TemplateBuilderValue) => {
    setBusy(true);
    try {
      if (editor === "edit" && template && !template.system) {
        await createCardTemplateVersion(template.id, value);
        onNotice("Создана новая версия шаблона.");
      } else {
        const created = await createCardTemplate(value);
        setSelectedTemplate(created.latestVersion.id);
        onNotice("Шаблон создан.");
      }
      setEditor(null); await reload();
    } catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось сохранить шаблон"); }
    finally { setBusy(false); }
  };

  const builderSubmitLabel = editor === "edit" && template?.system
    ? "Сохранить копию"
    : editor === "edit"
      ? "Сохранить новую версию"
      : "Создать шаблон";

  return (
    <section className="cards-workspace">
      {chatMode && (
        <header className="workspace-bar cards-workspace-bar">
          <nav><button className={tab === "library" ? "is-on" : ""} onClick={() => setTab("library")}>Библиотека</button><button className={tab === "templates" ? "is-on" : ""} onClick={() => setTab("templates")}>Шаблоны</button></nav>
          {onClose && <button type="button" className="icon-btn" aria-label="Закрыть карточки" title="Закрыть" onClick={onClose}>×</button>}
        </header>
      )}

      {!chatMode && !editor && (
        <div className="ws-toolbar">
          <div className="ws-seg" role="group" aria-label="Раздел карточек">
            <button type="button" aria-pressed={tab === "library"} onClick={() => { setEditor(null); setTab("library"); }}>Библиотека</button>
            <button type="button" aria-pressed={tab === "templates"} onClick={() => { setEditor(null); setTab("templates"); }}>Шаблоны</button>
          </div>
          <label className="ws-search">
            <span className="sr-only">{tab === "library" ? "Поиск по карточкам" : "Поиск по шаблонам"}</span>
            <input type="search" placeholder={tab === "library" ? "Найти карточку по названию или содержанию" : "Найти шаблон по названию или описанию"} value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          {tab === "library"
            ? cards.length > 0 && <span className="library-count">{visibleCards.length} из {cards.length}</span>
            : templates.length > 0 && <span className="library-count">{visibleTemplates.length} из {templates.length}</span>}
          <div className="library-tools-end">
            {tab === "library" ? (
              <>
                <input ref={importInputRef} type="file" accept="application/json,.json" hidden disabled={busy || readOnly || !!draft} onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  event.currentTarget.value = "";
                  if (file) void importFile(file);
                }} />
                <button type="button" className="ghost-btn library-import-button" disabled={busy || readOnly || !!draft} title="Загрузить JSON — шаблон определится автоматически" onClick={() => importInputRef.current?.click()}>Импорт карточки</button>
              </>
            ) : (
              <button type="button" className="ghost-btn library-import-button" onClick={() => setEditor("create")}>Новый шаблон</button>
            )}
          </div>
        </div>
      )}

      {tab === "templates" ? (
        chatMode ? (
          <div className="library-shell chat-template-shell">
            <div className="library-grid chat-template-grid">
              {templates.length === 0 && <div className="cards-empty-state"><strong>Шаблонов пока нет</strong><p>Встроенные шаблоны появятся здесь, когда карточки будут доступны.</p></div>}
              {templates.map((item) => <article key={item.id} className="library-card chat-template-card">
                <div className="chat-template-card-head">
                  <div><span className="chat-template-kind">{item.system ? "Встроенный" : "Личный"}</span><h2>{item.name}</h2><p>{item.description}</p></div>
                </div>
                <CardVisual templateName={item.name} version={item.latestVersion.version} schema={item.latestVersion.schema} ui={item.latestVersion.ui} data={blankData(item.latestVersion.schema)} status="Шаблон" />
                {onGenerate && <footer><button type="button" className="primary-btn" disabled={busy || readOnly} onClick={() => onGenerate(item.latestVersion.id)}>Создать</button></footer>}
              </article>)}
            </div>
          </div>
        ) : (
          <div className={`cards-grid${editor ? " is-editing" : ""}`}>
            {editor === null && (
              <div className="cards-list-panel">
                {templates.length > 0 && visibleTemplates.length === 0 && <p className="cards-list-empty">Шаблоны не найдены</p>}
                {visibleTemplates.map((item) => <button type="button" key={item.id} className={selectedTemplate === item.latestVersion.id ? "is-active" : ""} onClick={() => { setSelectedTemplate(item.latestVersion.id); setEditor(null); }}>
                  <span className="template-list-row"><strong>{item.name}</strong><span>{item.system ? "Встроенный" : "Личный"}</span></span>
                  {item.description ? <p>{item.description}</p> : null}
                </button>)}
              </div>
            )}
            <div className="card-detail-panel">
              {editor ? (
                <CardTemplateBuilder
                  key={`${editor}-${template?.latestVersion.id || "new"}`}
                  initial={editor === "edit" && template ? { name: template.name, description: template.description, schema: template.latestVersion.schema, ui: template.latestVersion.ui, instructions: template.latestVersion.instructions } : undefined}
                  busy={busy}
                  submitLabel={builderSubmitLabel}
                  onSubmit={saveBuilder}
                  onCancel={() => setEditor(null)}
                />
              ) : template ? (
                <>
                  <div className="card-detail-head">
                    <div className="card-detail-title">
                      <h2>{template.name}</h2>
                      <div className="template-actions">
                        <button type="button" className="ghost-btn" onClick={() => setEditor("edit")}>Изменить</button>
                        {!template.system && <button type="button" className="ghost-btn danger-btn" aria-label="Удалить шаблон" title="Удалить шаблон" disabled={busy} onClick={async () => {
                          if (!window.confirm(`Удалить шаблон «${template.name}»?`)) return;
                          setBusy(true);
                          try { await archiveCardTemplate(template.id); setSelectedTemplate(""); await reload(); onNotice("Шаблон удалён."); }
                          catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось удалить шаблон"); }
                          finally { setBusy(false); }
                        }}>Удалить</button>}
                      </div>
                    </div>
                    <p>{template.description}</p>
                  </div>
                  <div className="template-preview">
                    <CardVisual templateName={template.name} version={template.latestVersion.version} schema={template.latestVersion.schema} ui={template.latestVersion.ui} data={blankData(template.latestVersion.schema)} status="Шаблон" />
                  </div>
                  <details className="technical-card-data"><summary>Дополнительно · технические данные</summary><pre>{JSON.stringify({ schema: template.latestVersion.schema, ui: template.latestVersion.ui }, null, 2)}</pre></details>
                </>
              ) : <div className="cards-empty-state">{templates.length
                ? visibleTemplates.length
                  ? <><strong>Выберите шаблон</strong><p>Слева находятся встроенные и личные шаблоны. Откройте нужный, чтобы посмотреть карточку.</p></>
                  : <><strong>Шаблоны не найдены</strong><p>Попробуйте другое название или слово из описания.</p></>
                : <><strong>Шаблонов пока нет</strong><p>Создайте личный шаблон кнопкой «Новый шаблон».</p></>
              }</div>}
            </div>
          </div>
        )
      ) : (
        <div className="library-shell">
          {chatMode && (
            <div className="library-tools">
              <label className="library-search"><span className="sr-only">Поиск по карточкам</span><input type="search" placeholder="Найти карточку по названию или содержанию" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
              <div className="library-tools-end">
                <span className="library-count">{visibleCards.length} из {cards.length}</span>
              </div>
            </div>
          )}
          <div className="library-grid">
            {draft && <article className="library-card library-import-draft"><DraftReview key={draft.id} onCancel={() => { setDraft(null); setDraftQueue([]); }} draft={draft} template={draftTemplate} queueSize={draftQueue.length} busy={busy} onSave={async (data, provenance, title) => {
                  setBusy(true);
                  try {
                    await updateCardDraft(draft.id, { data, provenance, gaps: draft.gaps }); await saveCardDraft(draft.id, title); setQuery(""); await reload();
                    const [nextDraft, ...rest] = draftQueue; setDraft(nextDraft || null); setDraftQueue(rest); if (!nextDraft) setTab("library");
                  } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка сохранения"); }
                  finally { setBusy(false); }
                }} /></article>}
            {!draft && cards.length === 0 && <div className="cards-empty-state"><strong>Сохранённых карточек пока нет</strong><p>Откройте диалог и нажмите значок карточек у поля сообщения.</p></div>}
            {!draft && cards.length > 0 && visibleCards.length === 0 && <div className="cards-empty-state"><strong>Карточки не найдены</strong><p>Попробуйте другое название или слово из содержимого.</p></div>}
            {visibleCards.map((card) => {
              const cardTemplate = templateForVersion(templates, card.templateVersionId);
              if (editingCardId === card.id) return <SavedCardEditor key={card.id} card={card} template={cardTemplate} busy={busy} onCancel={() => setEditingCardId("")} onSave={async (data, editedFields, title) => {
                setBusy(true);
                try { await reviseCard(card.id, { data, editedFields, title }); await reload(); setEditingCardId(""); onNotice("Изменения карточки сохранены."); }
                catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось сохранить изменения"); }
                finally { setBusy(false); }
              }} />;
              return <article key={card.id} className={`library-card ${chatMode ? "chat-library-card" : ""} ${expandedCards.has(card.id) ? "is-expanded" : "is-preview"}`}>
                <CardVisual templateName={cardTemplate?.name || card.template?.name || "Карточка"} version={cardTemplate?.latestVersion.version || card.template?.version} schema={cardTemplate?.latestVersion.schema || card.template?.schema || fallbackSchemaForData(card.latestRevision.data)} ui={cardTemplate?.latestVersion.ui || card.template?.ui || {}} data={{ ...card.latestRevision.data, title: card.title }} provenance={card.latestRevision.provenance} status={card.authorship} />
                <footer className="saved-card-footer">
                  <button type="button" className="card-text-action card-expand" aria-expanded={expandedCards.has(card.id)} onClick={() => setExpandedCards((current) => { const next = new Set(current); if (next.has(card.id)) next.delete(card.id); else next.add(card.id); return next; })}>{expandedCards.has(card.id) ? "Свернуть" : "Открыть полностью"}</button>
                  <button type="button" className="card-text-action" title="Скачать PDF и JSON в ZIP-архиве" disabled={!!downloadingCardId} onClick={async () => {
                    setDownloadingCardId(card.id);
                    try { await downloadCard(card.id, card.title); }
                    catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось скачать карточку"); }
                    finally { setDownloadingCardId(""); }
                  }}>{downloadingCardId === card.id ? "Подготовка…" : "Скачать"}</button>
                  {chatMode && onInsert && <button className="primary-btn" disabled={!branchId || !checkpointId || busy || readOnly} onClick={() => onInsert(card)}>Вставить в диалог</button>}
                  {!chatMode && <button type="button" className="card-text-action" onClick={() => setEditingCardId(card.id)}>Изменить</button>}
                  {!chatMode && <button type="button" className="card-text-action danger-btn" disabled={busy} onClick={async () => {
                    if (!window.confirm(`Убрать карточку «${card.title}» из библиотеки?`)) return;
                    setBusy(true);
                    try { await archiveCard(card.id); await reload(); onNotice("Карточка убрана из библиотеки."); }
                    catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось убрать карточку"); }
                    finally { setBusy(false); }
                  }}>Удалить</button>}
                </footer>
                <PlannedCardActions card={card} onNotice={onNotice} onState={(actionState) => setCards((current) => current.map((item) => item.id === card.id ? { ...item, actionState } : item))} />
              </article>;
            })}
          </div>
        </div>
      )}
    </section>
  );
}
