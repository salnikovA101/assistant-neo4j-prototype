import { useEffect, useMemo, useState } from "react";
import {
  archiveCard, archiveCardTemplate, createCardTemplate,
  createCardTemplateVersion, fetchCards, fetchCardTemplates, importCardDraft,
  reviseCard, saveCardDraft, updateCardDraft,
} from "../api";
import { blankData, fallbackSchemaForData } from "../cardModel";
import type { CardDraft, CardTemplate, SavedCard } from "../types";
import { CardTemplateBuilder, type TemplateBuilderValue } from "./CardTemplateBuilder";
import { CardVisual } from "./CardVisual";
import { IconTrash } from "./Icons";

function templateForVersion(templates: CardTemplate[], versionId: string): CardTemplate | undefined {
  return templates.find((item) => item.latestVersion.id === versionId);
}

function changedProvenance(provenance: Record<string, unknown>, key: string): Record<string, unknown> {
  const pointer = `/${key.replaceAll("~", "~0").replaceAll("/", "~1")}`;
  const next = Object.fromEntries(Object.entries(provenance).filter(([existing]) => existing !== pointer && !existing.startsWith(`${pointer}/`)));
  next[pointer] = [{ verification: "user-edited" }];
  return next;
}

function DraftReview({ draft, template, queueSize, busy, onSave }: {
  draft: CardDraft;
  template?: CardTemplate;
  queueSize: number;
  busy: boolean;
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
      <button className="primary-btn" disabled={busy || !title} onClick={() => onSave(data, provenance, title)}>Сохранить карточку</button>
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
      status={`Новая правка после ${card.latestRevision.revision}`}
      onChange={(key, value) => {
        setData((current) => ({ ...current, [key]: value }));
        setEdited((current) => new Set(current).add(key));
      }}
    />
    {!title && <p className="draft-save-hint">Название обязательно.</p>}
    <footer><button type="button" className="primary-btn" disabled={busy || !title || edited.size === 0} onClick={() => onSave(data, [...edited], title)}>Сохранить правку</button><button type="button" className="ghost-btn" disabled={busy} onClick={onCancel}>Отмена</button></footer>
  </article>;
}

export function CardsWorkspace({
  checkpointId, branchId, onNotice, chatMode = false,
  onGenerate, onInsert, onClose, initialTab = "templates", readOnly = false,
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
  const [tab, setTab] = useState<"templates" | "library">(initialTab);
  const [templates, setTemplates] = useState<CardTemplate[]>([]);
  const [cards, setCards] = useState<SavedCard[]>([]);
  const [draft, setDraft] = useState<CardDraft | null>(null);
  const [draftQueue, setDraftQueue] = useState<CardDraft[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [busy, setBusy] = useState(false);
  const [editor, setEditor] = useState<"create" | "edit" | null>(null);
  const [editingCardId, setEditingCardId] = useState("");

  const reload = async () => {
    const [nextTemplates, nextCards] = await Promise.all([fetchCardTemplates(), fetchCards()]);
    setTemplates(nextTemplates); setCards(nextCards);
    setSelectedTemplate((current) => current && nextTemplates.some((item) => item.latestVersion.id === current)
      ? current : "");
  };
  useEffect(() => { void reload().catch((error: Error) => onNotice(error.message)); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setTab(initialTab); }, [initialTab]);
  useEffect(() => {
    // Keep the detail pane useful on first open.  Both the full workspace and
    // the compact chat pane should show the first available template instead
    // of an empty "Выберите шаблон" state; the user can still switch templates
    // explicitly from the list.
    if (!selectedTemplate && templates[0]) setSelectedTemplate(templates[0].latestVersion.id);
  }, [selectedTemplate, tab, templates]);
  const template = useMemo(() => templateForVersion(templates, selectedTemplate), [templates, selectedTemplate]);
  const draftTemplate = draft ? templateForVersion(templates, draft.templateVersionId) : undefined;

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

  return (
    <section className="cards-workspace">
      <header className="workspace-bar">
        {!chatMode && <label className="ghost-btn card-import-control">Импорт JSON<input type="file" accept="application/json,.json" hidden onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          if (!selectedTemplate) {
            onNotice("Сначала выберите шаблон на вкладке «Шаблоны».");
            event.target.value = "";
            return;
          }
          setBusy(true);
          try {
            const parsed = JSON.parse(await file.text()); const objects = Array.isArray(parsed) ? parsed : [parsed];
            if (!objects.length || objects.some((item) => !item || typeof item !== "object" || Array.isArray(item))) throw new Error("JSON должен содержать object или непустой array объектов");
            const imported = await importCardDraft(selectedTemplate, Array.isArray(parsed) ? objects : objects[0], checkpointId || undefined);
            setDraft(imported[0] || null); setDraftQueue(imported.slice(1)); setTab("templates"); onNotice(`Импортировано черновиков: ${imported.length}. Проверьте их перед сохранением.`);
          } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка импорта"); }
          finally { setBusy(false); event.target.value = ""; }
        }} /></label>}
        <nav><button className={tab === "templates" ? "is-on" : ""} onClick={() => setTab("templates")}>Шаблоны</button><button className={tab === "library" ? "is-on" : ""} onClick={() => setTab("library")}>Библиотека</button></nav>
        {onClose && <button type="button" className="icon-btn" aria-label="Закрыть карточки" title="Закрыть" onClick={onClose}>×</button>}
      </header>

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
        <div className="cards-grid">
          <div className="cards-list-panel">
            {templates.map((item) => <button type="button" key={item.id} className={selectedTemplate === item.latestVersion.id ? "is-active" : ""} onClick={() => { setSelectedTemplate(item.latestVersion.id); setEditor(null); }}><strong>{item.name}</strong><span>{item.system ? "Встроенный" : "Личный"}</span><p>{item.description}</p></button>)}
            {!chatMode && <button type="button" className="new-template" onClick={() => setEditor("create")}>+ Новый шаблон</button>}
          </div>
          <div className="card-detail-panel">
            {editor ? (
              <CardTemplateBuilder
                key={`${editor}-${template?.latestVersion.id || "new"}`}
                initial={editor === "edit" && template ? { name: template.name, description: template.description, schema: template.latestVersion.schema, ui: template.latestVersion.ui, instructions: template.latestVersion.instructions } : undefined}
                busy={busy}
                submitLabel={editor === "edit" ? "Сохранить новую версию" : "Создать шаблон"}
                onSubmit={saveBuilder}
                onCancel={() => setEditor(null)}
              />
            ) : template ? (
              <>
                <div className="card-detail-head">
                  <div><h2>{template.name}</h2><p>{template.description}</p></div>
                  <div className="template-actions">
                    {chatMode && onGenerate && <button type="button" className="primary-btn" disabled={busy || readOnly} onClick={() => onGenerate(template.latestVersion.id)}>Создать</button>}
                    {!chatMode && <button type="button" className="ghost-btn" onClick={() => setEditor("edit")}>Изменить</button>}
                    {!chatMode && !template.system && <button type="button" className="delete-icon-btn" aria-label="Удалить шаблон" title="Удалить шаблон" disabled={busy} onClick={async () => {
                      if (!window.confirm(`Удалить шаблон «${template.name}»?`)) return;
                      setBusy(true);
                      try { await archiveCardTemplate(template.id); setSelectedTemplate(""); await reload(); onNotice("Шаблон удалён."); }
                      catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось удалить шаблон"); }
                      finally { setBusy(false); }
                    }}><IconTrash /></button>}
                  </div>
                </div>
                <CardVisual templateName={template.name} version={template.latestVersion.version} schema={template.latestVersion.schema} ui={template.latestVersion.ui} data={blankData(template.latestVersion.schema)} status="Шаблон" />
                {!chatMode && <details className="technical-card-data"><summary>Дополнительно · технические данные</summary><pre>{JSON.stringify({ schema: template.latestVersion.schema, ui: template.latestVersion.ui }, null, 2)}</pre></details>}
                {draft && <DraftReview draft={draft} template={draftTemplate} queueSize={draftQueue.length} busy={busy} onSave={async (data, provenance, title) => {
                  setBusy(true);
                  try {
                    await updateCardDraft(draft.id, { data, provenance, gaps: draft.gaps }); await saveCardDraft(draft.id, title); await reload();
                    const [nextDraft, ...rest] = draftQueue; setDraft(nextDraft || null); setDraftQueue(rest); if (!nextDraft) setTab("library");
                  } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка сохранения"); }
                  finally { setBusy(false); }
                }} />}
              </>
            ) : <div className="cards-empty-state">{templates.length
              ? <><strong>Выберите шаблон</strong><p>Слева находятся встроенные и личные шаблоны. Откройте нужный, чтобы посмотреть поля или начать заполнение.</p></>
              : <><strong>Шаблонов пока нет</strong><p>{chatMode ? "Встроенные шаблоны появятся здесь, когда карточки будут доступны." : "Создайте личный шаблон кнопкой «+ Новый шаблон»."}</p></>
            }</div>}
          </div>
        </div>
        )
      ) : (
        <div className="library-shell">
          <div className="library-grid">
            {cards.length === 0 && <div className="cards-empty-state"><strong>Сохранённых карточек пока нет</strong><p>Откройте диалог, нажмите <code>+</code> у поля сообщения и выберите «Открыть карточки».</p></div>}
            {cards.map((card) => {
              const cardTemplate = templateForVersion(templates, card.templateVersionId);
              if (editingCardId === card.id) return <SavedCardEditor key={card.id} card={card} template={cardTemplate} busy={busy} onCancel={() => setEditingCardId("")} onSave={async (data, editedFields, title) => {
                setBusy(true);
                try { await reviseCard(card.id, { data, editedFields, title }); await reload(); setEditingCardId(""); onNotice("Новая правка карточки сохранена."); }
                catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось сохранить правку"); }
                finally { setBusy(false); }
              }} />;
              return <article key={card.id} className={chatMode ? "library-card chat-library-card" : "library-card"}>
                <CardVisual templateName={cardTemplate?.name || card.template?.name || "Карточка"} version={cardTemplate?.latestVersion.version || card.template?.version} schema={cardTemplate?.latestVersion.schema || card.template?.schema || fallbackSchemaForData(card.latestRevision.data)} ui={cardTemplate?.latestVersion.ui || card.template?.ui || {}} data={{ ...card.latestRevision.data, title: card.title }} provenance={card.latestRevision.provenance} status={`Правка ${card.latestRevision.revision}`} />
                <footer>
                  {chatMode && onInsert && <button className="primary-btn" disabled={!branchId || !checkpointId || busy || readOnly} onClick={() => onInsert(card)}>Вставить в диалог</button>}
                  {!chatMode && <button className="ghost-btn" onClick={() => setEditingCardId(card.id)}>Изменить</button>}
                  {!chatMode && <button type="button" className="ghost-btn danger-btn" disabled={busy} onClick={async () => {
                    if (!window.confirm(`Убрать карточку «${card.title}» из библиотеки?`)) return;
                    setBusy(true);
                    try { await archiveCard(card.id); await reload(); onNotice("Карточка убрана из библиотеки."); }
                    catch (error) { onNotice(error instanceof Error ? error.message : "Не удалось убрать карточку"); }
                    finally { setBusy(false); }
                  }}>Удалить</button>}
                </footer>
              </article>;
            })}
          </div>
        </div>
      )}
    </section>
  );
}
