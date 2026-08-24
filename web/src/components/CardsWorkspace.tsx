import { useEffect, useMemo, useState } from "react";
import {
  archiveCard,
  archiveCardTemplate,
  attachCard,
  createCardTemplate,
  fetchCards,
  fetchCardTemplates,
  importCardDraft,
  saveCardDraft,
} from "../api";
import type { CardDraft, CardTemplate, SavedCard } from "../types";
import { prettyJson } from "../format";

export function CardsWorkspace({
  checkpointId,
  branchId,
  onCheckpoint,
  onNotice,
}: {
  checkpointId: string;
  branchId: string;
  onCheckpoint: (id: string) => void;
  onNotice: (text: string) => void;
}) {
  const [tab, setTab] = useState<"templates" | "library">("templates");
  const [templates, setTemplates] = useState<CardTemplate[]>([]);
  const [cards, setCards] = useState<SavedCard[]>([]);
  const [draft, setDraft] = useState<CardDraft | null>(null);
  const [draftQueue, setDraftQueue] = useState<CardDraft[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [busy, setBusy] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [schemaText, setSchemaText] = useState('{\n  "type": "object",\n  "properties": {}\n}');

  const reload = async () => {
    const [nextTemplates, nextCards] = await Promise.all([fetchCardTemplates(), fetchCards()]);
    setTemplates(nextTemplates);
    setCards(nextCards);
    if (!selectedTemplate) setSelectedTemplate(nextTemplates[0]?.latestVersion.id || "");
  };

  useEffect(() => { void reload().catch((error: Error) => onNotice(error.message)); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const template = useMemo(
    () => templates.find((item) => item.latestVersion.id === selectedTemplate),
    [templates, selectedTemplate]
  );

  return (
    <section className="cards-workspace">
      <header className="workspace-bar">
        <nav>
          <button className={tab === "templates" ? "is-on" : ""} onClick={() => setTab("templates")}>Шаблоны</button>
          <button className={tab === "library" ? "is-on" : ""} onClick={() => setTab("library")}>Библиотека</button>
        </nav>
      </header>

      {tab === "templates" ? (
        <div className="cards-grid">
          <div className="cards-list-panel">
            {templates.map((item) => (
              <button
                type="button"
                key={item.id}
                className={selectedTemplate === item.latestVersion.id ? "is-active" : ""}
                onClick={() => setSelectedTemplate(item.latestVersion.id)}
              >
                <strong>{item.name}</strong>
                <span>v{item.latestVersion.version}{item.system ? " · встроенный" : " · личный"}</span>
                <p>{item.description}</p>
              </button>
            ))}
            <button type="button" className="new-template" onClick={() => setCreateOpen((value) => !value)}>+ Новый шаблон</button>
          </div>
          <div className="card-detail-panel">
            {createOpen ? (
              <form
                className="template-editor"
                onSubmit={async (event) => {
                  event.preventDefault();
                  setBusy(true);
                  try {
                    await createCardTemplate({ name, schema: JSON.parse(schemaText) });
                    setName("");
                    setCreateOpen(false);
                    await reload();
                  } catch (error) {
                    onNotice(error instanceof Error ? error.message : "Некорректный JSON Schema");
                  } finally { setBusy(false); }
                }}
              >
                <label>Название<input value={name} onChange={(event) => setName(event.target.value)} required /></label>
                <label>JSON Schema<textarea value={schemaText} onChange={(event) => setSchemaText(event.target.value)} /></label>
                <button className="primary-btn" disabled={busy}>Создать immutable v1</button>
              </form>
            ) : template ? (
              <>
                <div className="card-detail-head">
                  <div><h2>{template.name}</h2><p>{template.description}</p></div>
                  <div className="template-actions">
                    <button className="ghost-btn" onClick={async () => {
                      const copy = await createCardTemplate({
                        name: `${template.name} — копия`,
                        description: template.description,
                        schema: template.latestVersion.schema,
                        ui: template.latestVersion.ui,
                        instructions: template.latestVersion.instructions,
                      });
                      await reload(); setSelectedTemplate(copy.latestVersion.id);
                    }}>Дублировать</button>
                    {!template.system && <button className="ghost-btn danger-btn" onClick={async () => { await archiveCardTemplate(template.id); setSelectedTemplate(""); await reload(); }}>Архивировать</button>}
                  </div>
                </div>
                <pre className="schema-preview">{prettyJson(template.latestVersion.schema)}</pre>
                {draft && (
                  <article className="card-draft-preview">
                    <span>DRAFT · не сохранено{draftQueue.length ? ` · ещё ${draftQueue.length} в очереди` : ""}</span>
                    <pre>{prettyJson(draft.data)}</pre>
                    <button
                      className="primary-btn"
                      onClick={async () => {
                        setBusy(true);
                        try {
                          await saveCardDraft(draft.id, String(draft.data.title || template.name));
                          await reload();
                          const [nextDraft, ...rest] = draftQueue;
                          setDraft(nextDraft || null);
                          setDraftQueue(rest);
                          if (!nextDraft) setTab("library");
                        } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка сохранения"); }
                        finally { setBusy(false); }
                      }}
                    >Сохранить revision</button>
                  </article>
                )}
              </>
            ) : <p className="muted">Выберите шаблон.</p>}
          </div>
        </div>
      ) : (
        <div className="library-shell">
          <div className="library-tools">
            <label className="ghost-btn">
              Импорт JSON
              <input
                type="file"
                accept="application/json,.json"
                hidden
                onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (!file || !selectedTemplate) return;
                  setBusy(true);
                  try {
                    const parsed = JSON.parse(await file.text());
                    const objects = Array.isArray(parsed) ? parsed : [parsed];
                    if (!objects.length || objects.some((item) => !item || typeof item !== "object" || Array.isArray(item))) {
                      throw new Error("JSON должен содержать object или непустой array объектов");
                    }
                    const imported = await importCardDraft(selectedTemplate, Array.isArray(parsed) ? objects : objects[0]);
                    setDraft(imported[0] || null);
                    setDraftQueue(imported.slice(1));
                    setTab("templates");
                    onNotice(`Импортировано draft: ${imported.length}. Они помечены unverified и требуют сохранения.`);
                  } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка импорта"); }
                  finally { setBusy(false); event.target.value = ""; }
                }}
              />
            </label>
            <select value={selectedTemplate} onChange={(event) => setSelectedTemplate(event.target.value)}>
              {templates.map((item) => <option key={item.id} value={item.latestVersion.id}>{item.name} v{item.latestVersion.version}</option>)}
            </select>
          </div>
          <div className="library-grid">
          {cards.length === 0 && <p className="muted">Сохранённых карточек пока нет.</p>}
          {cards.map((card) => (
            <article key={card.id} className="library-card">
              <header><strong>{card.title}</strong><span>revision {card.latestRevision.revision}</span></header>
              <pre>{prettyJson(card.latestRevision.data)}</pre>
              <footer>
                <button
                  className="ghost-btn"
                  disabled={!branchId || !checkpointId || busy}
                  onClick={async () => {
                    setBusy(true);
                    try {
                      const result = await attachCard(branchId, checkpointId, card.latestRevision.id);
                      onCheckpoint(result.checkpointId);
                      onNotice("Карточка прикреплена к ветке.");
                    } catch (error) { onNotice(error instanceof Error ? error.message : "Ошибка прикрепления"); }
                    finally { setBusy(false); }
                  }}
                >Прикрепить к ветке</button>
                <button
                  className="ghost-btn danger-btn"
                  onClick={async () => { await archiveCard(card.id); await reload(); }}
                >Архивировать</button>
              </footer>
            </article>
          ))}
          </div>
        </div>
      )}
    </section>
  );
}
