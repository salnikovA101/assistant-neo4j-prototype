import { useMemo, useState, type DragEvent } from "react";
import {
  blankData, buildCardContract, nextFieldKey, readCardDefinition,
  type CardField, type CardWidget,
} from "../cardModel";
import { CardVisual } from "./CardVisual";

const WIDGETS: { value: CardWidget; label: string }[] = [
  { value: "short_text", label: "Короткий текст" },
  { value: "long_text", label: "Длинный текст" },
  { value: "number", label: "Число с единицей" },
  { value: "list", label: "Список" },
  { value: "choice", label: "Выбор варианта" },
  { value: "boolean", label: "Да / нет" },
  { value: "date", label: "Дата" },
];

export type TemplateBuilderValue = {
  name: string;
  description: string;
  schema: Record<string, unknown>;
  ui: Record<string, unknown>;
  instructions: string;
};

function initialField(key = "field_1", number = 1): CardField {
  return { key, label: `Поле ${number}`, description: "", widget: "long_text", required: false, unit: "", options: ["Вариант 1", "Вариант 2"] };
}

function move<T>(items: T[], from: number, to: number): T[] {
  if (to < 0 || to >= items.length || from === to) return items;
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

function CopyIcon() {
  return <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="6" y="6" width="9" height="9" rx="1.5" /><path d="M4 12H3.5A1.5 1.5 0 0 1 2 10.5v-7A1.5 1.5 0 0 1 3.5 2h7A1.5 1.5 0 0 1 12 3.5V4" /></svg>;
}

function TrashIcon() {
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11M8 3.5h4M6 6l.7 10h6.6L14 6M8.3 8.5v5M11.7 8.5v5" /></svg>;
}

export function CardTemplateBuilder({ initial, busy, submitLabel, onSubmit, onCancel }: {
  initial?: { name: string; description: string; schema: Record<string, unknown>; ui: Record<string, unknown>; instructions: string };
  busy: boolean;
  submitLabel: string;
  onSubmit: (value: TemplateBuilderValue) => Promise<void>;
  onCancel: () => void;
}) {
  const definition = readCardDefinition(initial?.schema || { type: "object", properties: {} }, initial?.ui || {});
  const viewOnlyLegacy = Boolean(initial && !definition.editable);
  const [name, setName] = useState(initial?.name || "");
  const [titleInstruction, setTitleInstruction] = useState(definition.titleInstruction);
  const [fields, setFields] = useState<CardField[]>(definition.fields.length ? definition.fields : [initialField()]);
  const [dragged, setDragged] = useState<number | null>(null);
  const generated = useMemo(() => buildCardContract(fields, titleInstruction), [fields, titleInstruction]);
  const contract = viewOnlyLegacy ? { schema: initial!.schema, ui: initial!.ui } : generated;
  const update = (index: number, patch: Partial<CardField>) => setFields((current) => current.map((field, itemIndex) => itemIndex === index ? { ...field, ...patch } : field));
  const duplicate = (index: number) => setFields((current) => {
    const source = current[index];
    const key = nextFieldKey(current);
    const clone = { ...source, key, label: `${source.label} — копия`, options: [...source.options] };
    const next = [...current]; next.splice(index + 1, 0, clone); return next;
  });

  return (
    <form className="template-builder" onSubmit={async (event) => {
      event.preventDefault();
      await onSubmit({
        name: name.trim(),
        description: initial?.description || "",
        schema: contract.schema,
        ui: contract.ui,
        instructions: initial?.instructions || "",
      });
    }}>
      <header className="builder-heading">
        <div><h2>{initial ? "Редактирование шаблона" : "Новый шаблон"}</h2><p>Поля задают структуру карточки, которую заполнит ассистент.</p></div>
        <button type="button" className="ghost-btn" onClick={onCancel}>Отмена</button>
      </header>
      <div className="builder-meta">
        <label>Название шаблона<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Например, Паспорт эксперимента" required /></label>
        <label>Правило для заголовка<input value={titleInstruction} onChange={(event) => setTitleInstruction(event.target.value)} placeholder="Кратко назови конкретный результат" /></label>
      </div>
      <div className="builder-layout">
        {viewOnlyLegacy ? (
          <section className="legacy-schema-notice"><strong>Старый вложенный шаблон</strong><p>Он доступен для просмотра. Чтобы менять поля через конструктор, создайте новый плоский шаблон.</p></section>
        ) : (
          <section className="builder-fields" aria-label="Поля карточки">
            <div className="builder-section-head"><strong>Поля карточки</strong><span>{fields.length}</span></div>
            {fields.map((field, index) => (
              <article
                key={field.key}
                className={`field-brick ${dragged === index ? "is-dragging" : ""}`}
                onDragOver={(event: DragEvent) => event.preventDefault()}
                onDrop={() => { if (dragged != null) setFields((current) => move(current, dragged, index)); setDragged(null); }}
              >
                <div className="field-brick-top">
                  <span className="drag-handle" draggable title="Перетащить поле" onDragStart={() => setDragged(index)} onDragEnd={() => setDragged(null)}>⋮⋮</span>
                  <input className="field-name-input" value={field.label} aria-label={`Название ${field.key}`} onChange={(event) => update(index, { label: event.target.value })} />
                  <button type="button" className="field-icon-btn" title="Дублировать поле" aria-label={`Дублировать ${field.label}`} onClick={() => duplicate(index)}><CopyIcon /></button>
                  <button type="button" className="field-icon-btn danger-btn" title="Удалить поле" aria-label={`Удалить ${field.label}`} onClick={() => setFields((current) => current.filter((_, itemIndex) => itemIndex !== index))}><TrashIcon /></button>
                </div>
                <div className="field-inline-settings">
                  <label className="field-type-control"><span>Формат ответа</span><select value={field.widget} onChange={(event) => update(index, { widget: event.target.value as CardWidget })}>{WIDGETS.map((widget) => <option key={widget.value} value={widget.value}>{widget.label}</option>)}</select></label>
                  <label className="field-required"><input type="checkbox" checked={field.required} onChange={(event) => update(index, { required: event.target.checked })} /><i aria-hidden="true" /><span>Обязательное</span></label>
                  {field.widget === "number" && <label className="field-unit-control"><span>Единица</span><input value={field.unit} onChange={(event) => update(index, { unit: event.target.value })} placeholder="°C, %, г/л" /></label>}
                </div>
                {field.widget === "choice" && <label className="field-options"><span>Варианты ответа</span><textarea value={field.options.join("\n")} onChange={(event) => update(index, { options: event.target.value.split("\n") })} placeholder="Один вариант на строку" /></label>}
                <div className="field-answer-example"><span>Пример ответа</span><em>Ответ ассистента появится здесь</em></div>
                <label className="field-guidance"><span>Подсказка ассистенту</span><textarea value={field.description} onChange={(event) => update(index, { description: event.target.value })} placeholder="Что именно нужно указать в этом поле" aria-label={`Пояснение для ${field.label}`} /></label>
              </article>
            ))}
            <button type="button" className="add-field-btn" onClick={() => setFields((current) => { const key = nextFieldKey(current); return [...current, initialField(key, current.length + 1)]; })}>+ Добавить поле</button>
          </section>
        )}
        <aside className="builder-preview"><div className="builder-section-head"><strong>Предпросмотр</strong></div><CardVisual templateName={name || "Новый шаблон"} schema={contract.schema} ui={contract.ui} data={blankData(contract.schema)} /></aside>
      </div>
      <details className="builder-advanced">
        <summary>Дополнительно · JSON Schema и UI</summary>
        <p>Техническое представление только для просмотра. Все изменения выполняются через конструктор.</p>
        <div className="builder-json-view"><section><strong>JSON Schema</strong><pre>{JSON.stringify(contract.schema, null, 2)}</pre></section><section><strong>UI JSON</strong><pre>{JSON.stringify(contract.ui, null, 2)}</pre></section></div>
      </details>
      <footer className="builder-footer"><button className="primary-btn" disabled={busy || !name.trim() || (!viewOnlyLegacy && fields.length === 0)}>{submitLabel}</button></footer>
    </form>
  );
}
