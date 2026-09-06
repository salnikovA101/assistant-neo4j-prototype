import { renderMarkdown } from "../format";
import { readCardDefinition } from "../cardModel";

function empty(value: unknown): boolean {
  return value == null || value === "" || (Array.isArray(value) && value.length === 0);
}

function dateText(value: unknown): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return String(value || "");
  const [year, month, day] = value.split("-").map(Number);
  return new Intl.DateTimeFormat("ru-RU").format(new Date(year, month - 1, day));
}

function ReadValue({ value, widget, unit }: { value: unknown; widget: string; unit: string }) {
  if (empty(value)) return <span className="card-empty">Не заполнено</span>;
  if (value && typeof value === "object" && !Array.isArray(value)) return (
    <div className="card-nested-answer">{Object.entries(value as Record<string, unknown>).map(([key, item]) => (
      <div key={key}><strong>{key.replaceAll("_", " ")}</strong><ReadValue value={item} widget={Array.isArray(item) ? "list" : "short_text"} unit="" /></div>
    ))}</div>
  );
  if (widget === "long_text") return (
    <div className="md card-long-answer" dangerouslySetInnerHTML={{ __html: renderMarkdown(String(value)) }} />
  );
  if (widget === "list") return (
    <ul className="card-list-answer">{(Array.isArray(value) ? value : [value]).map((item, index) => <li key={index}>{String(item)}</li>)}</ul>
  );
  if (widget === "boolean") return <span className={`card-value-chip ${value ? "is-yes" : "is-no"}`}>{value ? "Да" : "Нет"}</span>;
  if (widget === "choice") return <span className="card-value-chip">{String(value)}</span>;
  if (widget === "date") return <span>{dateText(value)}</span>;
  return <span>{String(value)}{widget === "number" && unit ? ` ${unit}` : ""}</span>;
}

function EditValue({
  value,
  widget,
  unit,
  options,
  onChange,
}: {
  value: unknown;
  widget: string;
  unit: string;
  options: string[];
  onChange: (value: unknown) => void;
}) {
  if (widget === "long_text") return <textarea value={String(value ?? "")} onChange={(event) => onChange(event.target.value || null)} />;
  if (widget === "number") return <div className="card-number-input"><input type="number" value={value == null ? "" : String(value)} onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} />{unit && <span>{unit}</span>}</div>;
  if (widget === "list") return <textarea value={(Array.isArray(value) ? value : []).join("\n")} placeholder="Один пункт на строку" onChange={(event) => onChange(event.target.value.split("\n").map((item) => item.trim()).filter(Boolean))} />;
  if (widget === "choice") return <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value || null)}><option value="">Не выбрано</option>{options.map((option) => <option key={option}>{option}</option>)}</select>;
  if (widget === "boolean") return <select value={value == null ? "" : value ? "true" : "false"} onChange={(event) => onChange(event.target.value === "" ? null : event.target.value === "true")}><option value="">Не выбрано</option><option value="true">Да</option><option value="false">Нет</option></select>;
  if (widget === "date") return <input type="date" value={String(value ?? "")} onChange={(event) => onChange(event.target.value || null)} />;
  return <input value={String(value ?? "")} onChange={(event) => onChange(event.target.value || null)} />;
}

export function CardVisual({
  templateName,
  schema,
  ui = {},
  data,
  editable = false,
  status,
  onChange,
}: {
  templateName: string;
  version?: number;
  schema: Record<string, unknown>;
  ui?: Record<string, unknown>;
  data: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  editable?: boolean;
  status?: string;
  onChange?: (key: string, value: unknown) => void;
}) {
  const definition = readCardDefinition(schema, ui);
  const titleKey = definition.titleField || "title";
  const title = data[titleKey];
  return (
    <article className={`visual-card ${editable ? "is-editable" : ""}`}>
      <header className="visual-card-head">
        <div>
          <span className="visual-card-kind">{templateName}</span>
          {editable ? (
            <input
              className="visual-card-title-input"
              value={String(title ?? "")}
              placeholder="Название карточки"
              aria-label="Название карточки"
              onChange={(event) => onChange?.(titleKey, event.target.value || null)}
            />
          ) : <h3>{empty(title) ? "Без названия" : String(title)}</h3>}
        </div>
        {status && <span className="visual-card-status">{status}</span>}
      </header>
      <div className="visual-card-fields">
        {definition.fields.map((field) => {
          return <section key={field.key} className="visual-card-field">
            <div className="visual-card-label">
              <strong>{field.label}</strong>
              {field.description && <button type="button" className="card-info" title={field.description} aria-label={`Пояснение: ${field.description}`}>i</button>}
            </div>
            <div className="visual-card-answer">
              {editable ? (
                <EditValue value={data[field.key]} widget={field.widget} unit={field.unit} options={field.options} onChange={(value) => onChange?.(field.key, value)} />
              ) : (
                <ReadValue value={data[field.key]} widget={field.widget} unit={field.unit} />
              )}
            </div>
          </section>;
        })}
      </div>
    </article>
  );
}
