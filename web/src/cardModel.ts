export type CardWidget =
  | "short_text"
  | "long_text"
  | "number"
  | "list"
  | "choice"
  | "boolean"
  | "date";

export type CardUiField = {
  widget?: CardWidget;
  unit?: string;
  options?: string[];
};

export type CardUi = {
  version?: number;
  layout?: string;
  titleField?: string;
  order?: string[];
  fields?: Record<string, CardUiField>;
};

export type CardField = {
  key: string;
  label: string;
  description: string;
  widget: CardWidget;
  required: boolean;
  unit: string;
  options: string[];
};

export type CardDefinition = {
  fields: CardField[];
  titleField: string;
  titleInstruction: string;
  editable: boolean;
};

function asObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function schemaTypes(schema: Record<string, unknown>): string[] {
  const raw = schema.type;
  return Array.isArray(raw) ? raw.map(String) : raw ? [String(raw)] : [];
}

function inferWidget(schema: Record<string, unknown>, configured?: CardWidget): CardWidget {
  if (configured) return configured;
  const types = schemaTypes(schema);
  if (types.includes("array")) return "list";
  if (types.includes("number") || types.includes("integer")) return "number";
  if (types.includes("boolean")) return "boolean";
  if (schema.format === "date") return "date";
  if (Array.isArray(schema.enum)) return "choice";
  return "long_text";
}

export function normalizeCardUi(raw: Record<string, unknown> | undefined): CardUi {
  const ui = asObject(raw);
  return {
    version: typeof ui.version === "number" ? ui.version : undefined,
    layout: typeof ui.layout === "string" ? ui.layout : undefined,
    titleField: typeof ui.titleField === "string" ? ui.titleField : undefined,
    order: Array.isArray(ui.order) ? ui.order.map(String) : undefined,
    fields: Object.fromEntries(
      Object.entries(asObject(ui.fields)).map(([key, value]) => {
        const field = asObject(value);
        return [key, {
          widget: typeof field.widget === "string" ? field.widget as CardWidget : undefined,
          unit: typeof field.unit === "string" ? field.unit : undefined,
          options: Array.isArray(field.options) ? field.options.map(String) : undefined,
        }];
      })
    ),
  };
}

export function readCardDefinition(
  schemaInput: Record<string, unknown>,
  uiInput: Record<string, unknown> = {}
): CardDefinition {
  const schema = asObject(schemaInput);
  const properties = asObject(schema.properties);
  const ui = normalizeCardUi(uiInput);
  const titleField = ui.titleField || ("title" in properties ? "title" : "");
  const preferred = (ui.order || []).filter((key) => key in properties && key !== titleField);
  const remaining = Object.keys(properties).filter((key) => key !== titleField && !preferred.includes(key));
  const order = [...preferred, ...remaining];
  const required = new Set(Array.isArray(schema.required) ? schema.required.map(String) : []);
  const editable = schemaTypes(schema).includes("object")
    && Object.values(properties).every((value) => {
      const child = asObject(value);
      return !schemaTypes(child).includes("object") && !("properties" in child);
    });
  const fields = order.map((key) => {
    const child = asObject(properties[key]);
    const configured = ui.fields?.[key];
    const enumOptions = Array.isArray(child.enum)
      ? child.enum.filter((value) => value != null).map(String)
      : [];
    return {
      key,
      label: typeof child.title === "string" ? child.title : key,
      description: typeof child.description === "string" ? child.description : "",
      widget: inferWidget(child, configured?.widget),
      required: required.has(key),
      unit: configured?.unit || "",
      options: configured?.options || enumOptions,
    } satisfies CardField;
  });
  const titleSchema = asObject(properties[titleField]);
  return {
    fields,
    titleField,
    titleInstruction: typeof titleSchema.description === "string"
      ? titleSchema.description
      : "Кратко назови конкретный результат",
    editable,
  };
}

function propertyFor(field: CardField): Record<string, unknown> {
  const base: Record<string, unknown> = {
    title: field.label.trim() || field.key,
    description: field.description.trim(),
  };
  if (field.widget === "number") return { ...base, type: ["number", "null"] };
  if (field.widget === "list") return {
    ...base,
    type: ["array", "null"],
    items: { type: "string" },
  };
  if (field.widget === "boolean") return { ...base, type: ["boolean", "null"] };
  if (field.widget === "date") return { ...base, type: ["string", "null"], format: "date" };
  if (field.widget === "choice") return {
    ...base,
    type: ["string", "null"],
    enum: [...field.options.filter((item) => item.trim()).map((item) => item.trim()), null],
  };
  return { ...base, type: ["string", "null"] };
}

export function buildCardContract(fields: CardField[], titleInstruction: string) {
  const properties: Record<string, unknown> = {
    title: {
      type: ["string", "null"],
      title: "Название карточки",
      description: titleInstruction.trim() || "Кратко назови конкретный результат",
    },
  };
  const required = ["title"];
  const uiFields: Record<string, CardUiField> = {};
  for (const field of fields) {
    properties[field.key] = propertyFor(field);
    if (field.required) required.push(field.key);
    uiFields[field.key] = {
      widget: field.widget,
      ...(field.unit.trim() ? { unit: field.unit.trim() } : {}),
      ...(field.options.length ? { options: field.options } : {}),
    };
  }
  return {
    schema: {
      type: "object",
      properties,
      required,
      additionalProperties: false,
    } as Record<string, unknown>,
    ui: {
      version: 1,
      layout: "stack",
      titleField: "title",
      order: fields.map((field) => field.key),
      fields: uiFields,
    } as Record<string, unknown>,
  };
}

export function nextFieldKey(fields: CardField[]): string {
  const used = new Set(fields.map((field) => field.key));
  let number = 1;
  while (used.has(`field_${number}`)) number += 1;
  return `field_${number}`;
}

export function blankData(schemaInput: Record<string, unknown>): Record<string, unknown> {
  const properties = asObject(schemaInput.properties);
  return Object.fromEntries(Object.entries(properties).map(([key, raw]) => {
    const types = schemaTypes(asObject(raw));
    return [key, types.includes("array") ? [] : null];
  }));
}

export function fallbackSchemaForData(data: Record<string, unknown>): Record<string, unknown> {
  const properties = Object.fromEntries(Object.entries(data).map(([key, value]) => [key, {
    title: key === "title" ? "Название карточки" : key.replaceAll("_", " "),
    type: Array.isArray(value)
      ? ["array", "null"]
      : typeof value === "number"
        ? ["number", "null"]
        : typeof value === "boolean"
          ? ["boolean", "null"]
          : ["string", "null"],
    ...(Array.isArray(value) ? { items: { type: "string" } } : {}),
  }]));
  return { type: "object", properties, additionalProperties: true };
}
