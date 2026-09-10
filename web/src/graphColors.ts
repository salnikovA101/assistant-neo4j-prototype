import { normalizedLabels } from "./uiLabels";

export const DEFAULT_GRAPH_NODE_COLOR = "#a5abb6";

const BASE_LIGHTNESS = 0.72;
const BASE_CHROMA = 0.16;
const MIN_MIXED_CHROMA = 0.085;
const labelColorCache = new Map<string, string>();
const labelSetColorCache = new Map<string, string>();
const encoder = new TextEncoder();

type Oklab = { l: number; a: number; b: number };
type LinearRgb = { r: number; g: number; b: number };

function fnv1a(value: string): number {
  let hash = 0x811c9dc5;
  for (const byte of encoder.encode(value)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function hueFor(value: string): number {
  return (fnv1a(value) % 360) * (Math.PI / 180);
}

function labForHue(hue: number, chroma = BASE_CHROMA): Oklab {
  return {
    l: BASE_LIGHTNESS,
    a: Math.cos(hue) * chroma,
    b: Math.sin(hue) * chroma,
  };
}

function linearRgb(lab: Oklab): LinearRgb {
  const l = lab.l + 0.3963377774 * lab.a + 0.2158037573 * lab.b;
  const m = lab.l - 0.1055613458 * lab.a - 0.0638541728 * lab.b;
  const s = lab.l - 0.0894841775 * lab.a - 1.291485548 * lab.b;
  const l3 = l * l * l;
  const m3 = m * m * m;
  const s3 = s * s * s;
  return {
    r: 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
    g: -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
    b: -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3,
  };
}

function inGamut(rgb: LinearRgb): boolean {
  return rgb.r >= 0 && rgb.r <= 1 && rgb.g >= 0 && rgb.g <= 1 && rgb.b >= 0 && rgb.b <= 1;
}

function encodeSrgb(value: number): number {
  const bounded = Math.max(0, Math.min(1, value));
  return bounded <= 0.0031308 ? 12.92 * bounded : 1.055 * bounded ** (1 / 2.4) - 0.055;
}

function channelHex(value: number): string {
  return Math.round(encodeSrgb(value) * 255).toString(16).padStart(2, "0");
}

function labToHex(lab: Oklab): string {
  const hue = Math.atan2(lab.b, lab.a);
  let chroma = Math.hypot(lab.a, lab.b);
  let candidate = lab;
  let rgb = linearRgb(candidate);
  while (!inGamut(rgb) && chroma > 0.001) {
    chroma *= 0.94;
    candidate = labForHue(hue, chroma);
    rgb = linearRgb(candidate);
  }
  return `#${channelHex(rgb.r)}${channelHex(rgb.g)}${channelHex(rgb.b)}`;
}

export function colorForLabel(label: string): string {
  const clean = label.trim();
  if (!clean) return DEFAULT_GRAPH_NODE_COLOR;
  const cached = labelColorCache.get(clean);
  if (cached) return cached;
  const color = labToHex(labForHue(hueFor(clean)));
  labelColorCache.set(clean, color);
  return color;
}

export function colorForLabels(labels: string[] | undefined, fallback = DEFAULT_GRAPH_NODE_COLOR): string {
  const clean = normalizedLabels(labels);
  if (!clean.length) return fallback;
  if (clean.length === 1) return colorForLabel(clean[0]);

  const key = clean.join("\u001f");
  const cached = labelSetColorCache.get(key);
  if (cached) return cached;

  let a = 0;
  let b = 0;
  for (const label of clean) {
    const lab = labForHue(hueFor(label));
    a += lab.a;
    b += lab.b;
  }
  a /= clean.length;
  b /= clean.length;

  const chroma = Math.hypot(a, b);
  if (chroma < MIN_MIXED_CHROMA) {
    const hue = chroma > 1e-8 ? Math.atan2(b, a) : hueFor(key);
    a = Math.cos(hue) * MIN_MIXED_CHROMA;
    b = Math.sin(hue) * MIN_MIXED_CHROMA;
  }

  const color = labToHex({ l: BASE_LIGHTNESS, a, b });
  labelSetColorCache.set(key, color);
  return color;
}
