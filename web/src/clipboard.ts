/** Copy text on both HTTPS and plain HTTP (LAN hosts are not a secure context). */

const CLIPBOARD_TIMEOUT_MS = 800;

export async function copyText(text: string): Promise<void> {
  const value = text ?? "";
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await Promise.race([
        navigator.clipboard.writeText(value),
        new Promise<never>((_, reject) => {
          window.setTimeout(() => reject(new Error("clipboard timeout")), CLIPBOARD_TIMEOUT_MS);
        }),
      ]);
      return;
    } catch {
      // Denied, hung permission prompt, or missing — textarea fallback works on http://host.
    }
  }
  fallbackCopy(value);
}

function fallbackCopy(text: string): void {
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.cssText = "position:fixed;top:0;left:0;width:2em;height:2em;padding:0;border:0;opacity:0";
  document.body.appendChild(field);
  field.focus();
  field.select();
  field.setSelectionRange(0, field.value.length);
  const ok = document.execCommand("copy");
  field.remove();
  if (!ok) throw new Error("copy failed");
}
