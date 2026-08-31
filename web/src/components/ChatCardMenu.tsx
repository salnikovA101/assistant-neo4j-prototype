import { IconCards } from "./Icons";

export function ChatCardMenu({
  enabled,
  onOpen,
}: {
  enabled: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      className="icon-btn"
      title="Шаблоны и карточки"
      aria-label="Открыть карточки диалога"
      disabled={!enabled}
      onClick={onOpen}
    >
      <IconCards />
    </button>
  );
}
