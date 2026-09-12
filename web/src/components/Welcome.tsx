import { IconGraph, IconSearch } from "./Icons";

export function Welcome({ mode, onMode, stagedEnabled, onHelp }: {
  mode: "auto" | "staged";
  onMode: (mode: "auto" | "staged") => void;
  stagedEnabled: boolean;
  onHelp: () => void;
}) {
  return (
    <div className="welcome welcome-modes">
      <div className="welcome-modes-header">
        <h2 id="welcome-modes-title">Режимы работы</h2>
        <button type="button" className="welcome-help-link" onClick={onHelp}>Полная справка по работе с ассистентом</button>
      </div>
      <div className="welcome-mode-grid" role="group" aria-labelledby="welcome-modes-title">
        <button
          type="button"
          className="welcome-mode-card"
          aria-pressed={mode === "staged"}
          disabled={!stagedEnabled}
          onClick={() => onMode("staged")}
        >
          <strong className="welcome-mode-heading"><IconGraph />Исследование</strong>
          <span className="welcome-mode-description">Для работы над одной задачей по шагам: изучайте ответы, уточняйте и сравнивайте варианты в одном чате.</span>
          {!stagedEnabled && <span className="welcome-mode-description">Режим отключён администратором.</span>}
        </button>
        <button
          type="button"
          className="welcome-mode-card"
          aria-pressed={mode === "auto"}
          onClick={() => onMode("auto")}
        >
          <strong className="welcome-mode-heading"><IconSearch />Вопрос по базе</strong>
          <span className="welcome-mode-description">Для отдельного вопроса: получите развёрнутый ответ за один раз. Если переходите к другой теме, начните новый чат.</span>
        </button>
      </div>
    </div>
  );
}
