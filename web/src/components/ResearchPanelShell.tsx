import type { ReactNode } from "react";
import { IconClose } from "./Icons";

export type ResearchTab = "map" | "directions" | "data";

export function ResearchPanelShell({
  tab,
  branchName,
  staged,
  openDirections,
  dataAvailable,
  onTab,
  onClose,
  children,
}: {
  tab: ResearchTab;
  branchName: string;
  staged: boolean;
  openDirections: number;
  dataAvailable: boolean;
  onTab: (tab: ResearchTab) => void;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <aside className="graph-pane research-shell" aria-label="Ход работы">
      <header className="research-shell-header">
        <div className="panel-title">
          <strong>Ход работы</strong>
          <span>{branchName}{staged ? " · с планом" : " · сразу"}</span>
        </div>
        <nav className="side-pane-tabs" aria-label="Раздел хода работы">
          <button type="button" className={tab === "map" ? "is-on" : ""} onClick={() => onTab("map")}>Карта</button>
          {staged && (
            <button type="button" className={tab === "directions" ? "is-on" : ""} onClick={() => onTab("directions")}>
              План{openDirections > 0 && <b>{openDirections}</b>}
            </button>
          )}
          <button
            type="button"
            className={tab === "data" ? "is-on" : ""}
            disabled={!dataAvailable}
            title={dataAvailable ? "Факты из базы" : "В этом диалоге ещё нет фактов из базы"}
            onClick={() => onTab("data")}
          >Факты</button>
        </nav>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть ход работы"><IconClose /></button>
      </header>
      <div className="research-shell-content">{children}</div>
    </aside>
  );
}
