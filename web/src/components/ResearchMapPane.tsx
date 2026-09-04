import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { fetchResearchMap } from "../api";
import type { ResearchBranch, ResearchMap, ResearchStep } from "../types";
import { IconClose, IconEdit, IconGraph } from "./Icons";
import { MODE_LABELS } from "../uiLabels";

const BRANCH_COLORS = ["#6ea8ff", "#8bd5ca", "#c6a0f6", "#f5bde6", "#eed49f", "#91d7e3"];
const COLUMN_PITCH = 256;
const ROW_PITCH = 236;
const CARD_WIDTH = 216;
const CARD_HEIGHT = 174;
const CANVAS_LEFT = 48;
const CANVAS_TOP = 42;
const BRANCH_LABEL_HEIGHT = 28;
const MIN_SCALE = 0.35;
const MAX_SCALE = 1.8;

type ViewState = { x: number; y: number; scale: number };

function visibleBranchName(branch: ResearchBranch, lane: number): string {
  if (lane === 0 && branch.name.trim().toLowerCase() === "main") return "Основной вариант";
  return branch.name;
}

function clampScale(value: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));
}

export function ResearchMapPane({
  conversationId,
  activeBranchId,
  selectedStepId,
  refreshKey,
  dataCheckpointId,
  onSelectStep,
  onSelectEmptyBranch,
  onOpenGraph,
  onOpenData,
  onRenameBranch,
  onClose,
  embedded = false,
}: {
  conversationId: string;
  activeBranchId: string;
  selectedStepId: string;
  refreshKey: number;
  dataCheckpointId: string;
  onSelectStep: (branchId: string, step: ResearchStep) => void;
  onSelectEmptyBranch: (branch: ResearchBranch) => void;
  onOpenGraph: (checkpointId: string, stepId: string) => void;
  onOpenData: () => void;
  onRenameBranch: (id: string, name: string) => Promise<void>;
  onClose: () => void;
  embedded?: boolean;
}) {
  const [payload, setPayload] = useState<ResearchMap | null>(null);
  const [error, setError] = useState("");
  const [editingId, setEditingId] = useState("");
  const [name, setName] = useState("");
  const [view, setView] = useState<ViewState>({ x: 16, y: 16, scale: 1 });
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  useEffect(() => {
    let alive = true;
    setError("");
    fetchResearchMap(conversationId, activeBranchId)
      .then((data) => {
        if (alive) setPayload(data);
      })
      .catch((reason: Error) => {
        if (alive) setError(reason.message);
      });
    return () => {
      alive = false;
    };
  }, [conversationId, activeBranchId, refreshKey]);

  const model = useMemo(() => {
    const branches = [...(payload?.branches || [])].sort((left, right) => {
      const rootOrder = Number(Boolean(left.originStepId)) - Number(Boolean(right.originStepId));
      return rootOrder || left.createdAt - right.createdAt || left.id.localeCompare(right.id);
    });
    const steps = payload?.steps || [];
    const branchById = new Map(branches.map((branch) => [branch.id, branch]));
    const stepById = new Map(steps.map((step) => [step.id, step]));

    const rowByStep = new Map<string, number>();
    const resolveRow = (stepId: string, visiting = new Set<string>()): number => {
      const known = rowByStep.get(stepId);
      if (known !== undefined) return known;
      if (visiting.has(stepId)) return 0;
      visiting.add(stepId);
      const step = stepById.get(stepId);
      const row = step?.parentStepId && stepById.has(step.parentStepId)
        ? resolveRow(step.parentStepId, visiting) + 1
        : 0;
      rowByStep.set(stepId, row);
      return row;
    };
    steps.forEach((step) => resolveRow(step.id));

    const laneByBranch = new Map<string, number>();
    const usedLanes = new Set<number>();
    const resolveLane = (branchId: string, visiting = new Set<string>()): number => {
      const known = laneByBranch.get(branchId);
      if (known !== undefined) return known;
      if (visiting.has(branchId)) return 0;
      visiting.add(branchId);
      const branch = branchById.get(branchId);
      const origin = branch?.originStepId ? stepById.get(branch.originStepId) : undefined;
      let candidate = origin && origin.branchId !== branchId
        ? resolveLane(origin.branchId, visiting) + 1
        : 0;
      while (usedLanes.has(candidate)) candidate += 1;
      usedLanes.add(candidate);
      laneByBranch.set(branchId, candidate);
      return candidate;
    };
    branches.forEach((branch) => resolveLane(branch.id));

    const branchOrder = new Map(branches.map((branch) => [branch.id, resolveLane(branch.id)]));
    const firstStepByBranch = new Map<string, string>();
    const ownSteps = new Map<string, ResearchStep[]>();
    steps.forEach((step) => {
      if (!firstStepByBranch.has(step.branchId)) firstStepByBranch.set(step.branchId, step.id);
      ownSteps.set(step.branchId, [...(ownSteps.get(step.branchId) || []), step]);
    });

    const emptyBranches = branches.filter((branch) => !(ownSteps.get(branch.id)?.length) && branch.originStepId);
    const maxLane = Math.max(0, ...branches.map((branch) => resolveLane(branch.id)));
    const emptyRows = emptyBranches.map((branch) => {
      const originRow = branch.originStepId ? rowByStep.get(branch.originStepId) ?? 0 : 0;
      return originRow + 1;
    });
    const maxRow = Math.max(0, ...steps.map((step) => rowByStep.get(step.id) || 0), ...emptyRows);
    const canvasWidth = CANVAS_LEFT + (maxLane + 1) * COLUMN_PITCH + 34;
    const canvasHeight = CANVAS_TOP + (maxRow + 1) * ROW_PITCH + 40;

    const cardPosition = (step: ResearchStep) => {
      const lane = resolveLane(step.branchId);
      const row = rowByStep.get(step.id) || 0;
      return {
        lane,
        row,
        x: CANVAS_LEFT + lane * COLUMN_PITCH,
        y: CANVAS_TOP + row * ROW_PITCH + BRANCH_LABEL_HEIGHT,
      };
    };

    return {
      branches,
      steps,
      branchById,
      branchOrder,
      rowByStep,
      firstStepByBranch,
      emptyBranches,
      stepById,
      maxRow,
      canvasWidth,
      canvasHeight,
      cardPosition,
    };
  }, [payload]);

  const fitView = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport || model.steps.length === 0) return;
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    const scale = clampScale(Math.min((width - 34) / model.canvasWidth, (height - 34) / model.canvasHeight, 1));
    setView({
      scale,
      x: Math.max(17, (width - model.canvasWidth * scale) / 2),
      y: Math.max(17, (height - model.canvasHeight * scale) / 2),
    });
  }, [model.canvasHeight, model.canvasWidth, model.steps.length]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || model.steps.length === 0) return;
    let frame = 0;
    const scheduleFit = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(fitView);
    };
    const observer = new ResizeObserver(scheduleFit);
    observer.observe(viewport);
    scheduleFit();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [fitView, model.steps.length]);

  const zoomAtCenter = (factor: number) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const centerX = viewport.clientWidth / 2;
    const centerY = viewport.clientHeight / 2;
    setView((current) => {
      const scale = clampScale(current.scale * factor);
      const worldX = (centerX - current.x) / current.scale;
      const worldY = (centerY - current.y) / current.scale;
      return { scale, x: centerX - worldX * scale, y: centerY - worldY * scale };
    });
  };

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const pointX = event.clientX - rect.left;
    const pointY = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * 0.0015);
    setView((current) => {
      const scale = clampScale(current.scale * factor);
      const worldX = (pointX - current.x) / current.scale;
      const worldY = (pointY - current.y) / current.scale;
      return { scale, x: pointX - worldX * scale, y: pointY - worldY * scale };
    });
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest("button,input,article")) return;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.classList.add("is-panning");
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    setView((current) => ({ ...current, x: current.x + dx, y: current.y + dy }));
  };

  const endPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    event.currentTarget.classList.remove("is-panning");
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const connectionPath = (source: ResearchStep, targetX: number, targetY: number): string => {
    const sourcePosition = model.cardPosition(source);
    const sourceX = sourcePosition.x + CARD_WIDTH / 2;
    const sourceY = sourcePosition.y + CARD_HEIGHT;
    const destinationX = targetX + CARD_WIDTH / 2;
    const destinationY = targetY;
    if (Math.abs(sourceX - destinationX) < 1) return `M ${sourceX} ${sourceY} L ${destinationX} ${destinationY}`;
    const middleY = sourceY + Math.max(26, (destinationY - sourceY) * 0.5);
    return `M ${sourceX} ${sourceY} C ${sourceX} ${middleY}, ${destinationX} ${middleY}, ${destinationX} ${destinationY}`;
  };

  return (
    <section className={`${embedded ? "research-panel-view" : "graph-pane"} research-map-pane`} aria-label="Карта хода">
      {!embedded && <header className="research-map-header">
        <div className="panel-title"><strong>Карта хода</strong><span>{model.branches.length} вариантов · {model.steps.length} шагов</span></div>
        <div className="side-pane-tabs" aria-label="Раздел правой панели">
          <button type="button" className="is-on">Карта</button>
          <button type="button" disabled={!dataCheckpointId} title={dataCheckpointId ? "Вернуться к данным" : "Откройте данные из нужного шага"} onClick={onOpenData}>Данные</button>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Скрыть карту"><IconClose /></button>
      </header>}

      {error && <p className="explorer-status is-error">{error}</p>}
      {!payload && !error && <p className="explorer-status">Загрузка карты…</p>}
      {payload && model.steps.length === 0 && <div className="research-map-empty"><strong>Здесь появится ход работы</strong><span>Каждый вопрос и ответ станет отдельным шагом.</span></div>}

      {payload && model.steps.length > 0 && (
        <div ref={viewportRef} className="research-map-viewport" role="tree" aria-label="Варианты и шаги диалога" onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={endPan} onPointerCancel={endPan}>
          <div className="research-map-controls" aria-label="Масштаб карты">
            <button type="button" onClick={() => zoomAtCenter(0.82)} aria-label="Уменьшить карту">−</button>
            <span>{Math.round(view.scale * 100)}%</span>
            <button type="button" onClick={() => zoomAtCenter(1.22)} aria-label="Увеличить карту">+</button>
            <button type="button" onClick={fitView}>Вписать</button>
          </div>
          <div className="research-map-canvas" style={{ width: `${model.canvasWidth}px`, height: `${model.canvasHeight}px`, transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
            <div className="research-grid-lines" aria-hidden="true">
              {Array.from({ length: model.maxRow + 1 }, (_, row) => <i key={row} style={{ top: `${CANVAS_TOP + row * ROW_PITCH}px` }}><span>{row + 1}</span></i>)}
            </div>

            <svg className="research-connections" width={model.canvasWidth} height={model.canvasHeight} aria-hidden="true">
              {model.steps.map((step) => {
                if (!step.parentStepId) return null;
                const parent = model.stepById.get(step.parentStepId);
                if (!parent) return null;
                const target = model.cardPosition(step);
                const lane = model.branchOrder.get(step.branchId) || 0;
                return <path key={step.id} d={connectionPath(parent, target.x, target.y)} stroke={BRANCH_COLORS[lane % BRANCH_COLORS.length]} />;
              })}
              {model.emptyBranches.map((branch) => {
                const parent = branch.originStepId ? model.stepById.get(branch.originStepId) : undefined;
                if (!parent) return null;
                const lane = model.branchOrder.get(branch.id) || 0;
                const row = (model.rowByStep.get(parent.id) || 0) + 1;
                const targetX = CANVAS_LEFT + lane * COLUMN_PITCH;
                const targetY = CANVAS_TOP + row * ROW_PITCH + BRANCH_LABEL_HEIGHT;
                return <path key={branch.id} d={connectionPath(parent, targetX, targetY)} stroke={BRANCH_COLORS[lane % BRANCH_COLORS.length]} />;
              })}
            </svg>

            {model.steps.map((step) => {
              const branch = model.branchById.get(step.branchId);
              if (!branch) return null;
              const position = model.cardPosition(step);
              const lane = position.lane;
              const color = BRANCH_COLORS[lane % BRANCH_COLORS.length];
              const showBranchLabel = model.firstStepByBranch.get(step.branchId) === step.id;
              const isSelected = selectedStepId === step.id;
              return (
                <div key={step.id} className={`research-grid-cell ${isSelected ? "is-selected" : ""}`} style={{ left: `${position.x}px`, top: `${position.y - BRANCH_LABEL_HEIGHT}px`, "--branch-color": color, "--step-card-width": `${CARD_WIDTH}px`, "--step-card-height": `${CARD_HEIGHT}px` } as CSSProperties} role="treeitem" aria-current={isSelected ? "step" : undefined}>
                  <div className="research-grid-branch-label">
                    {showBranchLabel && (editingId === branch.id ? (
                      <form onSubmit={(event) => {
                        event.preventDefault();
                        void onRenameBranch(branch.id, name).then(() => {
                          setEditingId("");
                          setPayload((current) => current ? { ...current, branches: current.branches.map((item) => item.id === branch.id ? { ...item, name: name.trim() } : item) } : current);
                        }).catch(() => undefined);
                      }}>
                        <input autoFocus maxLength={64} value={name} onChange={(event) => setName(event.target.value)} aria-label="Название варианта" />
                        <button type="submit" disabled={!name.trim()}>✓</button>
                      </form>
                    ) : (
                      <><i /><span>{visibleBranchName(branch, lane)}</span><small>{MODE_LABELS[branch.mode]}</small><button type="button" onClick={() => { setEditingId(branch.id); setName(visibleBranchName(branch, lane)); }} aria-label={`Переименовать ${visibleBranchName(branch, lane)}`} title="Переименовать вариант"><IconEdit /></button></>
                    ))}
                  </div>
                  <article
                    className="research-step-card"
                    tabIndex={0}
                    aria-label={`Открыть вопрос ${step.displayNo}: ${step.question.preview || "без текста"}`}
                    onClick={() => onSelectStep(step.branchId, step)}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      onSelectStep(step.branchId, step);
                    }}
                  >
                    <header><span>Вопрос {step.displayNo}</span>{step.unitNos.length > 0 && <b>+{step.unitNos.length} {step.unitNos.length === 1 ? "цепочка" : "цепочек"}</b>}</header>
                    <strong>{step.question.preview || "Вопрос без текста"}</strong>
                    <p className={!step.answer?.preview ? "is-muted" : ""}>{step.answer?.preview || (step.answer?.status === "streaming" ? "Ассистент отвечает…" : "Ответ ещё не сформирован")}</p>
                    <footer><span>{step.answer?.status === "error" ? "Ошибка" : step.answer?.status === "streaming" ? "В работе" : "Готово"}</span>{step.graphCheckpointId && step.graphUnitCount > 0 && <button type="button" onClick={(event) => { event.stopPropagation(); onSelectStep(step.branchId, step); onOpenGraph(step.graphCheckpointId!, step.id); }}><IconGraph /> Данные</button>}</footer>
                  </article>
                </div>
              );
            })}

            {model.emptyBranches.map((branch) => {
              const parent = branch.originStepId ? model.stepById.get(branch.originStepId) : undefined;
              if (!parent) return null;
              const lane = model.branchOrder.get(branch.id) || 0;
              const row = (model.rowByStep.get(parent.id) || 0) + 1;
              const color = BRANCH_COLORS[lane % BRANCH_COLORS.length];
              const isSelected = activeBranchId === branch.id && !selectedStepId;
              const label = visibleBranchName(branch, lane);
              return (
                <div key={branch.id} className={`research-empty-grid-cell ${isSelected ? "is-selected" : ""}`} style={{ left: `${CANVAS_LEFT + lane * COLUMN_PITCH}px`, top: `${CANVAS_TOP + row * ROW_PITCH}px`, "--branch-color": color, "--step-card-width": `${CARD_WIDTH}px` } as CSSProperties}>
                  <div className="research-grid-branch-label"><i /><span>{label}</span><small>{MODE_LABELS[branch.mode]}</small></div>
                  <article
                    className="research-empty-node"
                    tabIndex={0}
                    role="treeitem"
                    aria-current={isSelected ? "step" : undefined}
                    aria-label={`Открыть вариант «${label}»: сообщений пока нет`}
                    onClick={() => onSelectEmptyBranch(branch)}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      onSelectEmptyBranch(branch);
                    }}
                  >Вариант создан — сообщений пока нет</article>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
