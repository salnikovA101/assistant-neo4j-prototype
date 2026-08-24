declare module "vis-network/standalone" {
  export class DataSet<T = Record<string, unknown>> {
    constructor(data?: T[]);
  }
  export class Network {
    constructor(container: HTMLElement, data: unknown, options?: unknown);
    on(event: string, handler: (params: { nodes: string[]; edges: string[] }) => void): void;
    once(event: string, handler: () => void): void;
    fit(options?: unknown): void;
    redraw(): void;
    setSize(width: string, height: string): void;
    selectNodes(ids: string[]): void;
    selectEdges(ids: string[]): void;
    focus(id: string, options?: unknown): void;
    destroy(): void;
  }
}
