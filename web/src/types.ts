export type SearchDepth = "low" | "medium" | "high";

export type UiModel = {
  id: string;
  label: string;
  think: boolean;
  reasoning_effort: string;
  reasoning_effort_options: string[];
};

export type UiConfig = {
  think: boolean;
  reasoning_effort: string;
  reasoning_effort_options: string[];
  search_depth: SearchDepth;
  search_depth_options: SearchDepth[];
  max_searches_per_answer: number;
  audio_enabled: boolean;
  current_profile: string;
  llm_key_configured: boolean;
  username: string;
  models: UiModel[];
};

export type GraphNode = {
  id: string;
  label: string;
  caption: string;
  group: string;
  color: string;
  properties: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  from: string;
  to: string;
  label: string;
  role?: string;
  chain_ids?: string[];
  from_name?: string;
  to_name?: string;
  from_group?: string;
  to_group?: string;
  hub_name?: string;
  properties: Record<string, unknown>;
};

export type GraphView = {
  id: string;
  label: string;
  score?: number;
  source_chain_id?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type GraphPayload = {
  views: GraphView[];
  all: { nodes: GraphNode[]; edges: GraphEdge[] };
};

export type ChatRole = "user" | "assistant";

export type ToolCard = {
  id: string;
  name: string;
  status: "running" | "done" | "error";
  args?: unknown;
  result?: string;
  detail?: string;
};

export type ThinkStep = { kind: "think"; text: string };
export type ToolStep = ToolCard & { kind: "tool" };
export type ChatStep = ThinkStep | ToolStep;

export type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  thinking?: string;
  tools?: ToolCard[];
  steps?: ChatStep[];
  graphRunId?: string;
  graphChainCount?: number;
  status?: "streaming" | "done" | "error" | "aborted";
  elapsedSec?: number;
};

export type ExploreField = "all" | "name" | "rel" | "evidence";

export type ConversationSummary = {
  id: string;
  title: string;
  updatedAt: number;
  createdAt: number;
};

export type ConversationDetail = ConversationSummary & { messages: ChatMessage[] };

export type Account = { id: string; username: string };
