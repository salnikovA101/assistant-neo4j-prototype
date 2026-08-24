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
  staged_enabled: boolean;
  cards_enabled: boolean;
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
  page?: { nextCursor?: string | null; hasMore: boolean };
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
  status?: "streaming" | "waiting_approval" | "done" | "error" | "aborted" | "cancelled";
  elapsedSec?: number;
  checkpointId?: string;
  cardDraft?: CardDraft;
  cardTemplateName?: string;
};

export type ExploreField = "all" | "name" | "label" | "rel" | "evidence" | "source";

export type Branch = {
  id: string;
  conversationId: string;
  name: string;
  createdFromCheckpointId?: string | null;
  headCheckpointId?: string | null;
  createdAt: number;
  updatedAt: number;
};

export type AgendaItem = {
  id: string;
  text: string;
  status: "open" | "closed";
  position: number;
  questionCount: number;
  unitCount: number;
  graphSnapshotId?: string | null;
  reviewRecommended: boolean;
};

export type PendingApproval = {
  id: string;
  revision: number;
  status: string;
  assistantMessageId: string;
  toolCall: { id?: string; name: string; arguments?: { subquestions?: string[] } };
};

export type ConversationSummary = {
  id: string;
  title: string;
  updatedAt: number;
  createdAt: number;
  activeBranchId?: string;
  headCheckpointId?: string | null;
};

export type ConversationDetail = ConversationSummary & {
  messages: ChatMessage[];
  branches: Branch[];
  activeBranchId: string;
  headCheckpointId?: string | null;
  agenda: AgendaItem[];
  pendingApproval?: PendingApproval | null;
};

export type CardTemplate = {
  id: string;
  name: string;
  description: string;
  system: boolean;
  latestVersion: {
    id: string;
    version: number;
    schema: Record<string, unknown>;
    ui: Record<string, unknown>;
    instructions: string;
  };
};

export type CardDraft = {
  id: string;
  originCheckpointId?: string | null;
  templateVersionId: string;
  data: Record<string, unknown>;
  provenance: Record<string, unknown>;
  gaps: unknown[];
  status: string;
  savedCardId?: string;
  savedRevisionId?: string;
};

export type SavedCard = {
  id: string;
  title: string;
  templateVersionId: string;
  latestRevision: {
    id: string;
    revision: number;
    data: Record<string, unknown>;
    provenance: Record<string, unknown>;
    gaps: unknown[];
  };
};

export type Account = { id: string; username: string };
