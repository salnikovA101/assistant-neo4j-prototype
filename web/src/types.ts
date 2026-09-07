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
  run_id: string;
  models: UiModel[];
};

export type GraphNode = {
  id: string;
  label: string;
  caption: string;
  labels: string[];
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
  from_labels?: string[];
  to_labels?: string[];
  hub_name?: string;
  properties: Record<string, unknown>;
};

export type GraphView = {
  id: string;
  label: string;
  score?: number;
  source_chain_id?: string;
  unit_no?: number | null;
  is_new?: boolean;
  origin?: {
    step_id?: string | null;
    step_no?: number | null;
    question?: string;
    branch_id?: string | null;
    branch_name?: string | null;
    answer_checkpoint_id?: string | null;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type GraphFilters = {
  node_labels: string[];
  relationship_types: string[];
  sources: string[];
  min_confidence: number | null;
};

export type GraphFacetItem = { value: string; count: number };

export type GraphFacets = {
  matchingRelationships: number;
  matchingNodes: number;
  nodeLabels: GraphFacetItem[];
  relationshipTypes: GraphFacetItem[];
  sources: {
    items: GraphFacetItem[];
    nextCursor?: string | null;
    hasMore: boolean;
  };
};

export type GraphExpansion = {
  anchorNodeId: string;
  returned: number;
  totalMatching: number;
  hasMore: boolean;
};

export type GraphCollectionItem =
  | { kind: "node"; key: string; node: GraphNode }
  | { kind: "edge"; key: string; edge: GraphEdge };

export type GraphPayload = {
  views: GraphView[];
  all: { nodes: GraphNode[]; edges: GraphEdge[] };
  mode?: "auto" | "staged";
  effectiveScope?: "context" | "new_in_answer" | "unit" | "all_branches";
  page?: { nextCursor?: string | null; hasMore: boolean };
  expansion?: GraphExpansion;
  runId?: string;
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
  branchId?: string;
  modelId?: string;
  modelLabel?: string;
  retrievalRunId?: string;
  sqStatusWarning?: string;
  cardDraft?: CardDraft;
  cardTemplateName?: string;
  cardRequest?: {
    templateVersionId: string;
    templateName: string;
    version: number;
    schema: Record<string, unknown>;
    ui?: Record<string, unknown>;
  };
  cardReference?: {
    revisionId: string;
    templateVersionId?: string;
    title: string;
    revision: number;
    data: Record<string, unknown>;
    provenance?: Record<string, unknown>;
  };
};

export type ExploreField = "all" | "name" | "label" | "rel" | "evidence" | "source";

export type Branch = {
  id: string;
  conversationId: string;
  name: string;
  mode: "auto" | "staged";
  createdFromCheckpointId?: string | null;
  headCheckpointId?: string | null;
  createdAt: number;
  updatedAt: number;
};

export type ResearchStep = {
  id: string;
  displayNo: number;
  parentStepId?: string | null;
  branchId: string;
  question: { messageId: string; preview: string };
  answer?: { messageId: string; preview: string; status: string } | null;
  userCheckpointId?: string | null;
  answerCheckpointId?: string | null;
  graphCheckpointId?: string | null;
  resumeCheckpointId?: string | null;
  unitNos: number[];
  graphUnitCount: number;
  createdAt: number;
};

export type ResearchBranch = Branch & {
  originStepId?: string | null;
  headStepId?: string | null;
  unitCount: number;
};

export type ResearchMap = {
  conversationId: string;
  activeBranchId: string;
  branches: ResearchBranch[];
  steps: ResearchStep[];
};

export type AgendaItem = {
  ref: string;
  text: string;
  status: "not_closed" | "partial" | "closed";
  statusOrigin: "assistant" | "user" | "legacy";
  statusReason: string;
  statusSourceRefs: string[];
  statusMessageId?: string | null;
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
  toolCall: {
    id?: string;
    name: string;
    arguments?: {
      open_sq_refs?: string[];
      /** Legacy persisted approvals are normalized by the server. */
      open_sq_ids?: string[];
      new_subquestions?: string[];
      subquestions?: string[];
    };
  };
};

export type ConversationSummary = {
  id: string;
  title: string;
  updatedAt: number;
  createdAt: number;
  activeBranchId?: string;
  headCheckpointId?: string | null;
  mode?: "auto" | "staged";
  runId: string;
  accountRunId: string;
  readOnly: boolean;
  readOnlyReason?: "run_id_changed" | null;
};

export type TurnFailure = {
  reason: "aborted" | "error" | "cancelled" | string;
  message: string;
  text: string;
  createdAt: number;
};

export type ConversationDetail = ConversationSummary & {
  messages: ChatMessage[];
  branches: Branch[];
  activeBranchId: string;
  headCheckpointId?: string | null;
  branchHeadCheckpointId?: string | null;
  viewCheckpointId?: string | null;
  atBranchHead?: boolean;
  agenda: AgendaItem[];
  pendingApproval?: PendingApproval | null;
  turnFailures?: TurnFailure[];
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
  template?: {
    name: string;
    version: number;
    schema: Record<string, unknown>;
    ui: Record<string, unknown>;
  };
  latestRevision: {
    id: string;
    revision: number;
    data: Record<string, unknown>;
    provenance: Record<string, unknown>;
    gaps: unknown[];
  };
};

export type Account = { id: string; username: string; runId: string };
