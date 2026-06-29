export type CaseStatus = "active" | "archived";

export interface CaseSummary {
  id: string;
  name: string;
  description?: string | null;
  status: CaseStatus;
  case_slug: string;
  doc_count: number;
  active_job_count?: number;
  updated_at: string;
  summary_snippet?: string | null;
}

export interface CaseWorkspaceSummary {
  case_id: string;
  case_name: string;
  case_slug: string;
  document_count: number;
  completed_document_count: number;
  evidence_count?: number;
  entity_count?: number;
  relationship_count?: number;
  top_entity_types?: Array<{ entity_type: string; count: number }>;
  latest_activity_at?: string | null;
  five_w_one_h: {
    who?: string | null;
    what?: string | null;
    when?: string | null;
    where?: string | null;
    why?: string | null;
    how?: string | null;
  };
  unknowns: string[];
  intelligence_summary?: string | null;
  investigation_summary?: string | null;
  summary_text?: string | null;
  last_refreshed_at?: string | null;
  source_job_id?: string | null;
}

export interface CaseDetail {
  id: string;
  name: string;
  description?: string | null;
  status: CaseStatus;
  case_slug: string;
  created_at?: string | null;
  updated_at: string;
  summary?: CaseWorkspaceSummary | null;
}

export type DocumentStatus =
  | "queued"
  | "parsing"
  | "inserting"
  | "indexing"
  | "complete"
  | "completed_with_warnings"
  | "failed";

export type IngestionJobStatus = DocumentStatus;
export type ProcessingMode = "multimodal" | "text_first";

export interface IngestionJob {
  id: string;
  case_id: string;
  document_id: string;
  ingest_profile?: string;
  processing_mode?: ProcessingMode;
  complexity_class?: string | null;
  eta_seconds?: number | null;
  queue_priority?: "low" | "normal" | "high" | null;
  advanced_overrides?: IngestionAdvancedOverrides | null;
  preflight?: IngestionPreflight | null;
  effective_config?: Record<string, unknown> | null;
  route_type?: string | null;
  status: IngestionJobStatus;
  progress?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  parse_duration_s?: number | null;
  insert_duration_s?: number | null;
  finalize_duration_s?: number | null;
  current_stage?: string | null;
  error?: string | null;
}

export type IngestionProfile = "balanced_fast" | "balanced_fast_intel" | "full_enrichment";

export interface IngestionAdvancedOverrides {
  parse_method?: "auto" | "txt" | "ocr" | "native" | "vlm-first" | "transcript-first";
  ocr_mode?: "off" | "auto" | "force";
  enable_vlm?: boolean;
  enable_vlm_visible_text?: boolean;
  enable_preinsert_summary?: boolean;
  vlm_parallelism?: number;
  max_parallel_insert?: number;
  summary_max_tokens?: number;
  queue_priority?: "low" | "normal" | "high";
  enable_gleaning?: boolean;
}

export interface IngestionPreflight {
  source_kind: string;
  mime_type: string;
  extension: string;
  complexity_class: "small" | "medium" | "large" | "very_large";
  eta_seconds: number;
  metrics?: Record<string, unknown>;
  warnings?: string[];
}

export interface IngestionJobLog {
  id: number;
  job_id: string;
  case_id: string;
  created_at: string;
  level: string;
  message: string;
}

export interface DocumentSummary {
  id: string;
  case_id: string;
  original_filename: string;
  stored_file_path: string;
  content_hash_sha256?: string | null;
  mime_type: string;
  size_bytes: number;
  confidence_source_reliability: string;
  confidence_information_validity: string;
  confidence_code: string;
  tags?: string | null;
  notes?: string | null;
  ingest_model_name?: string | null;
  ingestion_status: DocumentStatus;
  ingestion_error?: string | null;
  latest_processing_mode?: ProcessingMode | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentDuplicateCheckItem {
  client_id: string;
  original_filename?: string;
  size_bytes?: number;
  content_hash_sha256: string;
}

export interface DocumentDuplicateCheckResult {
  client_id: string;
  original_filename?: string | null;
  size_bytes?: number | null;
  content_hash_sha256: string;
  matches: DocumentSummary[];
}

export type DocumentSearchSource = "all" | "raw" | "processed";

export interface DocumentSearchResult {
  document_id: string;
  original_filename: string;
  confidence_code?: string | null;
  source_kind: "raw" | "processed";
  segment_key: string;
  stored_file_path: string;
  snippet: string;
  snippet_parts?: Array<{ text: string; match: boolean }>;
  score: number;
}

export interface DocumentSearchPreview {
  document_id: string;
  original_filename: string;
  confidence_code?: string | null;
  source_kind: "raw" | "processed";
  segment_key: string;
  content: string;
  match_ranges: Array<{ start: number; end: number }>;
  window_start: number;
  truncated: boolean;
}

export type GraphView = "pole";
export type ConfidenceBand = "low" | "medium" | "high";

export interface GraphNode {
  id: string;
  label: string;
  entity_type: string;
  entity_subtype?: string | null;
  summary?: string;
  degree?: number;
  meta?: Record<string, unknown>;
  evidence?: GraphEvidence[];
}

export interface GraphEdge {
  id: string;
  src_id: string;
  tgt_id: string;
  label: string;
  relation_type: string;
  relation_raw_phrase?: string;
  confidence_score?: number;
  confidence_band?: ConfidenceBand;
  weight?: number;
  timestamp?: string;
  evidence?: GraphEvidence[];
}

export interface GraphViewPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated?: boolean;
}

// Actor-centric types

export interface RelationshipRow {
  id: string;
  doc_id: string | null;
  timestamp: string | null;
  actor_id?: string;
  actor: string;
  actor_type?: string;
  actor_summary?: string | null;
  action: string;
  description?: string | null;
  source_id?: string | null;
  target_id?: string;
  target: string;
  target_type?: string;
  target_summary?: string | null;
  location: string | null;
  tags: string[];
  evidence?: GraphEvidence[];
}

export interface RelationshipsResponse {
  relationships: RelationshipRow[];
  totalBeforeLimit: number;
  totalBeforeFilter: number;
}

export interface StatsSummary {
  totalDocuments: { count: number };
  totalRelationships: { count: number };
  totalActors: { count: number };
  categories: { category: string; count: number }[];
}

export interface TagCluster {
  id: number;
  name: string;
  exemplars: string[];
  tagCount: number;
}

export interface ActorSearchResult {
  name: string;
  id: string;
  entity_type: string;
}

export interface GraphEvidence {
  file_path: string;
  reference_id: string;
  document_id?: string | null;
  confidence_code?: string | null;
  source_id?: string | null;
  snippet?: string | null;
}

export interface GraphEntityDetails extends GraphNode {
  description?: string | null;
  evidence: GraphEvidence[];
}

export interface GraphRelationshipDetails {
  src_id: string;
  tgt_id: string;
  description: string;
  relation_type?: string;
  relation_raw_phrase?: string;
  confidence_score?: number;
  confidence_band?: ConfidenceBand;
  keywords?: string[];
  evidence: GraphEvidence[];
}

export interface HighlightRelationship {
  src_id: string;
  tgt_id: string;
  edge_id?: string;
  relation_type?: string;
}

export interface HighlightReference {
  reference_id: string;
  file_path: string;
}

export interface HighlightChunk extends HighlightReference {
  snippet?: string;
}

export interface HighlightPayload {
  highlight_entities: string[];
  highlight_relationships: HighlightRelationship[];
  supporting_chunks?: HighlightChunk[];
  references: HighlightReference[];
}

export type ChatMode = "mix" | "local" | "global" | "hybrid" | "naive" | "bypass";
export type ChatRole = "user" | "assistant";

export interface ChatSummary {
  id: string;
  case_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  chat_id?: string;
  role: ChatRole;
  content: string;
  created_at: string;
  rag_metadata?: Record<string, unknown> | null;
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[];
}

export interface ChatMessageResponse {
  message: {
    id: string;
    role: "assistant";
    content: string;
    created_at: string;
  };
  highlight: HighlightPayload;
  references: HighlightReference[];
  chunks?: HighlightChunk[];
  model_name?: string | null;
}

export type AnalysisType = "link" | "flow" | "event";
export type AnalysisStatus =
  | "queued"
  | "generating"
  | "repair_queued"
  | "repairing"
  | "complete"
  | "failed";

export interface AnalysisChart {
  id: string;
  kind: string;
  title: string;
  item_ids: string[];
  mermaid_code: string;
  repair_attempts: number;
}

export interface AnalysisRecord {
  id: string;
  case_id: string;
  analysis_type: AnalysisType;
  prompt: string;
  title: string;
  status: AnalysisStatus;
  error?: string | null;
  rag_answer?: string | null;
  summary_text?: string | null;
  charts: AnalysisChart[];
  highlight: HighlightPayload;
  subgraph: GraphViewPayload;
  references: HighlightReference[];
  chunks: HighlightChunk[];
  model_name?: string | null;
  created_at: string;
  updated_at: string;
}

interface Envelope<T> {
  status: "success" | "error";
  message: string;
  data: T;
  metadata?: Record<string, unknown>;
}

const API_BASE = (import.meta as ImportMeta & { env: { VITE_API_BASE?: string } }).env.VITE_API_BASE || "";

function buildApiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function parseDispositionFilename(value: string | null) {
  if (!value) {
    return null;
  }
  const match = /filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?/i.exec(value);
  if (!match) {
    return null;
  }
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return match[1];
  }
}

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

let _backendOnline = true;
type ConnectionListener = (online: boolean) => void;
const _connectionListeners = new Set<ConnectionListener>();

export function isBackendOnline(): boolean {
  return _backendOnline;
}

export function onConnectionChange(fn: ConnectionListener): () => void {
  _connectionListeners.add(fn);
  return () => { _connectionListeners.delete(fn); };
}

function setBackendOnline(online: boolean) {
  if (_backendOnline !== online) {
    _backendOnline = online;
    _connectionListeners.forEach((fn) => fn(online));
  }
}

function getRetryDelay(attempt: number): number {
  const base = BASE_DELAY_MS * Math.pow(2, attempt - 1);
  return base * (0.75 + Math.random() * 0.5);
}

async function fetchWithRetry(path: string, fetchInit: RequestInit): Promise<Response> {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(buildApiUrl(path), fetchInit);
      setBackendOnline(true);
      return response;
    } catch (error) {
      if (attempt < MAX_RETRIES && error instanceof TypeError) {
        setBackendOnline(false);
        await new Promise((resolve) => setTimeout(resolve, getRetryDelay(attempt)));
        continue;
      }
      throw error;
    }
  }
  throw new Error("Request failed after retries");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetchWithRetry(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    const text = await response.text();
    throw new Error(`Unexpected response (${response.status}): ${text.slice(0, 140)}`);
  }

  const payload = (await response.json()) as Envelope<T>;

  if (!response.ok || payload.status === "error") {
    const message = payload?.message || "Request failed";
    throw new Error(message);
  }

  return payload.data;
}

async function requestForm<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetchWithRetry(path, { method: "POST", body: formData });

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    const text = await response.text();
    throw new Error(`Unexpected response (${response.status}): ${text.slice(0, 140)}`);
  }

  const payload = (await response.json()) as Envelope<T>;

  if (!response.ok || payload.status === "error") {
    const message = payload?.message || "Request failed";
    throw new Error(message);
  }

  return payload.data;
}

async function requestBlob(path: string): Promise<{ blob: Blob; filename: string | null }> {
  const response = await fetchWithRetry(path, {});
  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as Envelope<unknown>;
      throw new Error(payload?.message || "Request failed");
    }
    const text = await response.text();
    throw new Error(`Unexpected response (${response.status}): ${text.slice(0, 140)}`);
  }
  const blob = await response.blob();
  const filename = parseDispositionFilename(response.headers.get("content-disposition"));
  return { blob, filename };
}

export async function fetchCases(): Promise<CaseSummary[]> {
  return request<CaseSummary[]>("/api/cases");
}

export async function createCase(data: { name: string; description?: string }): Promise<CaseDetail> {
  return request<CaseDetail>("/api/cases", {
    method: "POST",
    body: JSON.stringify(data)
  });
}

export async function updateCase(
  caseId: string,
  data: { name?: string; description?: string | null; status?: CaseStatus }
): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}`, {
    method: "PATCH",
    body: JSON.stringify(data)
  });
}

export async function deleteCase(caseId: string): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/api/cases/${caseId}`, {
    method: "DELETE"
  });
}

export async function fetchCase(caseId: string): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}`);
}

export async function fetchCaseSummary(caseId: string): Promise<CaseWorkspaceSummary> {
  return request<CaseWorkspaceSummary>(`/api/cases/${caseId}/summary`);
}

export async function refreshCaseSummary(caseId: string): Promise<CaseWorkspaceSummary> {
  return request<CaseWorkspaceSummary>(`/api/cases/${caseId}/summary/refresh`, {
    method: "POST"
  });
}

export async function fetchGraph(
  caseId: string,
  params: {
    limit?: number;
    entity_types?: string[];
    relation_types?: string[];
    keyword_filters?: string;
    focus_entity?: string;
    max_hops?: number;
    min_confidence?: number;
    date_from?: string;
    date_to?: string;
    include_undated?: boolean;
  }
): Promise<GraphViewPayload> {
  const query = new URLSearchParams();
  if (params.limit) query.set("limit", String(params.limit));
  if (params.entity_types?.length) query.set("entity_types", params.entity_types.join(","));
  if (params.relation_types?.length) query.set("relation_types", params.relation_types.join(","));
  if (params.keyword_filters?.trim()) query.set("keyword_filters", params.keyword_filters.trim());
  if (params.focus_entity) query.set("focus_entity", params.focus_entity);
  if (params.max_hops !== undefined) query.set("max_hops", String(params.max_hops));
  if (params.min_confidence !== undefined) query.set("min_confidence", String(params.min_confidence));
  if (params.date_from) query.set("date_from", params.date_from);
  if (params.date_to) query.set("date_to", params.date_to);
  if (params.include_undated !== undefined) query.set("include_undated", String(params.include_undated));
  return request<GraphViewPayload>(`/api/cases/${caseId}/graph?${query.toString()}`);
}

export async function fetchChats(caseId: string): Promise<ChatSummary[]> {
  return request<ChatSummary[]>(`/api/cases/${caseId}/chats`);
}

export async function createChat(
  caseId: string,
  data?: { title?: string }
): Promise<ChatSummary> {
  return request<ChatSummary>(`/api/cases/${caseId}/chats`, {
    method: "POST",
    body: JSON.stringify(data ?? {})
  });
}

export async function fetchChat(caseId: string, chatId: string): Promise<ChatDetail> {
  return request<ChatDetail>(`/api/cases/${caseId}/chats/${chatId}`);
}

export async function deleteChat(
  caseId: string,
  chatId: string
): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/api/cases/${caseId}/chats/${chatId}`, {
    method: "DELETE"
  });
}

export async function sendChatMessage(
  caseId: string,
  chatId: string,
  data: { content: string; mode: ChatMode; options?: Record<string, unknown> }
): Promise<ChatMessageResponse> {
  return request<ChatMessageResponse>(`/api/cases/${caseId}/chats/${chatId}/messages`, {
    method: "POST",
    body: JSON.stringify(data)
  });
}

export async function fetchAnalyses(caseId: string): Promise<AnalysisRecord[]> {
  return request<AnalysisRecord[]>(`/api/cases/${caseId}/analyses`);
}

export async function createAnalysis(
  caseId: string,
  data: { prompt: string; analysis_type: AnalysisType }
): Promise<AnalysisRecord> {
  return request<AnalysisRecord>(`/api/cases/${caseId}/analyses`, {
    method: "POST",
    body: JSON.stringify(data)
  });
}

export async function fetchAnalysis(caseId: string, analysisId: string): Promise<AnalysisRecord> {
  return request<AnalysisRecord>(`/api/cases/${caseId}/analyses/${analysisId}`);
}

export async function deleteAnalysis(caseId: string, analysisId: string): Promise<void> {
  await request<{ deleted: boolean }>(`/api/cases/${caseId}/analyses/${analysisId}`, {
    method: "DELETE"
  });
}

export async function repairAnalysisChart(
  caseId: string,
  analysisId: string,
  data: { chart_id: string; error: string; mermaid_code?: string }
): Promise<AnalysisRecord> {
  return request<AnalysisRecord>(`/api/cases/${caseId}/analyses/${analysisId}/repair`, {
    method: "POST",
    body: JSON.stringify(data)
  });
}

export async function retryAnalysis(caseId: string, analysisId: string): Promise<AnalysisRecord> {
  return request<AnalysisRecord>(`/api/cases/${caseId}/analyses/${analysisId}/retry`, {
    method: "POST"
  });
}

export async function fetchDocuments(caseId: string): Promise<DocumentSummary[]> {
  return request<DocumentSummary[]>(`/api/cases/${caseId}/documents`);
}

export async function searchDocuments(
  caseId: string,
  params: { q: string; source?: DocumentSearchSource; limit?: number }
): Promise<DocumentSearchResult[]> {
  const query = new URLSearchParams();
  query.set("q", params.q.trim());
  if (params.source) query.set("source", params.source);
  if (params.limit) query.set("limit", String(params.limit));
  return request<DocumentSearchResult[]>(`/api/cases/${caseId}/documents/search?${query.toString()}`);
}

export async function fetchDocumentSearchPreview(
  caseId: string,
  documentId: string,
  params: { q: string; source_kind: "raw" | "processed"; segment_key: string }
): Promise<DocumentSearchPreview> {
  const query = new URLSearchParams();
  query.set("q", params.q.trim());
  query.set("source_kind", params.source_kind);
  query.set("segment_key", params.segment_key);
  return request<DocumentSearchPreview>(
    `/api/cases/${caseId}/documents/${documentId}/search-preview?${query.toString()}`
  );
}

export async function fetchDocumentReferencePreview(
  caseId: string,
  documentId: string,
  params: { reference_id: string; q?: string; snippet?: string }
): Promise<DocumentSearchPreview> {
  const query = new URLSearchParams();
  query.set("reference_id", params.reference_id.trim());
  if (params.q?.trim()) {
    query.set("q", params.q.trim());
  }
  if (params.snippet?.trim()) {
    query.set("snippet", params.snippet.trim());
  }
  return request<DocumentSearchPreview>(
    `/api/cases/${caseId}/documents/${documentId}/reference-preview?${query.toString()}`
  );
}

export async function checkDocumentDuplicates(
  caseId: string,
  payload: { files: DocumentDuplicateCheckItem[] }
): Promise<DocumentDuplicateCheckResult[]> {
  return request<DocumentDuplicateCheckResult[]>(`/api/cases/${caseId}/documents/duplicates/check`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function uploadDocument(
  caseId: string,
  payload: {
    file: File;
    confidence_source_reliability: string;
    confidence_information_validity: string;
    ingest_profile?: IngestionProfile;
    processing_mode?: ProcessingMode;
    advanced_overrides?: IngestionAdvancedOverrides;
    content_hash_sha256?: string;
    allow_duplicate?: boolean;
    tags?: string;
    notes?: string;
  }
): Promise<{
  document_id: string;
  job_id: string;
  content_hash_sha256?: string;
  ingest_profile?: string;
  processing_mode?: ProcessingMode;
  preflight?: IngestionPreflight;
  advanced_overrides?: IngestionAdvancedOverrides | null;
}> {
  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("confidence_source_reliability", payload.confidence_source_reliability);
  formData.append("confidence_information_validity", payload.confidence_information_validity);
  if (payload.ingest_profile) {
    formData.append("ingest_profile", payload.ingest_profile);
  }
  if (payload.processing_mode) {
    formData.append("processing_mode", payload.processing_mode);
  }
  if (payload.advanced_overrides && Object.keys(payload.advanced_overrides).length > 0) {
    formData.append("advanced_overrides", JSON.stringify(payload.advanced_overrides));
  }
  if (payload.content_hash_sha256) {
    formData.append("content_hash_sha256", payload.content_hash_sha256);
  }
  if (payload.allow_duplicate !== undefined) {
    formData.append("allow_duplicate", String(payload.allow_duplicate));
  }
  if (payload.tags) {
    formData.append("tags", payload.tags);
  }
  if (payload.notes) {
    formData.append("notes", payload.notes);
  }
  return requestForm<{
    document_id: string;
    job_id: string;
    content_hash_sha256?: string;
    ingest_profile?: string;
    processing_mode?: ProcessingMode;
    preflight?: IngestionPreflight;
    advanced_overrides?: IngestionAdvancedOverrides | null;
  }>(
    `/api/cases/${caseId}/documents`,
    formData
  );
}

export async function fetchDocument(caseId: string, documentId: string): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/api/cases/${caseId}/documents/${documentId}`);
}

export async function fetchJobs(caseId: string): Promise<IngestionJob[]> {
  return request<IngestionJob[]>(`/api/cases/${caseId}/jobs`);
}

export async function fetchJobLogs(
  caseId: string,
  jobId: string,
  options?: { after_id?: number; limit?: number }
): Promise<IngestionJobLog[]> {
  const query = new URLSearchParams();
  if (typeof options?.after_id === "number" && options.after_id > 0) {
    query.set("after_id", String(Math.floor(options.after_id)));
  }
  if (typeof options?.limit === "number" && options.limit > 0) {
    query.set("limit", String(Math.floor(options.limit)));
  }
  const suffix = query.toString();
  return request<IngestionJobLog[]>(
    `/api/cases/${caseId}/jobs/${jobId}/logs${suffix ? `?${suffix}` : ""}`
  );
}

// Actor-centric API wrappers

export async function fetchStats(caseId: string): Promise<StatsSummary> {
  return request<StatsSummary>(`/api/cases/${caseId}/stats`);
}

export async function fetchTagClusters(caseId: string): Promise<TagCluster[]> {
  return request<TagCluster[]>(`/api/cases/${caseId}/tag-clusters`);
}

export async function fetchRelationships(
  caseId: string,
  params: {
    limit?: number;
    clusters?: string[];
    categories?: string[];
    yearMin?: number;
    yearMax?: number;
    includeUndated?: boolean;
    keywords?: string;
    maxHops?: number | null;
  }
): Promise<RelationshipsResponse> {
  const query = new URLSearchParams();
  query.set("limit", String(params.limit ?? 500));
  if (params.clusters?.length) query.set("clusters", params.clusters.join(","));
  if (params.categories?.length) query.set("categories", params.categories.join(","));
  if (typeof params.yearMin === "number") query.set("yearMin", String(params.yearMin));
  if (typeof params.yearMax === "number") query.set("yearMax", String(params.yearMax));
  if (typeof params.includeUndated === "boolean") query.set("includeUndated", String(params.includeUndated));
  if (params.keywords?.trim()) query.set("keywords", params.keywords.trim());
  if (params.maxHops !== undefined && params.maxHops !== null) query.set("maxHops", String(params.maxHops));
  return request<RelationshipsResponse>(`/api/cases/${caseId}/relationships?${query.toString()}`);
}

export async function fetchActorRelationships(
  caseId: string,
  actorName: string,
  params: {
    clusters?: string[];
    categories?: string[];
    yearMin?: number;
    yearMax?: number;
    includeUndated?: boolean;
    keywords?: string;
    maxHops?: number | null;
  }
): Promise<RelationshipsResponse> {
  const query = new URLSearchParams();
  if (params.clusters?.length) query.set("clusters", params.clusters.join(","));
  if (params.categories?.length) query.set("categories", params.categories.join(","));
  if (typeof params.yearMin === "number") query.set("yearMin", String(params.yearMin));
  if (typeof params.yearMax === "number") query.set("yearMax", String(params.yearMax));
  if (typeof params.includeUndated === "boolean") query.set("includeUndated", String(params.includeUndated));
  if (params.keywords?.trim()) query.set("keywords", params.keywords.trim());
  if (params.maxHops !== undefined && params.maxHops !== null) query.set("maxHops", String(params.maxHops));
  return request<RelationshipsResponse>(
    `/api/cases/${caseId}/actor/${encodeURIComponent(actorName)}/relationships${query.toString() ? `?${query.toString()}` : ""}`
  );
}

export async function searchActors(caseId: string, query: string): Promise<ActorSearchResult[]> {
  const q = query.trim();
  if (!q) return [];
  const search = new URLSearchParams({ q });
  return request<ActorSearchResult[]>(`/api/cases/${caseId}/search?${search.toString()}`);
}

export async function fetchActorCounts(caseId: string, limit = 300): Promise<Record<string, number>> {
  const search = new URLSearchParams({ limit: String(limit) });
  return request<Record<string, number>>(`/api/cases/${caseId}/actor-counts?${search.toString()}`);
}

export async function fetchActorCount(caseId: string, name: string): Promise<number> {
  const data = await request<{ count: number }>(`/api/cases/${caseId}/actor/${encodeURIComponent(name)}/count`);
  return data.count;
}

export async function fetchGraphEntity(caseId: string, entityId: string): Promise<GraphEntityDetails> {
  return request<GraphEntityDetails>(
    `/api/cases/${caseId}/graph/entity/${encodeURIComponent(entityId)}`
  );
}

export async function fetchGraphRelationship(
  caseId: string,
  params: { src_id: string; tgt_id: string }
): Promise<GraphRelationshipDetails> {
  const query = new URLSearchParams({
    src_id: params.src_id,
    tgt_id: params.tgt_id
  });
  return request<GraphRelationshipDetails>(`/api/cases/${caseId}/graph/relationship?${query.toString()}`);
}

export async function deleteDocument(caseId: string, documentId: string): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/api/cases/${caseId}/documents/${documentId}`, {
    method: "DELETE"
  });
}

export function buildDocumentDownloadUrl(caseId: string, documentId: string): string {
  return buildApiUrl(`/api/cases/${caseId}/documents/${documentId}/download`);
}

export async function downloadDocument(caseId: string, documentId: string): Promise<void> {
  const { blob, filename } = await requestBlob(
    `/api/cases/${caseId}/documents/${documentId}/download`
  );
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || "document";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

async function downloadBlobPath(path: string, fallbackFilename: string): Promise<void> {
  const { blob, filename } = await requestBlob(path);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || fallbackFilename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export async function downloadGraphEntitiesCsv(caseId: string): Promise<void> {
  await downloadBlobPath(
    `/api/cases/${caseId}/graph/export/entities.csv`,
    "entities.csv"
  );
}

export async function downloadGraphRelationsCsv(caseId: string): Promise<void> {
  await downloadBlobPath(
    `/api/cases/${caseId}/graph/export/relations.csv`,
    "relations.csv"
  );
}

export interface SettingsPayload {
  overrides: Record<string, unknown>;
  effective: Record<string, unknown>;
  mutable_fields: string[];
}

export async function fetchSettings(): Promise<SettingsPayload> {
  return request<SettingsPayload>("/api/settings");
}

export async function updateSettings(overrides: Record<string, unknown>): Promise<SettingsPayload> {
  return request<SettingsPayload>("/api/settings", {
    method: "PATCH",
    body: JSON.stringify(overrides)
  });
}

export async function updateDocument(
  caseId: string,
  documentId: string,
  data: {
    notes?: string | null;
    confidence_source_reliability?: string;
    confidence_information_validity?: string;
  }
): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/api/cases/${caseId}/documents/${documentId}`, {
    method: "PATCH",
    body: JSON.stringify(data)
  });
}

export async function updateDocumentNotes(
  caseId: string,
  documentId: string,
  notes: string | null
): Promise<DocumentSummary> {
  return updateDocument(caseId, documentId, { notes });
}

export async function reingestDocument(
  caseId: string,
  documentId: string,
  ingestProfile?: IngestionProfile,
  processingMode?: ProcessingMode,
  advancedOverrides?: IngestionAdvancedOverrides,
  notes?: string | null,
  confidenceSourceReliability?: string,
  confidenceInfoValidity?: string
): Promise<{
  job_id: string;
  ingest_profile?: string;
  processing_mode?: ProcessingMode;
  preflight?: IngestionPreflight;
  advanced_overrides?: IngestionAdvancedOverrides | null;
}> {
  const query = new URLSearchParams();
  if (ingestProfile) {
    query.set("ingest_profile", ingestProfile);
  }
  if (processingMode) {
    query.set("processing_mode", processingMode);
  }
  if (advancedOverrides && Object.keys(advancedOverrides).length > 0) {
    query.set("advanced_overrides", JSON.stringify(advancedOverrides));
  }
  if (notes !== undefined && notes !== null) {
    query.set("notes", notes);
  }
  if (confidenceSourceReliability) {
    query.set("confidence_source_reliability", confidenceSourceReliability);
  }
  if (confidenceInfoValidity) {
    query.set("confidence_information_validity", confidenceInfoValidity);
  }
  const suffix = query.toString();
  return request<{
    job_id: string;
    ingest_profile?: string;
    processing_mode?: ProcessingMode;
    preflight?: IngestionPreflight;
    advanced_overrides?: IngestionAdvancedOverrides | null;
  }>(`/api/cases/${caseId}/documents/${documentId}/reingest${suffix ? `?${suffix}` : ""}`, {
    method: "POST"
  });
}
