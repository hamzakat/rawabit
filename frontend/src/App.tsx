import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import {
  AlertTriangle,
  Archive,
  ArchiveRestore,
  ArrowLeft,
  Download,
  Eye,
  FileText,
  FolderOpen,
  ListChecks,
  Loader2,
  MoreHorizontal,
  Network,
  Pencil,
  Plus,
  RefreshCcw,
  Search,
  Settings2,
  Share2,
  SlidersHorizontal,
  Trash2,
  Upload,
  Info
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { MarkdownText } from "@/components/ui/markdown";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AnalysisWorkspace } from "@/components/analysis/analysis-workspace";
import { SettingsDialog } from "@/components/settings-dialog";
import { GraphCanvas, type GraphLegendSelection } from "@/components/graph/graph-canvas";
import type {
  ChatMessage,
  ChatMode,
  ChatSummary,
  DocumentDuplicateCheckResult,
  DocumentSearchPreview,
  DocumentSearchResult,
  DocumentSearchSource,
  CaseDetail,
  CaseStatus,
  CaseSummary,
  CaseWorkspaceSummary,
  DocumentSummary,
  GraphEdge,
  GraphEntityDetails,
  GraphNode,
  GraphRelationshipDetails,
  GraphViewPayload,
  GraphEvidence,
  HighlightPayload,
  IngestionAdvancedOverrides,
  IngestionJob,
  IngestionJobLog,
  IngestionPreflight,
  IngestionProfile,
  ProcessingMode,
  RelationshipRow,
  RelationshipsResponse,
  TagCluster,
  ActorSearchResult
} from "@/lib/api";
import {
  createChat,
  checkDocumentDuplicates,
  deleteChat,
  fetchTagClusters,
  fetchActorRelationships,
  fetchChat,
  fetchChats,
  searchActors as apiSearchActors,
  createCase,
  deleteCase,
  deleteDocument,
  downloadDocument,
  downloadGraphEntitiesCsv,
  downloadGraphRelationsCsv,
  buildDocumentDownloadUrl,
  fetchCase,
  fetchCaseSummary,
  fetchCases,
  fetchDocumentReferencePreview,
  fetchDocumentSearchPreview,
  fetchGraph,
  fetchGraphEntity,
  fetchGraphRelationship,
  fetchJobLogs,
  fetchJobs,
  fetchDocuments,
  refreshCaseSummary,
  reingestDocument,
  searchDocuments as apiSearchDocuments,
  sendChatMessage,
  updateCase,
  updateDocumentNotes,
  updateDocument,
  uploadDocument,
  isBackendOnline,
  onConnectionChange
} from "@/lib/api";

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "2-digit"
});

const dateTimeFormatter = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit"
});

const filterOptions = [
  { label: "Active", value: "active" },
  { label: "Archived", value: "archived" },
  { label: "All", value: "all" }
] as const;

const confidenceRows = ["A", "B", "C", "X"] as const;
const confidenceColumns = ["1", "2", "3", "4"] as const;
const confidenceSourceDescriptions: Record<(typeof confidenceRows)[number], string> = {
  A: "Reliable source",
  B: "Usually reliable",
  C: "Reliability uncertain",
  X: "Unreliable source"
};
const confidenceSourceCompact: Record<(typeof confidenceRows)[number], string> = {
  A: "Reliable",
  B: "Usually reliable",
  C: "Uncertain",
  X: "Unreliable"
};

const confidenceValidityDescriptions: Record<(typeof confidenceColumns)[number], string> = {
  "1": "Confirmed",
  "2": "Probably true",
  "3": "Possibly true",
  "4": "Doubtful"
};
const confidenceValidityCompact: Record<(typeof confidenceColumns)[number], string> = {
  "1": "Confirmed",
  "2": "Probable",
  "3": "Possible",
  "4": "Doubtful"
};

const confidenceSourceScore: Record<(typeof confidenceRows)[number], number> = {
  A: 4,
  B: 3,
  C: 2,
  X: 1
};

const confidenceValidityScore: Record<(typeof confidenceColumns)[number], number> = {
  "1": 4,
  "2": 3,
  "3": 2,
  "4": 1
};

type FilterValue = (typeof filterOptions)[number]["value"];

type ConfidenceSelection = {
  source: (typeof confidenceRows)[number];
  validity: (typeof confidenceColumns)[number];
} | null;

type WorkspaceSelectedEdge = {
  id: string;
  src_id: string;
  tgt_id: string;
};

type LegendSelection = GraphLegendSelection;

type UploadDuplicateChoice = "upload_new" | null;

type UploadQueueItem = {
  clientId: string;
  file: File;
  contentHashSha256?: string | null;
  hashReady: boolean;
  hashError?: string | null;
  selectedConfidence: ConfidenceSelection;
  ingestProfile: IngestionProfile;
  processingMode: ProcessingMode;
  advancedOpen: boolean;
  advanced: IngestionAdvancedOverrides;
  estimatedPreflight: IngestionPreflight | null;
  duplicateMatches: DocumentSummary[];
  duplicateChoice: UploadDuplicateChoice;
  uploadState: "idle" | "uploading" | "uploaded" | "failed";
  uploadError?: string | null;
  notes: string;
};

type EvidencePreviewTarget = {
  document_id: string;
  original_filename: string;
  confidence_code?: string | null;
  source_kind: "raw" | "processed";
  segment_key: string;
  stored_file_path?: string;
  snippet?: string | null;
};

const DEFAULT_GRAPH_LIMIT = 500;
const GRAPH_LIMIT_INCREMENT = 250;
const MAX_AUTOLOAD_LIMIT = 2000;
const MAX_HIGHLIGHT_AUTOLOAD_ATTEMPTS = 3;
const CHAT_COMPOSER_MIN_HEIGHT_PX = 80;
const CHAT_COMPOSER_MAX_HEIGHT_PX = 144;
const CHAT_THREAD_BOTTOM_THRESHOLD_PX = 48;

const ingestProfileOptions: Array<{ value: IngestionProfile; label: string; hint: string }> = [
  {
    value: "balanced_fast_intel",
    label: "Standard",
    hint: "Default profile for mixed evidence. Balanced speed and extraction depth."
  },
  {
    value: "full_enrichment",
    label: "Deep",
    hint: "Slower, deeper extraction for high-value or difficult evidence."
  }
];

const chatModeOptions: Array<{ value: ChatMode; label: string }> = [
  { value: "mix", label: "Mix" },
  { value: "local", label: "Local" },
  { value: "global", label: "Global" },
  { value: "hybrid", label: "Hybrid" },
  { value: "naive", label: "Naive" },
  { value: "bypass", label: "Bypass" }
];

const emptyHighlightPayload: HighlightPayload = {
  highlight_entities: [],
  highlight_relationships: [],
  references: []
};

function normalizeHighlightPayload(payload: Partial<HighlightPayload> | null | undefined): HighlightPayload {
  return {
    highlight_entities: Array.isArray(payload?.highlight_entities)
      ? payload?.highlight_entities.filter((value): value is string => typeof value === "string")
      : [],
    highlight_relationships: Array.isArray(payload?.highlight_relationships)
      ? payload.highlight_relationships.filter(
          (entry): entry is HighlightPayload["highlight_relationships"][number] =>
            Boolean(entry?.src_id) && Boolean(entry?.tgt_id)
        )
      : [],
    supporting_chunks: Array.isArray(payload?.supporting_chunks)
      ? payload.supporting_chunks.filter(
          (chunk): chunk is NonNullable<HighlightPayload["supporting_chunks"]>[number] =>
            Boolean(chunk?.reference_id) && Boolean(chunk?.file_path)
        )
      : undefined,
    references: Array.isArray(payload?.references)
      ? payload.references.filter(
          (reference): reference is HighlightPayload["references"][number] =>
            Boolean(reference?.reference_id) && Boolean(reference?.file_path)
        )
      : []
  };
}

function pathBasename(filePath: string) {
  const normalized = filePath.replace(/\\/g, "/").trim();
  if (!normalized) {
    return "Unknown file";
  }
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function pairKey(srcId: string, tgtId: string) {
  return `${srcId}=>${tgtId}`;
}

function HighlightedSnippet({
  parts,
  fallback
}: {
  parts?: Array<{ text: string; match: boolean }>;
  fallback: string;
}) {
  const safeParts = parts?.length ? parts : [{ text: fallback, match: false }];
  return (
    <>
      {safeParts.map((part, index) =>
        part.match ? (
          <mark
            key={`${part.text}-${index}`}
            className="rounded-sm bg-amber-200/80 px-0.5 text-foreground"
          >
            {part.text}
          </mark>
        ) : (
          <span key={`${part.text}-${index}`}>{part.text}</span>
        )
      )}
    </>
  );
}

function HighlightedPreviewText({
  content,
  ranges
}: {
  content: string;
  ranges: Array<{ start: number; end: number }>;
}) {
  if (!ranges.length) {
    return <>{content}</>;
  }
  const nodes: JSX.Element[] = [];
  let cursor = 0;
  ranges.forEach((range, index) => {
    const start = Math.max(0, Math.min(range.start, content.length));
    const end = Math.max(start, Math.min(range.end, content.length));
    if (start > cursor) {
      nodes.push(<span key={`text-${index}`}>{content.slice(cursor, start)}</span>);
    }
    if (end > start) {
      nodes.push(
        <mark
          key={`match-${index}`}
          className="rounded-sm bg-amber-200/85 px-0.5 text-foreground"
        >
          {content.slice(start, end)}
        </mark>
      );
    }
    cursor = end;
  });
  if (cursor < content.length) {
    nodes.push(<span key="text-tail">{content.slice(cursor)}</span>);
  }
  return <>{nodes}</>;
}

function asObjectRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function extractAssistantMetadata(message: ChatMessage): Record<string, unknown> | null {
  if (message.role !== "assistant") {
    return null;
  }
  return asObjectRecord(message.rag_metadata);
}

function hasHighlightPayloadData(payload: HighlightPayload | null | undefined) {
  return Boolean(
    payload &&
      (payload.highlight_entities.length > 0 ||
        payload.highlight_relationships.length > 0 ||
        payload.references.length > 0 ||
        (payload.supporting_chunks?.length ?? 0) > 0)
  );
}

function extractLegacyAssistantHighlight(message: ChatMessage): HighlightPayload | null {
  const metadata = extractAssistantMetadata(message);
  if (!metadata) {
    return null;
  }
  const highlightSource = asObjectRecord(metadata.highlight);
  const highlightReferences = Array.isArray(highlightSource?.references)
    ? highlightSource.references
    : undefined;
  const metadataReferences = Array.isArray(metadata.references) ? metadata.references : undefined;
  const highlightChunks = Array.isArray(highlightSource?.supporting_chunks)
    ? highlightSource.supporting_chunks
    : undefined;
  const metadataChunks = Array.isArray(metadata.chunks) ? metadata.chunks : undefined;
  const normalized = normalizeHighlightPayload({
    ...(highlightSource ?? {}),
    references: highlightReferences ?? metadataReferences,
    supporting_chunks: highlightChunks ?? metadataChunks
  });
  return hasHighlightPayloadData(normalized) ? normalized : null;
}

function extractAssistantHighlight(message: ChatMessage): HighlightPayload | null {
  return extractLegacyAssistantHighlight(message);
}

function extractAssistantModelName(message: ChatMessage): string | null {
  const metadata = extractAssistantMetadata(message);
  const rawValue =
    metadata && typeof metadata.model_name === "string" ? metadata.model_name.trim() : "";
  return rawValue || null;
}

function dedupeEvidence(items: GraphEvidence[]): GraphEvidence[] {
  const seen = new Set<string>();
  const output: GraphEvidence[] = [];
  items.forEach((item) => {
    const key = [
      item.file_path ?? "",
      item.reference_id ?? "",
      item.document_id ?? "",
      item.source_id ?? "",
      item.confidence_code ?? ""
    ].join("|");
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    output.push(item);
  });
  return output;
}

function normalizeEntityType(value: string | null | undefined): string {
  const normalized = (value ?? "").trim().toLowerCase();
  if (!normalized) {
    return "other";
  }
  return normalized;
}

function formatDate(value?: string) {
  if (!value) {
    return "Unknown";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return dateFormatter.format(parsed);
}

function formatDateTime(value?: string) {
  if (!value) {
    return "Unknown";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return dateTimeFormatter.format(parsed);
}

function formatBytes(sizeBytes: number) {
  if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = sizeBytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function formatModelLabel(modelName?: string | null) {
  const normalized = modelName?.trim();
  if (!normalized) {
    return "Pending";
  }
  return normalized;
}

function formatDuration(parseS?: number | null, insertS?: number | null, finalizeS?: number | null) {
  const total = (parseS ?? 0) + (insertS ?? 0) + (finalizeS ?? 0);
  if (total <= 0) {
    return "-";
  }
  const minutes = Math.floor(total / 60);
  const seconds = Math.round(total % 60);
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function buildUploadRowId(file: File, index: number) {
  const randomPart =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${randomPart}-${index}-${file.name}`;
}

async function hashFileSha256(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function sortChatsByUpdated(chats: ChatSummary[]): ChatSummary[] {
  return [...chats].sort((left, right) => {
    const updatedDiff = right.updated_at.localeCompare(left.updated_at);
    if (updatedDiff !== 0) {
      return updatedDiff;
    }
    return right.created_at.localeCompare(left.created_at);
  });
}

function chatMessageKey(message?: ChatMessage) {
  if (!message) {
    return null;
  }
  return `${message.id}:${message.role}:${message.created_at}`;
}

function areChatMessagesEqual(left: ChatMessage[], right: ChatMessage[]) {
  if (left === right) {
    return true;
  }
  if (left.length !== right.length) {
    return false;
  }
  return left.every((message, index) => {
    const other = right[index];
    return (
      message.id === other.id &&
      message.chat_id === other.chat_id &&
      message.role === other.role &&
      message.content === other.content &&
      message.created_at === other.created_at &&
      JSON.stringify(message.rag_metadata ?? null) === JSON.stringify(other.rag_metadata ?? null)
    );
  });
}

function isChatThreadNearBottom(element: HTMLDivElement) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= CHAT_THREAD_BOTTOM_THRESHOLD_PX;
}

function sortDocumentsByCreatedAt(documents: DocumentSummary[]) {
  return [...documents].sort((left, right) => {
    const leftTime = Date.parse(left.created_at);
    const rightTime = Date.parse(right.created_at);
    if (Number.isFinite(leftTime) && Number.isFinite(rightTime)) {
      return rightTime - leftTime;
    }
    return right.created_at.localeCompare(left.created_at);
  });
}

function profileLabel(profile?: string | null) {
  if (!profile) {
    return "unknown";
  }
  if (profile === "full_enrichment") {
    return "Deep";
  }
  if (profile === "balanced_fast_intel") {
    return "Standard";
  }
  if (profile === "balanced_fast") {
    return "Standard";
  }
  return profile;
}

function confidenceCellToneClass(
  source: (typeof confidenceRows)[number],
  validity: (typeof confidenceColumns)[number],
  selected: boolean
) {
  if (selected) {
    return "border-primary bg-primary text-primary-foreground shadow-sm";
  }
  const score = confidenceSourceScore[source] + confidenceValidityScore[validity];
  if (score >= 7) {
    return "border-primary/40 bg-primary/15 text-foreground hover:bg-primary/20";
  }
  if (score >= 6) {
    return "border-primary/35 bg-primary/12 text-foreground hover:bg-primary/16";
  }
  if (score >= 5) {
    return "border-primary/30 bg-primary/9 text-foreground hover:bg-primary/13";
  }
  if (score >= 4) {
    return "border-primary/25 bg-primary/7 text-foreground hover:bg-primary/11";
  }
  return "border-primary/20 bg-primary/5 text-foreground hover:bg-primary/9";
}

function estimateUploadComplexity(file: File | null): IngestionPreflight | null {
  if (!file) {
    return null;
  }
  const sizeMb = file.size / (1024 * 1024);
  const extension = (() => {
    const idx = file.name.lastIndexOf(".");
    if (idx < 0) {
      return "";
    }
    return file.name.slice(idx).toLowerCase();
  })();
  let complexity: IngestionPreflight["complexity_class"] = "small";
  if (sizeMb > 45) {
    complexity = "very_large";
  } else if (sizeMb > 15) {
    complexity = "large";
  } else if (sizeMb > 2) {
    complexity = "medium";
  }
  const etaMap: Record<IngestionPreflight["complexity_class"], number> = {
    small: 90,
    medium: 210,
    large: 480,
    very_large: 960
  };
  const warnings =
    complexity === "large" || complexity === "very_large"
      ? ["Expected ingestion time is high for this file size."]
      : [];
  return {
    source_kind: "local_estimate",
    mime_type: file.type || "application/octet-stream",
    extension,
    complexity_class: complexity,
    eta_seconds: etaMap[complexity],
    metrics: {
      size_bytes: file.size,
      size_mb: Math.round(sizeMb * 1000) / 1000
    },
    warnings
  };
}

function statusBadge(status: DocumentSummary["ingestion_status"] | IngestionJob["status"]) {
  const label = status.replace(/_/g, " ");
  if (status === "complete") {
    return <Badge className="bg-emerald-600 text-white">{label}</Badge>;
  }
  if (status === "completed_with_warnings") {
    return <Badge className="bg-amber-600 text-white">{label}</Badge>;
  }
  if (status === "failed") {
    return <Badge className="bg-destructive text-destructive-foreground">{label}</Badge>;
  }
  return <Badge variant="muted">{label}</Badge>;
}

function isProcessingStatus(status: DocumentSummary["ingestion_status"] | IngestionJob["status"]) {
  return status === "queued" || status === "parsing" || status === "inserting" || status === "indexing";
}

function ProcessingIndicator({ label, className }: { label: string; className?: string }) {
  return (
    <span
      className={className ?? "inline-flex items-center text-primary"}
      aria-label={label}
      title={label}
    >
      <Loader2 className="h-4 w-4 animate-spin" />
    </span>
  );
}

function RenameCaseDialog({
  open,
  busy,
  initialName,
  initialDescription,
  onOpenChange,
  onSubmit
}: {
  open: boolean;
  busy: boolean;
  initialName?: string | null;
  initialDescription?: string | null;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: { name: string; description?: string }) => void | Promise<void>;
}) {
  const [name, setName] = useState(initialName ?? "");
  const [description, setDescription] = useState(initialDescription ?? "");

  useEffect(() => {
    if (!open) {
      return;
    }
    setName(initialName ?? "");
    setDescription(initialDescription ?? "");
  }, [initialDescription, initialName, open]);

  const handleSubmit = () => {
    const trimmedName = name.trim();
    if (!trimmedName) {
      return;
    }
    void onSubmit({
      name: trimmedName,
      description: description.trim() || undefined
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename case</DialogTitle>
          <DialogDescription>Update the case name and optional description without changing the case slug.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-2">
            <label className="text-sm font-medium" htmlFor="rename-case-name">
              Case name
            </label>
            <Input
              id="rename-case-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Case name"
              disabled={busy}
            />
          </div>
          <div className="grid gap-2">
            <label className="text-sm font-medium" htmlFor="rename-case-description">
              Description (optional)
            </label>
            <Input
              id="rename-case-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Short analyst-facing description"
              disabled={busy}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={busy || !name.trim()}>
            {busy ? "Saving..." : "Save changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CaseJobsView({ caseId }: { caseId: string }) {
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [logsOpen, setLogsOpen] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobLogs, setJobLogs] = useState<IngestionJobLog[]>([]);
  const [jobLogsLoading, setJobLogsLoading] = useState(false);
  const [jobLogsError, setJobLogsError] = useState<string | null>(null);
  const lastLogIdRef = useRef(0);
  const logsPreRef = useRef<HTMLPreElement | null>(null);

  const docById = useMemo(() => {
    const map = new Map<string, DocumentSummary>();
    documents.forEach((doc) => map.set(doc.id, doc));
    return map;
  }, [documents]);

  const selectedJob = useMemo(() => {
    if (!selectedJobId) {
      return null;
    }
    return jobs.find((job) => job.id === selectedJobId) ?? null;
  }, [jobs, selectedJobId]);

  const visibleJobs = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return [...jobs].sort((a, b) => (b.started_at ?? "").localeCompare(a.started_at ?? "")).filter((job) => {
      if (!normalizedSearch) {
        return true;
      }
      const doc = docById.get(job.document_id);
      return (
        job.document_id.toLowerCase().includes(normalizedSearch) ||
        job.status.toLowerCase().includes(normalizedSearch) ||
        (doc?.original_filename ?? "").toLowerCase().includes(normalizedSearch)
      );
    });
  }, [jobs, docById, search]);

  const refresh = async () => {
    try {
      setLoading(true);
      setError(null);
      const [caseInfo, docs, jobData] = await Promise.all([
        fetchCase(caseId),
        fetchDocuments(caseId),
        fetchJobs(caseId)
      ]);
      setCaseDetail(caseInfo);
      setDocuments(docs);
      setJobs(jobData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ingestion jobs.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, [caseId]);

  useEffect(() => {
    let active = true;
    let inFlight = false;
    const timer = window.setInterval(() => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      void Promise.all([fetchCase(caseId), fetchDocuments(caseId), fetchJobs(caseId)])
        .then(([caseInfo, docs, jobData]) => {
          if (!active) {
            return;
          }
          setCaseDetail(caseInfo);
          setDocuments(docs);
          setJobs(jobData);
        })
        .catch((err) =>
          active
            ? setError(err instanceof Error ? err.message : "Unable to refresh ingestion jobs.")
            : undefined
        )
        .finally(() => {
          inFlight = false;
        });
    }, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [caseId]);

  useEffect(() => {
    if (!logsOpen || !selectedJobId) {
      return;
    }
    let cancelled = false;
    const loadInitial = async () => {
      try {
        setJobLogsLoading(true);
        setJobLogsError(null);
        const data = await fetchJobLogs(caseId, selectedJobId, { limit: 1000 });
        if (cancelled) {
          return;
        }
        setJobLogs(data);
        lastLogIdRef.current = data.length > 0 ? data[data.length - 1].id : 0;
      } catch (err) {
        if (!cancelled) {
          setJobLogsError(err instanceof Error ? err.message : "Unable to load job logs.");
        }
      } finally {
        if (!cancelled) {
          setJobLogsLoading(false);
        }
      }
    };
    void loadInitial();
    return () => {
      cancelled = true;
    };
  }, [caseId, logsOpen, selectedJobId]);

  useEffect(() => {
    if (!logsOpen || !selectedJobId) {
      return;
    }
    const timer = window.setInterval(() => {
      const afterId = lastLogIdRef.current;
      void fetchJobLogs(caseId, selectedJobId, { after_id: afterId, limit: 500 })
        .then((rows) => {
          if (rows.length === 0) {
            return;
          }
          lastLogIdRef.current = rows[rows.length - 1].id;
          setJobLogs((prev) => [...prev, ...rows]);
        })
        .catch((err) =>
          setJobLogsError(err instanceof Error ? err.message : "Unable to refresh job logs.")
        );
    }, 1500);
    return () => window.clearInterval(timer);
  }, [caseId, logsOpen, selectedJobId]);

  useEffect(() => {
    if (!logsPreRef.current) {
      return;
    }
    logsPreRef.current.scrollTop = logsPreRef.current.scrollHeight;
  }, [jobLogs]);

  const closeLogsDialog = () => {
    setLogsOpen(false);
    setSelectedJobId(null);
    setJobLogs([]);
    setJobLogsError(null);
    setJobLogsLoading(false);
    lastLogIdRef.current = 0;
  };

  const openLogsDialog = (jobId: string) => {
    setSelectedJobId(jobId);
    setLogsOpen(true);
  };

  const refreshLogs = async () => {
    if (!selectedJobId) {
      return;
    }
    try {
      setJobLogsLoading(true);
      setJobLogsError(null);
      const data = await fetchJobLogs(caseId, selectedJobId, { limit: 1000 });
      setJobLogs(data);
      lastLogIdRef.current = data.length > 0 ? data[data.length - 1].id : 0;
    } catch (err) {
      setJobLogsError(err instanceof Error ? err.message : "Unable to load job logs.");
    } finally {
      setJobLogsLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-3 py-4 sm:px-4 lg:px-5">
      <div className="mx-auto flex w-full max-w-[96rem] flex-col gap-4">
        <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border bg-card/90 px-4 py-3 shadow-glow backdrop-blur">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                window.location.href = `/cases/${caseId}`;
              }}
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Case
            </Button>
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Jobs</p>
              <h1 className="mt-1 text-2xl font-semibold">{caseDetail?.name ?? "Loading case"}</h1>
              <p className="text-sm text-muted-foreground">
                Track parsing, inserting, and indexing in real-time.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={refresh}>
              <RefreshCcw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </header>

        {error ? (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        ) : null}

        <section className="rounded-2xl border bg-card/90 p-6 shadow-soft">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex w-full flex-wrap items-center gap-3 md:w-auto">
              <div className="relative w-full max-w-sm">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  className="pl-9"
                  placeholder="Search by file name, status, or document id"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </div>
            </div>
            <Badge variant="muted">{visibleJobs.length} jobs</Badge>
          </div>

          {loading ? (
            <div className="mt-6 rounded-xl border border-dashed border-muted/60 p-10 text-center text-sm text-muted-foreground">
              Loading ingestion jobs...
            </div>
          ) : visibleJobs.length === 0 ? (
            <div className="mt-6 rounded-xl border border-dashed border-muted/60 p-10 text-center text-sm text-muted-foreground">
              No ingestion jobs yet. Upload evidence to start parsing.
            </div>
          ) : (
            <div className="mt-6">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Progress / Stage</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Started</TableHead>
                    <TableHead>Finished</TableHead>
                    <TableHead>Complexity / ETA</TableHead>
                    <TableHead>Error</TableHead>
                    <TableHead className="text-right">Logs</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visibleJobs.map((job) => {
                    const doc = docById.get(job.document_id);
                    return (
                      <TableRow key={job.id}>
                        <TableCell>
                          <div className="flex flex-col">
                            <span className="text-sm font-semibold text-foreground">
                              {doc?.original_filename ?? "Unknown document"}
                            </span>
                            <span className="text-xs font-mono text-muted-foreground">{job.document_id}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          {isProcessingStatus(job.status) && job.current_stage
                            ? <Badge variant="muted">{job.current_stage.replace(/_/g, " ")}</Badge>
                            : statusBadge(job.status)}
                        </TableCell>
                        <TableCell>
                          <div className="grid gap-1">
                            {typeof job.progress === "number" ? (
                              <span className="text-xs text-muted-foreground">{job.progress}%</span>
                            ) : (
                              <span className="text-xs text-muted-foreground">-</span>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className="text-xs text-muted-foreground">
                            {formatDuration(job.parse_duration_s, job.insert_duration_s, job.finalize_duration_s)}
                          </span>
                        </TableCell>
                        <TableCell>{formatDateTime(job.started_at ?? undefined)}</TableCell>
                        <TableCell>{formatDateTime(job.finished_at ?? undefined)}</TableCell>
                        <TableCell>
                          {job.complexity_class ? (
                            <div className="grid gap-1">
                              <span className="text-xs text-muted-foreground">
                                {job.complexity_class.replace("_", " ")}
                              </span>
                              {typeof job.eta_seconds === "number" ? (
                                <span className="text-xs text-muted-foreground">
                                  ~{Math.ceil(job.eta_seconds / 60)} min
                                </span>
                              ) : null}
                              {job.queue_priority ? (
                                <span className="text-xs text-muted-foreground">
                                  priority: {job.queue_priority}
                                </span>
                              ) : null}
                            </div>
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell>
                          {job.error ? (
                            <span className="text-xs text-destructive" title={job.error}>
                              {job.error}
                            </span>
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button variant="outline" size="sm" onClick={() => openLogsDialog(job.id)}>
                            View Logs
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </section>

        <Dialog open={logsOpen} onOpenChange={(open) => (open ? setLogsOpen(true) : closeLogsDialog())}>
          <DialogContent className="max-h-[90vh] max-w-4xl overflow-hidden">
            <DialogHeader>
              <DialogTitle>Ingestion Logs</DialogTitle>
              <DialogDescription>
                {selectedJob ? (
                  <span>
                    {docById.get(selectedJob.document_id)?.original_filename ?? "Unknown document"} | status:{" "}
                    {selectedJob.status}
                  </span>
                ) : (
                  <span>Job logs</span>
                )}
              </DialogDescription>
            </DialogHeader>

            {jobLogsError ? (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {jobLogsError}
              </div>
            ) : null}

            <div className="rounded-lg border bg-slate-950/95 p-3">
              {jobLogsLoading ? (
                <div className="h-80 overflow-auto text-xs text-slate-200">Loading logs...</div>
              ) : jobLogs.length === 0 ? (
                <div className="h-80 overflow-auto text-xs text-slate-200">
                  No logs available yet for this job.
                </div>
              ) : (
                <pre
                  ref={logsPreRef}
                  className="h-80 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-slate-100"
                >
                  {jobLogs
                    .map(
                      (entry) =>
                        `[${formatDateTime(entry.created_at)}] ${entry.level.toUpperCase()}: ${entry.message}`
                    )
                    .join("\n")}
                </pre>
              )}
            </div>

            <DialogFooter>
              <Button variant="outline" size="sm" onClick={refreshLogs} disabled={jobLogsLoading}>
                Refresh Logs
              </Button>
              <Button variant="outline" size="sm" onClick={closeLogsDialog}>
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}

function CasesDirectory() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterValue>("active");
  const [newCaseOpen, setNewCaseOpen] = useState(false);
  const [newCaseName, setNewCaseName] = useState("");
  const [newCaseDescription, setNewCaseDescription] = useState("");
  const [renameTarget, setRenameTarget] = useState<CaseSummary | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CaseSummary | null>(null);
  const [busyCaseId, setBusyCaseId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);

  useEffect(() => {
    setBackendOnline(isBackendOnline());
    return onConnectionChange(setBackendOnline);
  }, []);

  const visibleCases = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return cases.filter((item) => {
      const matchesFilter = filter === "all" || item.status === filter;
      const matchesSearch =
        normalizedSearch.length === 0 ||
        item.name.toLowerCase().includes(normalizedSearch) ||
        item.case_slug.toLowerCase().includes(normalizedSearch);
      return matchesFilter && matchesSearch;
    });
  }, [cases, filter, search]);

  const refresh = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchCases();
      setCases(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load cases.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    let active = true;
    let inFlight = false;
    const timer = window.setInterval(() => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      void fetchCases()
        .then((data) => {
          if (active) {
            setCases(data);
          }
        })
        .catch((err) => {
          if (active) {
            setError(err instanceof Error ? err.message : "Unable to refresh cases.");
          }
        })
        .finally(() => {
          inFlight = false;
        });
    }, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const handleCreate = async () => {
    if (!newCaseName.trim()) {
      setError("Case name is required.");
      return;
    }
    try {
      setBusyCaseId("create");
      const created = await createCase({
        name: newCaseName.trim(),
        description: newCaseDescription.trim() || undefined
      });
      setCases((prev) => [
        {
          ...created,
          doc_count: 0,
          active_job_count: 0
        },
        ...prev
      ]);
      setNewCaseOpen(false);
      setNewCaseName("");
      setNewCaseDescription("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create case.");
    } finally {
      setBusyCaseId(null);
    }
  };

  const handleArchiveToggle = async (item: CaseSummary) => {
    const nextStatus: CaseStatus = item.status === "active" ? "archived" : "active";
    try {
      setBusyCaseId(item.id);
      const updated = await updateCase(item.id, { status: nextStatus });
      setCases((prev) =>
        prev.map((entry) => (entry.id === item.id ? { ...entry, status: updated.status } : entry))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update case.");
    } finally {
      setBusyCaseId(null);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) {
      return;
    }
    try {
      setBusyCaseId(deleteTarget.id);
      await deleteCase(deleteTarget.id);
      setCases((prev) => prev.filter((entry) => entry.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete case.");
    } finally {
      setBusyCaseId(null);
    }
  };

  const handleRename = async (payload: { name: string; description?: string }) => {
    if (!renameTarget) {
      return;
    }
    try {
      setBusyCaseId(renameTarget.id);
      const updated = await updateCase(renameTarget.id, {
        name: payload.name,
        description: payload.description ?? null
      });
      setCases((previous) =>
        previous.map((entry) =>
          entry.id === renameTarget.id
            ? {
                ...entry,
                name: updated.name,
                description: updated.description,
                updated_at: updated.updated_at
              }
            : entry
        )
      );
      setRenameTarget(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update case.");
    } finally {
      setBusyCaseId(null);
    }
  };

  return (
    <div className="min-h-screen px-3 py-4 sm:px-4 lg:px-5">
      <div className="mx-auto flex w-full max-w-[96rem] flex-col gap-4">
        <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border bg-card/90 px-4 py-3 shadow-glow backdrop-blur">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Cases Directory</p>
            <h1 className="mt-2 text-3xl font-semibold">Rawabit 🖇️</h1>
            <p className="mt-1 max-w-xl text-sm text-muted-foreground">
              Create, archive, and manage investigative workspaces and cases.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings2 className="h-4 w-4" />
              Settings
            </Button>
          <Dialog open={newCaseOpen} onOpenChange={setNewCaseOpen}>
            <DialogTrigger asChild>
              <Button className="gap-2">
                <Plus className="h-4 w-4" />
                New Case
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create new case</DialogTitle>
                <DialogDescription>Cases are isolated workspaces stored under cases/&lt;case_slug&gt;/.</DialogDescription>
              </DialogHeader>
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <label className="text-sm font-medium" htmlFor="case-name">
                    Case name
                  </label>
                  <Input
                    id="case-name"
                    placeholder="Port Ops"
                    value={newCaseName}
                    onChange={(event) => setNewCaseName(event.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <label className="text-sm font-medium" htmlFor="case-description">
                    Description (optional)
                  </label>
                  <Input
                    id="case-description"
                    placeholder="Short description for analysts"
                    value={newCaseDescription}
                    onChange={(event) => setNewCaseDescription(event.target.value)}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setNewCaseOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreate} disabled={busyCaseId === "create"}>
                  {busyCaseId === "create" ? "Creating..." : "Create case"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          </div>
        </header>

        <section className="rounded-2xl border bg-card/90 p-6 shadow-soft backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex w-full flex-wrap items-center gap-3 md:w-auto">
              <div className="relative w-full max-w-sm">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  className="pl-9"
                  placeholder="Search cases by name or slug"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </div>
              <Select value={filter} onValueChange={(value) => setFilter(value as FilterValue)}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Filter" />
                </SelectTrigger>
                <SelectContent>
                  {filterOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-3 text-xs uppercase tracking-[0.24em] text-muted-foreground">
              <span>Cases</span>
              <span className="text-base font-semibold tracking-normal text-foreground">{visibleCases.length}</span>
            </div>
          </div>

        {error ? (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        ) : null}

        {!backendOnline ? (
          <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Reconnecting to server...
          </div>
        ) : null}

          <div className="mt-6">
            {loading ? (
              <div className="rounded-xl border border-dashed border-muted/60 p-10 text-center text-sm text-muted-foreground">
                Loading cases...
              </div>
            ) : visibleCases.length === 0 ? (
              <div className="rounded-xl border border-dashed border-muted/60 p-10 text-center">
                <p className="text-lg font-medium">No cases yet</p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Create a new case to start ingesting evidence and building the graph.
                </p>
                <Button className="mt-4" onClick={() => setNewCaseOpen(true)}>
                  <Plus className="h-4 w-4" />
                  New Case
                </Button>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Case Name</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Docs</TableHead>
                    <TableHead>Updated</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visibleCases.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell>
                        <div className="flex flex-col">
                          <div className="flex items-center gap-2">
                            <span className="text-base font-semibold text-foreground">{item.name}</span>
                            {(item.active_job_count ?? 0) > 0 ? (
                              <ProcessingIndicator label={`${item.active_job_count} job(s) in progress`} />
                            ) : null}
                          </div>
                          <span className="text-xs font-mono text-muted-foreground">{item.case_slug}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        {item.status === "archived" ? (
                          <Badge variant="muted">Archived</Badge>
                        ) : (
                          <Badge className="bg-emerald-600 text-white">Active</Badge>
                        )}
                      </TableCell>
                      <TableCell>{item.doc_count ?? 0}</TableCell>
                      <TableCell>{formatDateTime(item.updated_at)}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              window.location.href = `/cases/${item.id}`;
                            }}
                          >
                            <FolderOpen className="h-4 w-4" />
                            Open
                          </Button>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem
                                onClick={() => setRenameTarget(item)}
                                disabled={busyCaseId === item.id}
                              >
                                <Pencil className="mr-2 h-4 w-4" />
                                Rename
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                onClick={() => handleArchiveToggle(item)}
                                disabled={busyCaseId === item.id}
                              >
                                {item.status === "active" ? (
                                  <>
                                    <Archive className="mr-2 h-4 w-4" />
                                    Archive
                                  </>
                                ) : (
                                  <>
                                    <ArchiveRestore className="mr-2 h-4 w-4" />
                                    Unarchive
                                  </>
                                )}
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                className="text-destructive"
                                onClick={() => setDeleteTarget(item)}
                              >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </section>
      </div>

      <Dialog open={Boolean(deleteTarget)} onOpenChange={(open) => (!open ? setDeleteTarget(null) : null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete case</DialogTitle>
            <DialogDescription>
              This permanently deletes the case metadata and the workspace folder under cases/&lt;case_slug&gt;/. This
              action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Confirm you want to delete "{deleteTarget?.name}".
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={busyCaseId === deleteTarget?.id}>
              {busyCaseId === deleteTarget?.id ? "Deleting..." : "Delete case"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <RenameCaseDialog
        open={Boolean(renameTarget)}
        busy={busyCaseId === renameTarget?.id}
        initialName={renameTarget?.name}
        initialDescription={renameTarget?.description}
        onOpenChange={(open) => {
          if (!open) {
            setRenameTarget(null);
          }
        }}
        onSubmit={handleRename}
      />

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}

function CaseWorkspace({ caseId }: { caseId: string }) {
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [caseSummary, setCaseSummary] = useState<CaseWorkspaceSummary | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [caseSummaryRefreshing, setCaseSummaryRefreshing] = useState(false);
  const [caseSummaryRefreshError, setCaseSummaryRefreshError] = useState<string | null>(null);
  const [caseGraphExporting, setCaseGraphExporting] = useState<"entities" | "relations" | null>(null);
  const [caseGraphExportError, setCaseGraphExportError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GraphViewPayload>({
    nodes: [],
    edges: [],
    truncated: false
  });
  const [graphLimit, setGraphLimit] = useState(DEFAULT_GRAPH_LIMIT);
  const [graphRelationTypeFilters, setGraphRelationTypeFilters] = useState<string[]>([]);
  const [graphDateFrom, setGraphDateFrom] = useState("");
  const [graphDateTo, setGraphDateTo] = useState("");
  const [graphIncludeUndated, setGraphIncludeUndated] = useState(true);
  const [graphSearchQuery, setGraphSearchQuery] = useState("");
  const [graphSearchResults, setGraphSearchResults] = useState<ActorSearchResult[]>([]);
  const [graphSearchLoading, setGraphSearchLoading] = useState(false);
  const [graphControlsCollapsed, setGraphControlsCollapsed] = useState(true);
  const [graphShowLabels, setGraphShowLabels] = useState(false);
  const [selectedClusters, setSelectedClusters] = useState<string[]>([]);
  const [neighborDepth, setNeighborDepth] = useState<0 | 1 | 2>(0);
  const [graphLoading, setGraphLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [graphPaneTab, setGraphPaneTab] = useState<"original" | "focused">("original");
  const [workspaceMode, setWorkspaceMode] = useState<"workbench" | "analysis">("workbench");
  const [sidePaneTab, setSidePaneTab] = useState<"details" | "chats">("details");
  const [developerMode, setDeveloperMode] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [poleTypeFilters, setPoleTypeFilters] = useState<Set<string>>(new Set(["person", "organization", "object", "location", "event"]));
  const [backendOnline, setBackendOnline] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<WorkspaceSelectedEdge | null>(null);
  const [selectedEntityDetails, setSelectedEntityDetails] = useState<GraphEntityDetails | null>(null);
  const [selectedRelationshipDetails, setSelectedRelationshipDetails] = useState<GraphRelationshipDetails | null>(null);
  const [selectionDetailsLoading, setSelectionDetailsLoading] = useState(false);
  const [selectionDetailsError, setSelectionDetailsError] = useState<string | null>(null);
  const [legendSelection, setLegendSelection] = useState<LegendSelection>(null);
  const [relationshipsPayload, setRelationshipsPayload] = useState<RelationshipsResponse>({
    relationships: [],
    totalBeforeLimit: 0,
    totalBeforeFilter: 0
  });
  const [tagClusters, setTagClusters] = useState<TagCluster[]>([]);
  const [highlightPayload, setHighlightPayload] = useState<HighlightPayload>(emptyHighlightPayload);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatMessagesById, setChatMessagesById] = useState<Record<string, ChatMessage[]>>({});
  const [chatExpandedReferencesByMessageId, setChatExpandedReferencesByMessageId] = useState<
    Record<string, boolean>
  >({});
  const [selectedAssistantMessageId, setSelectedAssistantMessageId] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatComposer, setChatComposer] = useState("");
  const [chatMode, setChatMode] = useState<ChatMode>("hybrid");
  const [chatCreating, setChatCreating] = useState(false);
  const [chatSending, setChatSending] = useState(false);
  const [chatListOpen, setChatListOpen] = useState(false);
  const [chatDeleteTarget, setChatDeleteTarget] = useState<ChatSummary | null>(null);
  const [chatDeletingId, setChatDeletingId] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadItems, setUploadItems] = useState<UploadQueueItem[]>([]);
  const [uploadPreparing, setUploadPreparing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [caseDropActive, setCaseDropActive] = useState(false);
  const [documentSearchQuery, setDocumentSearchQuery] = useState("");
  const [documentSearchSource, setDocumentSearchSource] = useState<DocumentSearchSource>("all");
  const [documentSearchResults, setDocumentSearchResults] = useState<DocumentSearchResult[]>([]);
  const [documentSearchLoading, setDocumentSearchLoading] = useState(false);
  const [documentSearchError, setDocumentSearchError] = useState<string | null>(null);
  const [documentPreviewTarget, setDocumentPreviewTarget] = useState<EvidencePreviewTarget | null>(null);
  const [documentPreview, setDocumentPreview] = useState<DocumentSearchPreview | null>(null);
  const [documentPreviewLoading, setDocumentPreviewLoading] = useState(false);
  const [documentPreviewError, setDocumentPreviewError] = useState<string | null>(null);
  const [busyDocumentId, setBusyDocumentId] = useState<string | null>(null);
  const [evidenceModalOpen, setEvidenceModalOpen] = useState(false);
  const [evidenceModalTarget, setEvidenceModalTarget] = useState<GraphEvidence | null>(null);
  const [evidenceChunkContent, setEvidenceChunkContent] = useState<string | null>(null);
  const [evidenceChunkLoading, setEvidenceChunkLoading] = useState(false);
  const [multiSelectedNodeIds, setMultiSelectedNodeIds] = useState<Set<string>>(new Set());
  const [multiSelectedEdgeKeys, setMultiSelectedEdgeKeys] = useState<Set<string>>(new Set());
  const [editReingestTarget, setEditReingestTarget] = useState<DocumentSummary | null>(null);
  const [editReingestSourceReliability, setEditReingestSourceReliability] = useState<string>("A");
  const [editReingestInfoValidity, setEditReingestInfoValidity] = useState<string>("1");
  const [editReingestNotes, setEditReingestNotes] = useState("");
  const [editReingestProfile, setEditReingestProfile] = useState<IngestionProfile>("balanced_fast_intel");
  const [editReingestMode, setEditReingestMode] = useState<ProcessingMode>("multimodal");
  const [editReingestAdvancedOverrides, setEditReingestAdvancedOverrides] = useState("");
  const [editReingestBusy, setEditReingestBusy] = useState(false);
  const [jobLogsDocumentTarget, setJobLogsDocumentTarget] = useState<string | null>(null);
  const [docLogsOpen, setDocLogsOpen] = useState(false);
  const [docLogs, setDocLogs] = useState<IngestionJobLog[]>([]);
  const [docLogsLoading, setDocLogsLoading] = useState(false);
  const [docLogsError, setDocLogsError] = useState<string | null>(null);
  const docLogsPreRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (!evidenceModalTarget?.document_id || !evidenceModalTarget?.reference_id) {
      setEvidenceChunkContent(null);
      return;
    }
    let active = true;
    const load = async () => {
      setEvidenceChunkLoading(true);
      setEvidenceChunkContent(null);
      try {
        const preview = await fetchDocumentReferencePreview(caseId, evidenceModalTarget.document_id!, {
          reference_id: evidenceModalTarget.reference_id!,
        });
        if (active) {
          setEvidenceChunkContent(preview.content);
        }
      } catch {
        if (active) {
          setEvidenceChunkContent(null);
        }
      } finally {
        if (active) {
          setEvidenceChunkLoading(false);
        }
      }
    };
    void load();
    return () => { active = false; };
  }, [evidenceModalTarget, caseId]);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renamingCase, setRenamingCase] = useState(false);
  const [detailsPanePulse, setDetailsPanePulse] = useState(false);
  const detailsRequestRef = useRef(0);
  const detailsPaneRef = useRef<HTMLElement | null>(null);
  const detailsPulseTimeoutRef = useRef<number | null>(null);
  const highlightAutoloadAttemptsRef = useRef<Map<string, number>>(new Map());
  const completedJobIdsRef = useRef<Set<string>>(new Set());
  const chatThreadRef = useRef<HTMLDivElement | null>(null);
  const chatComposerRef = useRef<HTMLTextAreaElement | null>(null);
  const chatShouldStickToBottomRef = useRef(true);
  const chatForceScrollRef = useRef(false);
  const chatThreadStateRef = useRef<{
    chatId: string | null;
    lastMessageKey: string | null;
    messageCount: number;
  }>({
    chatId: null,
    lastMessageKey: null,
    messageCount: 0
  });

  useEffect(() => {
    return () => {
      if (detailsPulseTimeoutRef.current !== null) {
        window.clearTimeout(detailsPulseTimeoutRef.current);
      }
    };
  }, []);

  const jobByDocumentId = useMemo(() => {
    const map = new Map<string, IngestionJob>();
    jobs.forEach((job) => {
      if (!map.has(job.document_id)) {
        map.set(job.document_id, job);
      }
    });
    return map;
  }, [jobs]);

  const documentsById = useMemo(() => {
    const map = new Map<string, DocumentSummary>();
    documents.forEach((doc) => {
      map.set(doc.id, doc);
    });
    return map;
  }, [documents]);

  const sortedDocuments = useMemo(() => sortDocumentsByCreatedAt(documents), [documents]);

  const docLogsJobId = useMemo(() => {
    if (!jobLogsDocumentTarget) return null;
    const job = jobByDocumentId.get(jobLogsDocumentTarget);
    return job?.id ?? null;
  }, [jobLogsDocumentTarget, jobByDocumentId]);

  const isCaseFresh = useMemo(() => {
    const completedCount = documents.filter(
      (d) => d.ingestion_status === "complete" || d.ingestion_status === "completed_with_warnings"
    ).length;
    const hasNoCompleted = completedCount === 0;
    const hasDocuments = documents.length > 0;
    const hasProcessingJobs = jobs.some((job) => isProcessingStatus(job.status));
    return hasNoCompleted && hasDocuments && hasProcessingJobs;
  }, [documents, jobs]);

  const ingestionBanner = useMemo(() => {
    const completedCount = documents.filter(
      (d) => d.ingestion_status === "complete" || d.ingestion_status === "completed_with_warnings"
    ).length;
    const processingCount = documents.filter(
      (d) => d.ingestion_status !== "complete" &&
        d.ingestion_status !== "completed_with_warnings" &&
        d.ingestion_status !== "failed"
    ).length;
    if (completedCount > 0 && processingCount > 0) {
      return { completedCount, total: documents.length };
    }
    return null;
  }, [documents]);

  useEffect(() => {
    setBackendOnline(isBackendOnline());
    return onConnectionChange(setBackendOnline);
  }, []);

  const documentsByStoredPath = useMemo(() => {
    const map = new Map<string, DocumentSummary>();
    documents.forEach((doc) => {
      map.set(doc.stored_file_path.toLowerCase(), doc);
      map.set(pathBasename(doc.stored_file_path).toLowerCase(), doc);
    });
    return map;
  }, [documents]);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [activeChatId, chats]
  );

  const activeChatMessages = useMemo(
    () => (activeChatId ? chatMessagesById[activeChatId] ?? [] : []),
    [activeChatId, chatMessagesById]
  );
  const latestActiveChatMessage = activeChatMessages[activeChatMessages.length - 1] ?? null;
  const hasCaseEvidence = documents.length > 0;
  const hasPendingUserQuestion = latestActiveChatMessage?.role === "user";
  const effectiveChatMode: ChatMode = developerMode ? chatMode : "hybrid";
  const chatSendBlockedReason = !hasCaseEvidence
    ? "Upload evidence before asking questions."
    : hasPendingUserQuestion
      ? "Wait for the current question to be answered before sending another message."
      : null;
  const canSendChatMessage =
    chatComposer.trim().length > 0 &&
    !chatSending &&
    !chatCreating &&
    chatSendBlockedReason === null;

  const selectedNode = useMemo(
    () => graphData.nodes.find((node: GraphNode) => node.id === selectedNodeId) ?? null,
    [graphData.nodes, selectedNodeId]
  );

  const selectedGraphEdge = useMemo(() => {
    if (!selectedEdge) {
      return null;
    }
    return (
      graphData.edges.find(
        (edge: GraphEdge) =>
          edge.id === selectedEdge.id ||
          (edge.src_id === selectedEdge.src_id && edge.tgt_id === selectedEdge.tgt_id)
      ) ?? null
    );
  }, [graphData.edges, selectedEdge]);

  const selectedRelationship = useMemo<RelationshipRow | null>(() => {
    if (!selectedEdge) {
      return null;
    }
    return (
      relationshipsPayload.relationships.find((rel) => rel.id === selectedEdge.id) ||
      relationshipsPayload.relationships.find(
        (rel) =>
          (rel.actor_id?.trim() || rel.actor) === selectedEdge.src_id &&
          (rel.target_id?.trim() || rel.target) === selectedEdge.tgt_id
      ) ||
      null
    );
  }, [relationshipsPayload.relationships, selectedEdge]);

  useEffect(() => {
    let ignore = false;

    async function loadSelectionDetails() {
      setSelectedEntityDetails(null);
      setSelectedRelationshipDetails(null);
      setSelectionDetailsError(null);

      if (selectedNodeId) {
        setSelectionDetailsLoading(true);
        try {
          const details = await fetchGraphEntity(caseId, selectedNodeId);
          if (!ignore) {
            setSelectedEntityDetails(details);
          }
        } catch (err) {
          if (!ignore) {
            setSelectionDetailsError(err instanceof Error ? err.message : "Unable to load entity details.");
          }
        } finally {
          if (!ignore) {
            setSelectionDetailsLoading(false);
          }
        }
        return;
      }

      if (selectedEdge?.src_id && selectedEdge.tgt_id) {
        setSelectionDetailsLoading(true);
        try {
          const details = await fetchGraphRelationship(caseId, {
            src_id: selectedEdge.src_id,
            tgt_id: selectedEdge.tgt_id
          });
          if (!ignore) {
            setSelectedRelationshipDetails(details);
          }
        } catch (err) {
          if (!ignore) {
            setSelectionDetailsError(err instanceof Error ? err.message : "Unable to load relationship details.");
          }
        } finally {
          if (!ignore) {
            setSelectionDetailsLoading(false);
          }
        }
        return;
      }

      setSelectionDetailsLoading(false);
    }

    void loadSelectionDetails();

    return () => {
      ignore = true;
    };
  }, [caseId, selectedEdge?.src_id, selectedEdge?.tgt_id, selectedNodeId]);

  const graphRelationTypeOptions = useMemo(() => {
    const values = new Set<string>();
    graphData.edges.forEach((edge: GraphEdge) => {
      const relation = edge.relation_type.trim();
      if (relation) {
        values.add(relation);
      }
    });
    graphRelationTypeFilters.forEach((value) => values.add(value));
    return [...values].sort((left, right) => left.localeCompare(right));
  }, [graphData.edges, graphRelationTypeFilters]);

  const graphNodeIdSet = useMemo(() => new Set(graphData.nodes.map((node: GraphNode) => node.id)), [graphData.nodes]);
  const graphEdgeIdSet = useMemo(() => new Set(graphData.edges.map((edge: GraphEdge) => edge.id)), [graphData.edges]);
  const graphEdgePairSet = useMemo(() => {
    const values = new Set<string>();
    graphData.edges.forEach((edge: GraphEdge) => {
      values.add(pairKey(edge.src_id, edge.tgt_id));
      values.add(pairKey(edge.tgt_id, edge.src_id));
    });
    return values;
  }, [graphData.edges]);

  const hasHighlights =
    highlightPayload.highlight_entities.length > 0 || highlightPayload.highlight_relationships.length > 0;
  const hasActiveGraphFilters =
    graphRelationTypeFilters.length > 0 ||
    selectedClusters.length > 0 ||
    Boolean(graphDateFrom) ||
    Boolean(graphDateTo) ||
    !graphIncludeUndated ||
    neighborDepth > 0;
  const hasFocusedState = hasHighlights || hasActiveGraphFilters;

  const displayedGraphData = useMemo<GraphViewPayload>(() => {
    if (neighborDepth === 0 || !selectedNodeId) {
      return graphData;
    }
    if (!graphNodeIdSet.has(selectedNodeId)) {
      return graphData;
    }

    const adjacency = new Map<string, Set<string>>();
    graphData.nodes.forEach((node: GraphNode) => {
      adjacency.set(node.id, new Set<string>());
    });
    graphData.edges.forEach((edge: GraphEdge) => {
      adjacency.get(edge.src_id)?.add(edge.tgt_id);
      adjacency.get(edge.tgt_id)?.add(edge.src_id);
    });

    const selectedIds = new Set<string>([selectedNodeId]);
    let frontier = new Set<string>([selectedNodeId]);
    for (let hop = 0; hop < neighborDepth; hop += 1) {
      const next = new Set<string>();
      frontier.forEach((frontierNodeId) => {
        (adjacency.get(frontierNodeId) ?? new Set<string>()).forEach((neighborId) => {
          if (!selectedIds.has(neighborId)) {
            selectedIds.add(neighborId);
            next.add(neighborId);
          }
        });
      });
      frontier = next;
      if (frontier.size === 0) {
        break;
      }
    }

    return {
      nodes: graphData.nodes.filter((node) => selectedIds.has(node.id)),
      edges: graphData.edges.filter(
        (edge) => selectedIds.has(edge.src_id) && selectedIds.has(edge.tgt_id)
      ),
      truncated: graphData.truncated,
    };
  }, [graphData, graphNodeIdSet, neighborDepth, selectedNodeId]);

  const focusedGraphData = useMemo<GraphViewPayload>(() => {
    const focusedNodeIds = new Set<string>();
    highlightPayload.highlight_entities.forEach((entityId) => {
      if (entityId) {
        focusedNodeIds.add(entityId);
      }
    });

    const highlightedEdgeIds = new Set<string>();
    const highlightedEdgePairs = new Set<string>();
    highlightPayload.highlight_relationships.forEach((relationship) => {
      if (relationship.edge_id) {
        highlightedEdgeIds.add(relationship.edge_id);
      }
      if (relationship.src_id && relationship.tgt_id) {
        focusedNodeIds.add(relationship.src_id);
        focusedNodeIds.add(relationship.tgt_id);
        highlightedEdgePairs.add(pairKey(relationship.src_id, relationship.tgt_id));
        highlightedEdgePairs.add(pairKey(relationship.tgt_id, relationship.src_id));
      }
    });

    const edges = displayedGraphData.edges.filter((edge) => {
      if (highlightedEdgeIds.has(edge.id)) {
        return true;
      }
      return highlightedEdgePairs.has(pairKey(edge.src_id, edge.tgt_id));
    });

    edges.forEach((edge) => {
      focusedNodeIds.add(edge.src_id);
      focusedNodeIds.add(edge.tgt_id);
    });

    const nodes = displayedGraphData.nodes.filter((node) => focusedNodeIds.has(node.id));
    return {
      nodes,
      edges,
      truncated: false
    };
  }, [
    displayedGraphData.edges,
    displayedGraphData.nodes,
    highlightPayload.highlight_entities,
    highlightPayload.highlight_relationships
  ]);

  const activeGraphData =
    graphPaneTab === "focused" && hasHighlights ? focusedGraphData : displayedGraphData;
  const legendHighlightedNodeIds = useMemo(() => {
    if (legendSelection?.kind !== "node") {
      return [];
    }
    return activeGraphData.nodes
      .filter((node) => normalizeEntityType(node.entity_type) === legendSelection.value)
      .map((node) => node.id);
  }, [activeGraphData.nodes, legendSelection]);

  const legendHighlightedEdges = useMemo(() => {
    if (legendSelection?.kind !== "edge") {
      return [];
    }
    return activeGraphData.edges
      .filter((edge) => normalizeEntityType(edge.relation_type || edge.label) === legendSelection.value)
      .map((edge) => ({
        src_id: edge.src_id,
        tgt_id: edge.tgt_id,
        edge_id: edge.id
      }));
  }, [activeGraphData.edges, legendSelection]);

  const hasFocusedGraphData =
    graphPaneTab === "focused" && hasHighlights
      ? focusedGraphData.nodes.length > 0 || focusedGraphData.edges.length > 0
      : displayedGraphData.nodes.length > 0 || displayedGraphData.edges.length > 0;
  const activeHighlightedNodeIds =
    graphPaneTab === "focused" && hasHighlights
      ? [...new Set([...highlightPayload.highlight_entities, ...legendHighlightedNodeIds])]
      : [...new Set(legendHighlightedNodeIds)];
  const activeHighlightedEdges =
    graphPaneTab === "focused" && hasHighlights
      ? [...highlightPayload.highlight_relationships, ...legendHighlightedEdges]
      : [...legendHighlightedEdges];

  const handleLegendSelect = useCallback((selection: LegendSelection) => {
    setLegendSelection(selection);
  }, []);

  const resolveEvidenceDocument = useCallback(
    (evidence: GraphEvidence) => {
      if (evidence.document_id) {
        const byId = documentsById.get(evidence.document_id);
        if (byId) {
          return byId;
        }
      }
      return documentsByStoredPath.get(evidence.file_path.toLowerCase()) ??
        documentsByStoredPath.get(pathBasename(evidence.file_path).toLowerCase()) ??
        null;
    },
    [documentsById, documentsByStoredPath]
  );

  const refreshCaseSummaryData = useCallback(async () => {
    try {
      const summary = await fetchCaseSummary(caseId);
      setCaseSummary(summary);
    } catch {
      setCaseSummary(null);
    }
  }, [caseId]);

  const handleRefreshCaseSummary = useCallback(async () => {
    try {
      setCaseSummaryRefreshing(true);
      setCaseSummaryRefreshError(null);
      const summary = await refreshCaseSummary(caseId);
      setCaseSummary(summary);
    } catch (err) {
      setCaseSummaryRefreshError(err instanceof Error ? err.message : "Unable to refresh case summary.");
    } finally {
      setCaseSummaryRefreshing(false);
    }
  }, [caseId]);

  const handleExportGraphCsv = useCallback(
    async (kind: "entities" | "relations") => {
      try {
        setCaseGraphExporting(kind);
        setCaseGraphExportError(null);
        if (kind === "entities") {
          await downloadGraphEntitiesCsv(caseId);
        } else {
          await downloadGraphRelationsCsv(caseId);
        }
      } catch (err) {
        setCaseGraphExportError(
          err instanceof Error ? err.message : "Unable to export graph CSV."
        );
      } finally {
        setCaseGraphExporting(null);
      }
    },
    [caseId]
  );


  const refresh = async () => {
    try {
      setLoading(true);
      setError(null);
      const [caseInfo, docs] = await Promise.all([fetchCase(caseId), fetchDocuments(caseId)]);
      setCaseDetail(caseInfo);
      setCaseSummary(caseInfo.summary ?? null);
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load case.");
    } finally {
      setLoading(false);
    }
  };

  const refreshGraph = useCallback(async (options?: { showLoading?: boolean }) => {
    const showLoading = options?.showLoading !== false;
    try {
      if (showLoading) {
        setGraphLoading(true);
      }
      setGraphError(null);
      const limit = graphLimit;

      if (poleTypeFilters.size === 0) {
        setGraphData({ nodes: [], edges: [], truncated: false });
        setRelationshipsPayload({ relationships: [], totalBeforeLimit: 0, totalBeforeFilter: 0 });
        return;
      }

      const entityTypes =
        poleTypeFilters.size === 5
          ? undefined
          : [...poleTypeFilters];

      const payload = await fetchGraph(caseId, {
        limit,
        entity_types: entityTypes,
        relation_types: graphRelationTypeFilters,
        date_from: graphDateFrom || undefined,
        date_to: graphDateTo || undefined,
        include_undated: graphIncludeUndated,
      });
      setGraphData({
        nodes: payload.nodes,
        edges: payload.edges,
        truncated: payload.truncated ?? false,
      });
      setRelationshipsPayload({ relationships: [], totalBeforeLimit: 0, totalBeforeFilter: 0 });
    } catch (err) {
      setGraphData({ nodes: [], edges: [], truncated: false });
      setGraphError(err instanceof Error ? err.message : "Unable to load graph.");
    } finally {
      if (showLoading) {
        setGraphLoading(false);
      }
    }
  }, [
    caseId,
    poleTypeFilters,
    graphRelationTypeFilters,
    selectedClusters,
    graphLimit,
    graphDateFrom,
    graphDateTo,
    graphIncludeUndated,
  ]);

  const applyAssistantMessageSelection = useCallback(
    (
      message: ChatMessage,
      options?: {
        updateGraph?: boolean;
        updateSelection?: boolean;
      }
    ) => {
      const highlight = extractAssistantHighlight(message);
      if (!highlight) {
        return false;
      }
      if (options?.updateSelection !== false) {
        setSelectedAssistantMessageId(message.id);
      }
      setHighlightPayload(highlight);
      if (options?.updateGraph !== false) {
        setGraphPaneTab("focused");
      }
      return true;
    },
    []
  );

  const applyHighlightFromChatMessages = useCallback((messages: ChatMessage[]) => {
    if (selectedAssistantMessageId) {
      const selectedMessage = messages.find(
        (message) => message.id === selectedAssistantMessageId && message.role === "assistant"
      );
      if (
        selectedMessage &&
        applyAssistantMessageSelection(selectedMessage, {
          updateGraph: false,
          updateSelection: false,
        })
      ) {
        return;
      }
      setSelectedAssistantMessageId(null);
    }
    if (graphPaneTab !== "focused") {
      return;
    }
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role !== "assistant") {
        continue;
      }
      const candidate = extractAssistantHighlight(messages[index]);
      if (candidate) {
        applyAssistantMessageSelection(messages[index], { updateGraph: false, updateSelection: false });
        return;
      }
    }
    setHighlightPayload(emptyHighlightPayload);
  }, [applyAssistantMessageSelection, graphPaneTab, selectedAssistantMessageId]);

  const loadChatDetail = useCallback(
    async (chatId: string) => {
      try {
        setChatLoading(true);
        setChatError(null);
        setSidePaneTab("chats");
        const detail = await fetchChat(caseId, chatId);
        setActiveChatId(detail.id);
        setChats((previous) => {
          const withoutCurrent = previous.filter((chat) => chat.id !== detail.id);
          const merged = [
            {
              id: detail.id,
              case_id: detail.case_id,
              title: detail.title,
              created_at: detail.created_at,
              updated_at: detail.updated_at,
            },
            ...withoutCurrent,
          ];
          return sortChatsByUpdated(merged);
        });
        setChatMessagesById((previous) => {
          const currentMessages = previous[detail.id] ?? [];
          if (areChatMessagesEqual(currentMessages, detail.messages)) {
            return previous;
          }
          return {
            ...previous,
            [detail.id]: detail.messages,
          };
        });
        applyHighlightFromChatMessages(detail.messages);
        return true;
      } catch (err) {
        setChatError(err instanceof Error ? err.message : "Unable to load chat history.");
        return false;
      } finally {
        setChatLoading(false);
      }
    },
    [applyHighlightFromChatMessages, caseId]
  );

  const refreshChats = useCallback(
    async (preferredChatId?: string | null) => {
      try {
        setChatLoading(true);
        setChatError(null);
        const list = sortChatsByUpdated(await fetchChats(caseId));
        setChats(list);
        if (list.length === 0) {
          setActiveChatId(null);
          setHighlightPayload(emptyHighlightPayload);
          return;
        }
        const targetChatId =
          (preferredChatId && list.some((chat) => chat.id === preferredChatId)
            ? preferredChatId
            : null) ??
          (activeChatId && list.some((chat) => chat.id === activeChatId)
            ? activeChatId
            : null) ??
          list[0].id;
        if (!targetChatId) {
          return;
        }
        await loadChatDetail(targetChatId);
      } catch (err) {
        setChatError(err instanceof Error ? err.message : "Unable to load chats.");
      } finally {
        setChatLoading(false);
      }
    },
    [activeChatId, caseId, loadChatDetail]
  );

  const handleCreateChat = useCallback(
    async (title?: string) => {
      try {
        setChatCreating(true);
        setChatError(null);
        setSidePaneTab("chats");
        const created = await createChat(caseId, title ? { title } : undefined);
        setChats((previous) => sortChatsByUpdated([created, ...previous.filter((chat) => chat.id !== created.id)]));
        setChatMessagesById((previous) => ({ ...previous, [created.id]: [] }));
        setActiveChatId(created.id);
        setHighlightPayload(emptyHighlightPayload);
        const loaded = await loadChatDetail(created.id);
        if (loaded) {
          setChatListOpen(false);
        }
      } catch (err) {
        setChatError(err instanceof Error ? err.message : "Unable to create chat.");
      } finally {
        setChatCreating(false);
      }
    },
    [caseId, loadChatDetail]
  );

  const handleSelectChat = useCallback(
    async (chatId: string) => {
      const loaded = await loadChatDetail(chatId);
      if (loaded) {
        setChatListOpen(false);
      }
    },
    [loadChatDetail]
  );

  const handleDeleteChat = useCallback((chat: ChatSummary) => {
    setChatError(null);
    setChatDeleteTarget(chat);
  }, []);

  const closeDeleteChatDialog = useCallback(() => {
    if (chatDeletingId) {
      return;
    }
    setChatDeleteTarget(null);
  }, [chatDeletingId]);

  const confirmDeleteChat = useCallback(async () => {
    if (!chatDeleteTarget) {
      return;
    }
    const target = chatDeleteTarget;
    try {
      setChatDeletingId(target.id);
      setChatError(null);
      await deleteChat(caseId, target.id);
      setChats((previous) => previous.filter((chat) => chat.id !== target.id));
      setChatMessagesById((previous) => {
        const next = { ...previous };
        delete next[target.id];
        return next;
      });
      if (activeChatId === target.id) {
        setActiveChatId(null);
        setSelectedAssistantMessageId(null);
        setHighlightPayload(emptyHighlightPayload);
      }
      setChatDeleteTarget(null);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Unable to delete chat.");
    } finally {
      setChatDeletingId(null);
    }
  }, [activeChatId, caseId, chatDeleteTarget]);

  const handleSendChatMessage = useCallback(async () => {
    const content = chatComposer.trim();
    if (!content || chatSending || chatSendBlockedReason) {
      if (chatSendBlockedReason) {
        setChatError(chatSendBlockedReason);
      }
      return;
    }
    let targetChatId = activeChatId;
    const temporaryMessageId = `temp-user-${Date.now()}`;
    const temporaryUserMessage: ChatMessage = {
      id: temporaryMessageId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
      rag_metadata: null
    };
    const titlePreview = content.length > 72 ? `${content.slice(0, 69).trimEnd()}...` : content;

    try {
      setChatSending(true);
      setChatError(null);
      setSidePaneTab("chats");

      if (!targetChatId) {
        const created = await createChat(caseId, { title: titlePreview });
        targetChatId = created.id;
        setChats((previous) =>
          sortChatsByUpdated([created, ...previous.filter((chat) => chat.id !== created.id)])
        );
        setChatMessagesById((previous) => ({ ...previous, [created.id]: [] }));
        setActiveChatId(created.id);
      }
      if (!targetChatId) {
        throw new Error("Unable to create a chat session.");
      }

      setChatComposer("");
      chatForceScrollRef.current = true;
      setChatMessagesById((previous) => ({
        ...previous,
        [targetChatId as string]: [
          ...(previous[targetChatId as string] ?? []),
          temporaryUserMessage
        ]
      }));

      const hasMultiSelectContent = multiSelectedNodeIds.size > 0 || multiSelectedEdgeKeys.size > 0;
      const requestOptions: Record<string, unknown> = {};
      if (hasMultiSelectContent) {
        const selectedEntities: Array<{ id: string; name: string; type: string }> = [];
        multiSelectedNodeIds.forEach((nodeId) => {
          const node = graphData.nodes.find((n: GraphNode) => n.id === nodeId);
          selectedEntities.push({
            id: nodeId,
            name: node?.label ?? nodeId,
            type: node?.entity_type ?? "unknown"
          });
        });
        const selectedRelationships: Array<{ src_id: string; src_name: string; tgt_id: string; tgt_name: string; relation_type: string }> = [];
        multiSelectedEdgeKeys.forEach((key) => {
          const [srcId, tgtId] = key.split("||");
          const edge = graphData.edges.find(
            (e: GraphEdge) => e.src_id === srcId && e.tgt_id === tgtId
          );
          if (edge) {
            const srcNode = graphData.nodes.find((n: GraphNode) => n.id === srcId);
            const tgtNode = graphData.nodes.find((n: GraphNode) => n.id === tgtId);
            selectedRelationships.push({
              src_id: srcId,
              src_name: srcNode?.label ?? srcId,
              tgt_id: tgtId,
              tgt_name: tgtNode?.label ?? tgtId,
              relation_type: edge.relation_type || "ASSOCIATED_WITH"
            });
          }
        });
        requestOptions.selected_entities = selectedEntities;
        requestOptions.selected_relationships = selectedRelationships;
      }

      const response = await sendChatMessage(caseId, targetChatId, {
        content,
        mode: effectiveChatMode,
        options: hasMultiSelectContent ? requestOptions : undefined
      });
      if (hasMultiSelectContent) {
        setMultiSelectedNodeIds(new Set());
        setMultiSelectedEdgeKeys(new Set());
      }
      const normalizedHighlight = normalizeHighlightPayload({
        ...(response.highlight ?? {}),
        references:
          response.highlight?.references && response.highlight.references.length > 0
            ? response.highlight.references
            : response.references,
        supporting_chunks:
          response.highlight?.supporting_chunks && response.highlight.supporting_chunks.length > 0
            ? response.highlight.supporting_chunks
            : response.chunks
      });
      setHighlightPayload(normalizedHighlight);
      await loadChatDetail(targetChatId);
    } catch (err) {
      if (targetChatId) {
        setChatMessagesById((previous) => ({
          ...previous,
          [targetChatId as string]: (previous[targetChatId as string] ?? []).filter(
            (item) => item.id !== temporaryMessageId
          )
        }));
      }
      setChatError(err instanceof Error ? err.message : "Unable to send message.");
    } finally {
      setChatSending(false);
    }
  }, [activeChatId, caseId, chatComposer, chatSendBlockedReason, chatSending, effectiveChatMode, loadChatDetail, graphData, multiSelectedNodeIds, multiSelectedEdgeKeys]);

  const openChatReferencePreview = useCallback(
    async (reference: { reference_id: string; file_path: string }, snippet?: string | null) => {
      const filePath = reference.file_path;
      const normalized = filePath.toLowerCase();
      const matchedDocument =
        documentsByStoredPath.get(normalized) ??
        documentsByStoredPath.get(pathBasename(filePath).toLowerCase()) ??
        null;
      if (!matchedDocument) {
        setChatError("The referenced evidence file is not available in this case.");
        return;
      }
      const target: EvidencePreviewTarget = {
        document_id: matchedDocument.id,
        original_filename: matchedDocument.original_filename,
        confidence_code: matchedDocument.confidence_code,
        source_kind: "processed",
        segment_key: reference.reference_id,
        stored_file_path: matchedDocument.stored_file_path,
        snippet
      };
      setDocumentPreviewTarget(target);
      setDocumentPreview(null);
      setDocumentPreviewError(null);
      setDocumentPreviewLoading(true);
      try {
        const preview = await fetchDocumentReferencePreview(caseId, matchedDocument.id, {
          reference_id: reference.reference_id,
          q: snippet || reference.reference_id,
          snippet: snippet || undefined
        });
        setDocumentPreview(preview);
      } catch (err) {
        setDocumentPreviewError(
          err instanceof Error ? err.message : "Unable to load referenced evidence preview."
        );
      } finally {
        setDocumentPreviewLoading(false);
      }
    },
    [caseId, documentsByStoredPath]
  );

  const resetGraphScope = useCallback(() => {
    setGraphPaneTab("original");
    setGraphRelationTypeFilters([]);
    setSelectedClusters([]);
    setGraphDateFrom("");
    setGraphDateTo("");
    setGraphIncludeUndated(true);
    setNeighborDepth(0);
    setSelectedAssistantMessageId(null);
    setLegendSelection(null);
    setHighlightPayload(emptyHighlightPayload);
  }, []);

  const handleRenameCase = useCallback(
    async (payload: { name: string; description?: string }) => {
      if (!caseDetail) {
        return;
      }
      try {
        setRenamingCase(true);
        setError(null);
        const updated = await updateCase(caseId, {
          name: payload.name,
          description: payload.description ?? null
        });
        setCaseDetail((previous) =>
          previous
            ? {
                ...previous,
                name: updated.name,
                description: updated.description,
                updated_at: updated.updated_at
              }
            : previous
        );
        setCaseSummary((previous) =>
          previous
            ? {
                ...previous,
                case_name: updated.name
              }
            : previous
        );
        setRenameOpen(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to update case.");
      } finally {
        setRenamingCase(false);
      }
    },
    [caseDetail, caseId]
  );

  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedEdge(null);
    setLegendSelection(null);
    setHighlightPayload(emptyHighlightPayload);
    setCaseSummary(null);
    setGraphPaneTab("original");
    setSidePaneTab("details");
    setDeveloperMode(false);
    setGraphLimit(DEFAULT_GRAPH_LIMIT);
    setGraphRelationTypeFilters([]);
    setSelectedClusters([]);
    setGraphDateFrom("");
    setGraphDateTo("");
    setGraphIncludeUndated(true);
    setGraphSearchQuery("");
    setGraphSearchResults([]);
    setGraphControlsCollapsed(true);
    setNeighborDepth(0);
    setChats([]);
    setActiveChatId(null);
    setChatMessagesById({});
    setChatExpandedReferencesByMessageId({});
    setSelectedAssistantMessageId(null);
    setChatError(null);
    setChatComposer("");
    setChatMode("hybrid");
    setChatDeleteTarget(null);
    setChatDeletingId(null);
    setRenameOpen(false);
    setRenamingCase(false);
    completedJobIdsRef.current = new Set();
    highlightAutoloadAttemptsRef.current.clear();
    void refresh();
    void refreshCaseSummaryData();
    void refreshChats(null);
  }, [caseId]);

  useEffect(() => {
    if (!developerMode && chatMode !== "hybrid") {
      setChatMode("hybrid");
    }
  }, [chatMode, developerMode]);

  useEffect(() => {
    void refreshGraph();
  }, [refreshGraph]);

  useEffect(() => {
    let active = true;
    let inFlight = false;
    const timer = window.setInterval(() => {
      if (!active || inFlight) {
        return;
      }
      inFlight = true;
      void refreshGraph({ showLoading: false }).finally(() => {
        inFlight = false;
      });
    }, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [refreshGraph]);

  useEffect(() => {
    if (neighborDepth > 0 && (!selectedNodeId || !graphNodeIdSet.has(selectedNodeId))) {
      setNeighborDepth(0);
    }
  }, [graphNodeIdSet, neighborDepth, selectedNodeId]);

  useEffect(() => {
    if (hasActiveGraphFilters) {
      setGraphPaneTab("focused");
    }
  }, [hasActiveGraphFilters]);

  useEffect(() => {
    if (selectedNodeId && !graphData.nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(null);
    }
    if (
      selectedEdge &&
      !graphData.edges.some(
        (edge) =>
          edge.id === selectedEdge.id ||
          (edge.src_id === selectedEdge.src_id && edge.tgt_id === selectedEdge.tgt_id)
      )
    ) {
      setSelectedEdge(null);
    }
  }, [graphData.edges, graphData.nodes, selectedEdge, selectedNodeId]);

  useEffect(() => {
    if (!docLogsOpen || !docLogsJobId) return;
    let cancelled = false;
    const load = async () => {
      try {
        setDocLogsLoading(true);
        setDocLogsError(null);
        const data = await fetchJobLogs(caseId, docLogsJobId!, { limit: 1000 });
        if (cancelled) return;
        setDocLogs(data);
      } catch (err) {
        if (!cancelled) setDocLogsError(err instanceof Error ? err.message : "Unable to load logs.");
      } finally {
        if (!cancelled) setDocLogsLoading(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [caseId, docLogsOpen, docLogsJobId]);

  useEffect(() => {
    if (!docLogsPreRef.current) return;
    docLogsPreRef.current.scrollTop = docLogsPreRef.current.scrollHeight;
  }, [docLogs]);

  useEffect(() => {
    if (jobLogsDocumentTarget) {
      setDocLogsOpen(true);
      setDocLogs([]);
    }
  }, [jobLogsDocumentTarget]);


  useEffect(() => {
    const onHighlight = (event: Event) => {
      const payload = (event as CustomEvent<Partial<HighlightPayload>>).detail;
      setHighlightPayload(normalizeHighlightPayload(payload));
    };
    window.addEventListener("rawabit:graph-highlight", onHighlight as EventListener);
    return () => window.removeEventListener("rawabit:graph-highlight", onHighlight as EventListener);
  }, []);

  const handleChatThreadScroll = useCallback(() => {
    const thread = chatThreadRef.current;
    if (!thread) {
      return;
    }
    chatShouldStickToBottomRef.current = isChatThreadNearBottom(thread);
  }, []);

  useEffect(() => {
    const thread = chatThreadRef.current;
    const nextState = {
      chatId: activeChatId,
      lastMessageKey: chatMessageKey(activeChatMessages[activeChatMessages.length - 1]),
      messageCount: activeChatMessages.length
    };

    if (!thread) {
      chatThreadStateRef.current = nextState;
      chatForceScrollRef.current = false;
      return;
    }

    const previousState = chatThreadStateRef.current;
    const messagesChanged =
      nextState.messageCount !== previousState.messageCount ||
      nextState.lastMessageKey !== previousState.lastMessageKey;
    const shouldScroll =
      (activeChatId !== null && activeChatId !== previousState.chatId) ||
      chatForceScrollRef.current ||
      (messagesChanged && chatShouldStickToBottomRef.current);

    if (shouldScroll) {
      thread.scrollTop = thread.scrollHeight;
      chatShouldStickToBottomRef.current = true;
    }

    chatThreadStateRef.current = nextState;
    chatForceScrollRef.current = false;
  }, [activeChatId, activeChatMessages]);

  const adjustChatComposerHeight = useCallback(() => {
    const textarea = chatComposerRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    const nextHeight = Math.max(
      CHAT_COMPOSER_MIN_HEIGHT_PX,
      Math.min(textarea.scrollHeight, CHAT_COMPOSER_MAX_HEIGHT_PX)
    );
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > CHAT_COMPOSER_MAX_HEIGHT_PX ? "auto" : "hidden";
  }, []);

  useEffect(() => {
    adjustChatComposerHeight();
  }, [adjustChatComposerHeight, chatComposer]);

  const toggleRelationTypeFilter = useCallback((relationType: string) => {
    setGraphPaneTab("focused");
    setGraphRelationTypeFilters((previous) =>
      previous.includes(relationType)
        ? previous.filter((value) => value !== relationType)
        : [...previous, relationType]
    );
  }, []);

  const searchDebounceRef = useRef<number | undefined>(undefined);

  const handleGraphSearchChange = useCallback(
    (value: string) => {
      setGraphSearchQuery(value);
      if (searchDebounceRef.current) {
        window.clearTimeout(searchDebounceRef.current);
      }
      if (!value.trim()) {
        setGraphSearchResults([]);
        return;
      }
      searchDebounceRef.current = window.setTimeout(async () => {
        try {
          setGraphSearchLoading(true);
          const data = await apiSearchActors(caseId, value.trim());
          setGraphSearchResults(data);
        } catch (err) {
          setGraphSearchResults([]);
        } finally {
          setGraphSearchLoading(false);
        }
      }, 200);
    },
    [caseId]
  );


  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const data = await fetchJobs(caseId);
        if (active) {
          const nextCompleted = new Set(
            data
              .filter((job) => job.status === "complete" || job.status === "completed_with_warnings")
              .map((job) => job.id)
          );
          const hasNewComplete = [...nextCompleted].some(
            (jobId) => !completedJobIdsRef.current.has(jobId)
          );
          completedJobIdsRef.current = nextCompleted;
          setJobs(data);
          if (hasNewComplete) {
            void refreshCaseSummaryData();
            void refreshGraph({ showLoading: false });
          }
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unable to load ingestion jobs.");
        }
      }
    };

    void poll();
    timer = window.setInterval(() => {
      void poll();
    }, 3000);

    return () => {
      active = false;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [caseId, refreshCaseSummaryData, refreshGraph]);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const pollWorkspace = async () => {
      try {
        const [caseInfo, docs] = await Promise.all([fetchCase(caseId), fetchDocuments(caseId)]);
        if (!active) {
          return;
        }
        setCaseDetail(caseInfo);
        setCaseSummary(caseInfo.summary ?? null);
        setDocuments(docs);
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unable to refresh case workspace.");
        }
      }
    };

    void pollWorkspace();
    timer = window.setInterval(() => {
      void pollWorkspace();
    }, 3000);

    return () => {
      active = false;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [caseId]);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const pollChats = async () => {
      try {
        const list = sortChatsByUpdated(await fetchChats(caseId));
        if (!active) {
          return;
        }
        setChats(list);
        if (!activeChatId) {
          return;
        }
        if (!list.some((chat) => chat.id === activeChatId)) {
          setActiveChatId(null);
          setSelectedAssistantMessageId(null);
          setHighlightPayload(emptyHighlightPayload);
          return;
        }
        const detail = await fetchChat(caseId, activeChatId);
        if (!active) {
          return;
        }
        setChatMessagesById((previous) => {
          const currentMessages = previous[detail.id] ?? [];
          if (areChatMessagesEqual(currentMessages, detail.messages)) {
            return previous;
          }
          return {
            ...previous,
            [detail.id]: detail.messages
          };
        });
        applyHighlightFromChatMessages(detail.messages);
      } catch (err) {
        if (active) {
          setChatError(err instanceof Error ? err.message : "Unable to refresh chats.");
        }
      }
    };

    void pollChats();
    timer = window.setInterval(() => {
      void pollChats();
    }, 2500);

    return () => {
      active = false;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [activeChatId, applyHighlightFromChatMessages, caseId]);

  useEffect(() => {
    let active = true;
    let inFlight = false;
    const loadMeta = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const clusters = await fetchTagClusters(caseId);
        if (active) {
          setTagClusters(clusters);
        }
      } catch (err) {
        console.error("Failed to load tag clusters", err);
      } finally {
        inFlight = false;
      }
    };
    void loadMeta();
    const timer = window.setInterval(() => {
      void loadMeta();
    }, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [caseId]);

  const resetUploadDialogState = useCallback(() => {
    setUploadItems([]);
    setUploadPreparing(false);
  }, []);

  const updateUploadItem = useCallback(
    (clientId: string, updater: (item: UploadQueueItem) => UploadQueueItem) => {
      setUploadItems((prev) =>
        prev.map((item) => (item.clientId === clientId ? updater(item) : item))
      );
    },
    []
  );

  const prepareUploadFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) {
        resetUploadDialogState();
        return;
      }

      const initialItems: UploadQueueItem[] = files.map((file, index) => ({
        clientId: buildUploadRowId(file, index),
        file,
        contentHashSha256: null,
        hashReady: false,
        hashError: null,
        selectedConfidence: null,
        ingestProfile: "balanced_fast_intel",
        processingMode: "multimodal",
        advancedOpen: false,
        advanced: {},
        estimatedPreflight: estimateUploadComplexity(file),
        duplicateMatches: [],
        duplicateChoice: null,
        uploadState: "idle",
        uploadError: null,
        notes: ""
      }));

      setUploadItems(initialItems);
      setUploadPreparing(true);
      setError(null);

      try {
        const hashRows = await Promise.all(
          initialItems.map(async (item) => {
            try {
              return {
                clientId: item.clientId,
                contentHashSha256: await hashFileSha256(item.file),
                hashError: null
              };
            } catch (err) {
              return {
                clientId: item.clientId,
                contentHashSha256: null,
                hashError: err instanceof Error ? err.message : "Unable to hash file."
              };
            }
          })
        );

        const hashRowsById = new Map(
          hashRows.map((row) => [
            row.clientId,
            {
              contentHashSha256: row.contentHashSha256,
              hashError: row.hashError
            }
          ])
        );

        const duplicatePayload = hashRows
          .filter((row) => row.contentHashSha256)
          .map((row) => {
            const item = initialItems.find((candidate) => candidate.clientId === row.clientId);
            return {
              client_id: row.clientId,
              original_filename: item?.file.name,
              size_bytes: item?.file.size,
              content_hash_sha256: row.contentHashSha256 as string
            };
          });

        let duplicateRowsById = new Map<string, DocumentDuplicateCheckResult>();
        if (duplicatePayload.length > 0) {
          const results = await checkDocumentDuplicates(caseId, { files: duplicatePayload });
          duplicateRowsById = new Map(results.map((row) => [row.client_id, row]));
        }

        setUploadItems((prev) =>
          prev.map((item) => {
            const hashRow = hashRowsById.get(item.clientId);
            const duplicateRow = duplicateRowsById.get(item.clientId);
            const matches = duplicateRow?.matches ?? [];
            return {
              ...item,
              contentHashSha256: hashRow?.contentHashSha256 ?? null,
              hashReady: Boolean(hashRow?.contentHashSha256) && !hashRow?.hashError,
              hashError: hashRow?.hashError ?? null,
              duplicateMatches: matches,
              duplicateChoice:
                matches.length > 0 ? null : item.duplicateChoice,
              uploadState: "idle",
              uploadError: null
            };
          })
        );
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Unable to check for duplicate evidence in this case.";
        setError(message);
        setUploadItems((prev) =>
          prev.map((item) => ({
            ...item,
            hashReady: false,
            hashError: item.hashError ?? "Duplicate check failed."
          }))
        );
      } finally {
        setUploadPreparing(false);
      }
    },
    [caseId, resetUploadDialogState]
  );

  const handleUploadFileSelection = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";
      await prepareUploadFiles(files);
    },
    [prepareUploadFiles]
  );

  const handleUpload = async () => {
    if (uploadItems.length === 0) {
      setError("Select one or more files before uploading.");
      return;
    }
    const missingConfidence = uploadItems.find((item) => !item.selectedConfidence);
    if (missingConfidence) {
      setError(`Select a confidence grade for "${missingConfidence.file.name}".`);
      return;
    }
    const hashBlocked = uploadItems.find(
      (item) => !item.hashReady || Boolean(item.hashError) || !item.contentHashSha256
    );
    if (hashBlocked) {
      setError(`Hashing must complete successfully for "${hashBlocked.file.name}" before upload.`);
      return;
    }
    const unresolvedDuplicate = uploadItems.find(
      (item) => item.duplicateMatches.length > 0 && item.duplicateChoice === null
    );
    if (unresolvedDuplicate) {
      setError(
        `Choose "Upload as new evidence" for "${unresolvedDuplicate.file.name}" or remove it from the batch.`
      );
      return;
    }
    try {
      setUploading(true);
      setError(null);
      let uploadedCount = 0;
      let failedCount = 0;

      for (const item of uploadItems) {
        if (!item.selectedConfidence || !item.contentHashSha256) {
          failedCount += 1;
          setUploadItems((prev) =>
            prev.map((row) =>
              row.clientId === item.clientId
                ? {
                    ...row,
                    uploadState: "failed",
                    uploadError: "Missing confidence or content hash."
                  }
                : row
            )
          );
          continue;
        }

        setUploadItems((prev) =>
          prev.map((row) =>
            row.clientId === item.clientId
              ? {
                  ...row,
                  uploadState: "uploading",
                  uploadError: null
                }
              : row
          )
        );

        try {
          await uploadDocument(caseId, {
            file: item.file,
            confidence_source_reliability: item.selectedConfidence.source,
            confidence_information_validity: item.selectedConfidence.validity,
            ingest_profile: item.ingestProfile,
            processing_mode: item.processingMode,
            advanced_overrides: item.advanced,
            content_hash_sha256: item.contentHashSha256,
            allow_duplicate: item.duplicateChoice === "upload_new",
            notes: item.notes.trim() || undefined
          });
          uploadedCount += 1;
          setUploadItems((prev) =>
            prev.map((row) =>
              row.clientId === item.clientId
                ? {
                    ...row,
                    uploadState: "uploaded",
                    uploadError: null
                  }
                : row
            )
          );
        } catch (err) {
          failedCount += 1;
          setUploadItems((prev) =>
            prev.map((row) =>
              row.clientId === item.clientId
                ? {
                    ...row,
                    uploadState: "failed",
                    uploadError: err instanceof Error ? err.message : "Unable to upload evidence."
                  }
                : row
            )
          );
        }
      }

      await refresh();
      await refreshGraph();

      if (failedCount === 0) {
        setUploadOpen(false);
        resetUploadDialogState();
      } else {
        setError(`${failedCount} file(s) failed. Uploaded ${uploadedCount}.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to upload evidence.");
    } finally {
      setUploading(false);
    }
  };

  const handleUploadDialogOpenChange = (open: boolean) => {
    setUploadOpen(open);
    if (!open && !uploading && !uploadPreparing) {
      resetUploadDialogState();
    }
  };

  const handleCaseDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    if (!Array.from(event.dataTransfer.types).includes("Files")) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setCaseDropActive(true);
  }, []);

  const handleCaseDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }
    setCaseDropActive(false);
  }, []);

  const handleCaseDrop = useCallback(
    async (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setCaseDropActive(false);
      const files = Array.from(event.dataTransfer.files ?? []).filter((file) => file.size > 0);
      if (files.length === 0) {
        return;
      }
      setUploadOpen(true);
      await prepareUploadFiles(files);
    },
    [prepareUploadFiles]
  );

  const runDocumentSearch = useCallback(async () => {
    const query = documentSearchQuery.trim();
    if (!query) {
      setDocumentSearchResults([]);
      setDocumentSearchError(null);
      return;
    }
    try {
      setDocumentSearchLoading(true);
      setDocumentSearchError(null);
      const results = await apiSearchDocuments(caseId, {
        q: query,
        source: documentSearchSource,
        limit: 25
      });
      setDocumentSearchResults(results);
    } catch (err) {
      setDocumentSearchResults([]);
      setDocumentSearchError(err instanceof Error ? err.message : "Unable to search evidence.");
    } finally {
      setDocumentSearchLoading(false);
    }
  }, [caseId, documentSearchQuery, documentSearchSource]);

  const openDocumentSearchPreview = useCallback(
    async (result: DocumentSearchResult) => {
      const doc = documentsById.get(result.document_id);
      if (!doc) {
        setDocumentSearchError("The matching evidence file is unavailable in this case workspace.");
        return;
      }
      setDocumentPreviewTarget(result);
      setDocumentPreview(null);
      setDocumentPreviewError(null);
      setDocumentPreviewLoading(true);
      try {
        const preview = await fetchDocumentSearchPreview(caseId, result.document_id, {
          q: documentSearchQuery,
          source_kind: result.source_kind,
          segment_key: result.segment_key
        });
        setDocumentPreview(preview);
      } catch (err) {
        setDocumentPreviewError(
          err instanceof Error ? err.message : "Unable to load highlighted preview."
        );
      } finally {
        setDocumentPreviewLoading(false);
      }
    },
    [caseId, documentSearchQuery, documentsById]
  );

  const handleDocumentView = (doc: DocumentSummary) => {
    const url = buildDocumentDownloadUrl(caseId, doc.id);
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const openEvidenceDocument = useCallback(
    (evidence: GraphEvidence) => {
      const doc = resolveEvidenceDocument(evidence);
      if (!doc) {
        setGraphError("The linked evidence file is unavailable in this case workspace.");
        return;
      }
      const url = buildDocumentDownloadUrl(caseId, doc.id);
      window.open(url, "_blank", "noopener,noreferrer");
    },
    [caseId, resolveEvidenceDocument]
  );

  const drawAttentionToDetailsPane = useCallback((scrollIntoView: boolean) => {
    if (detailsPulseTimeoutRef.current !== null) {
      window.clearTimeout(detailsPulseTimeoutRef.current);
    }
    setDetailsPanePulse(false);
    window.setTimeout(() => setDetailsPanePulse(true), 0);
    detailsPulseTimeoutRef.current = window.setTimeout(() => {
      setDetailsPanePulse(false);
      detailsPulseTimeoutRef.current = null;
    }, 1800);
    if (scrollIntoView) {
      window.setTimeout(() => {
        detailsPaneRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    }
  }, []);

  const handleNodeSelect = useCallback(
    (node: GraphViewPayload["nodes"][number]) => {
      detailsRequestRef.current += 1;
      setSidePaneTab("details");
      setSelectedNodeId(node.id);
      setSelectedEdge(null);
      setMultiSelectedNodeIds(new Set());
      setMultiSelectedEdgeKeys(new Set());
    },
    []
  );

  const handleEdgeSelect = useCallback(
    (edge: GraphEdge) => {
      detailsRequestRef.current += 1;
      setSidePaneTab("details");
      setSelectedNodeId(null);
      setSelectedEdge({
        id: edge.id,
        src_id: edge.src_id,
        tgt_id: edge.tgt_id
      });
      setMultiSelectedNodeIds(new Set());
      setMultiSelectedEdgeKeys(new Set());
    },
    []
  );

  const handleAnalysisGraphInspect = useCallback(
    (target: { kind: "node"; node: GraphNode } | { kind: "edge"; edge: GraphEdge }) => {
      setWorkspaceMode("workbench");
      setSidePaneTab("details");
      if (target.kind === "node") {
        handleNodeSelect(target.node);
      } else {
        handleEdgeSelect(target.edge);
      }
      drawAttentionToDetailsPane(true);
    },
    [drawAttentionToDetailsPane, handleEdgeSelect, handleNodeSelect]
  );

  const handleCanvasBackgroundSelect = useCallback(() => {
    setSelectedNodeId(null);
    setSelectedEdge(null);
    setLegendSelection(null);
    setMultiSelectedNodeIds(new Set());
    setMultiSelectedEdgeKeys(new Set());
  }, []);

  const handleNodeMultiSelect = useCallback((node: GraphNode) => {
    setMultiSelectedNodeIds((previous) => {
      const next = new Set(previous);
      if (next.has(node.id)) {
        next.delete(node.id);
      } else {
        next.add(node.id);
      }
      return next;
    });
  }, []);

  const handleEdgeMultiSelect = useCallback((edge: GraphEdge) => {
    const key = `${edge.src_id}||${edge.tgt_id}`;
    setMultiSelectedEdgeKeys((previous) => {
      const next = new Set(previous);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  const handleMultiSelectClear = useCallback(() => {
    setMultiSelectedNodeIds(new Set());
    setMultiSelectedEdgeKeys(new Set());
  }, []);

  const multiSelectEntityLabels = useMemo(() => {
    if (multiSelectedNodeIds.size === 0 && multiSelectedEdgeKeys.size === 0) {
      return null;
    }
    const entityNames: string[] = [];
    for (const nodeId of multiSelectedNodeIds) {
      const node = graphData.nodes.find((n: GraphNode) => n.id === nodeId);
      entityNames.push(node?.label ?? nodeId);
    }
    const parts: string[] = [];
    if (entityNames.length > 0) {
      parts.push(`${entityNames.length} entit${entityNames.length === 1 ? "y" : "ies"}`);
    }
    if (multiSelectedEdgeKeys.size > 0) {
      parts.push(`${multiSelectedEdgeKeys.size} relationship${multiSelectedEdgeKeys.size === 1 ? "" : "s"}`);
    }
    return parts.join(" and ");
  }, [multiSelectedNodeIds, multiSelectedEdgeKeys, graphData.nodes]);

  const applySearchResult = useCallback(
    (result: ActorSearchResult) => {
      setGraphSearchQuery(result.name);
      setGraphSearchResults([]);
      setGraphPaneTab("focused");
      const existing = graphData.nodes.find(
        (node) => node.id === result.id || node.label === result.name
      );
      if (existing) {
        handleNodeSelect(existing);
        setNeighborDepth(1);
      } else {
        setSelectedNodeId(result.id);
        setNeighborDepth(1);
        setGraphLimit((prev) => Math.max(prev, graphLimit + 1));
      }
    },
    [graphData.nodes, handleNodeSelect, graphLimit]
  );

  useEffect(() => {
    if (!hasHighlights) {
      highlightAutoloadAttemptsRef.current.clear();
    }
  }, [hasHighlights]);

  useEffect(() => {
    if (!hasHighlights || graphLoading) {
      return;
    }
    const missingNodeIds = highlightPayload.highlight_entities.filter((id) => !graphNodeIdSet.has(id));
    const missingRelationship = highlightPayload.highlight_relationships.find((relationship) => {
      if (relationship.edge_id && graphEdgeIdSet.has(relationship.edge_id)) {
        return false;
      }
      return !graphEdgePairSet.has(pairKey(relationship.src_id, relationship.tgt_id));
    });
    if (missingNodeIds.length === 0 && !missingRelationship) {
      return;
    }

    const signature = JSON.stringify({
      entities: [...highlightPayload.highlight_entities].sort(),
      relationships: [...highlightPayload.highlight_relationships]
        .map((item) => `${item.edge_id ?? "none"}|${item.src_id}|${item.tgt_id}`)
        .sort(),
    });
    const attempts = highlightAutoloadAttemptsRef.current.get(signature) ?? 0;
    if (attempts >= MAX_HIGHLIGHT_AUTOLOAD_ATTEMPTS) {
      return;
    }
    highlightAutoloadAttemptsRef.current.set(signature, attempts + 1);

    const focusCandidate = missingNodeIds[0] ?? missingRelationship?.src_id ?? null;
    if (focusCandidate) {
      setSelectedNodeId(focusCandidate);
    }
    setGraphLimit((previous) => Math.min(previous + GRAPH_LIMIT_INCREMENT, MAX_AUTOLOAD_LIMIT));
  }, [
    graphEdgeIdSet,
    graphEdgePairSet,
    graphLoading,
    graphNodeIdSet,
    hasHighlights,
    highlightPayload.highlight_entities,
    highlightPayload.highlight_relationships,
  ]);

  const handleDocumentDownload = async (doc: DocumentSummary) => {
    try {
      setBusyDocumentId(doc.id);
      setError(null);
      await downloadDocument(caseId, doc.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to download evidence.");
    } finally {
      setBusyDocumentId(null);
    }
  };

  const handleDocumentReingest = async (doc: DocumentSummary) => {
    try {
      setBusyDocumentId(doc.id);
      setError(null);
      const job = jobByDocumentId.get(doc.id);
      const lastConfig = job?.effective_config as Record<string, unknown> | null | undefined;
      const profile = (lastConfig?.ingest_profile as IngestionProfile) || "balanced_fast_intel";
      const mode = (lastConfig?.processing_mode as ProcessingMode) || "multimodal";
      const advancedOverrides = (lastConfig?.advanced_overrides as IngestionAdvancedOverrides) || undefined;
      await reingestDocument(caseId, doc.id, profile, mode, advancedOverrides);
      await refresh();
      await refreshGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to re-ingest evidence.");
    } finally {
      setBusyDocumentId(null);
    }
  };

  const handleEditReingest = async () => {
    if (!editReingestTarget) return;
    try {
      setEditReingestBusy(true);
      setError(null);
      const confidenceCode = `${editReingestSourceReliability}${editReingestInfoValidity}`;
      await updateDocument(caseId, editReingestTarget.id, {
        notes: editReingestNotes || null,
        confidence_source_reliability: editReingestSourceReliability,
        confidence_information_validity: editReingestInfoValidity,
      });
      let advancedOverrides: IngestionAdvancedOverrides | undefined;
      if (editReingestAdvancedOverrides.trim()) {
        try {
          advancedOverrides = JSON.parse(editReingestAdvancedOverrides.trim());
        } catch {
          setError("Invalid JSON in advanced overrides");
          setEditReingestBusy(false);
          return;
        }
      }
      await reingestDocument(
        caseId,
        editReingestTarget.id,
        editReingestProfile,
        editReingestMode,
        advancedOverrides,
        editReingestNotes || null,
        editReingestSourceReliability,
        editReingestInfoValidity,
      );
      setEditReingestTarget(null);
      await refresh();
      await refreshGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to re-ingest evidence.");
    } finally {
      setEditReingestBusy(false);
    }
  };

  const handleDocumentDelete = async (doc: DocumentSummary) => {
    if (!window.confirm(`Delete "${doc.original_filename}"? This removes the raw file and metadata.`)) {
      return;
    }
    try {
      setBusyDocumentId(doc.id);
      setError(null);
      await deleteDocument(caseId, doc.id);
      await refresh();
      await refreshGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete evidence.");
    } finally {
      setBusyDocumentId(null);
    }
  };

  const selectedEntityRelationshipCount = useMemo(() => {
    if (!selectedNodeId) {
      return 0;
    }
    return relationshipsPayload.relationships.filter((relationship) => {
      const actorId = relationship.actor_id?.trim() || relationship.actor;
      const targetId = relationship.target_id?.trim() || relationship.target;
      return actorId === selectedNodeId || targetId === selectedNodeId;
    }).length;
  }, [relationshipsPayload.relationships, selectedNodeId]);

  const selectedEntityDescription =
    selectedEntityDetails?.description?.trim() ||
    selectedNode?.summary?.trim() ||
    (selectedNode
      ? `${selectedNode.label} appears in ${selectedEntityRelationshipCount} relationship${selectedEntityRelationshipCount === 1 ? "" : "s"} in this case.`
      : "No entity selected.");

  const relationshipType =
    selectedRelationshipDetails?.relation_type ||
    selectedRelationship?.action ||
    selectedGraphEdge?.relation_type ||
    "ASSOCIATED_WITH";

  const selectedRelationshipDescription =
    selectedRelationshipDetails?.description?.trim() ||
    selectedRelationship?.description?.trim() ||
    (selectedRelationship
      ? `${selectedRelationship.actor} ${selectedRelationship.action.replace(/_/g, " ").toLowerCase()} ${selectedRelationship.target}.`
      : selectedGraphEdge?.label?.trim() || "No relationship description available.");

  const activeEvidence = useMemo<GraphEvidence[]>(() => {
    if (selectedEntityDetails?.evidence?.length) {
      return dedupeEvidence(selectedEntityDetails.evidence);
    }
    if (selectedRelationshipDetails?.evidence?.length) {
      return dedupeEvidence(selectedRelationshipDetails.evidence);
    }
    if (selectedRelationship?.evidence?.length) {
      return dedupeEvidence(selectedRelationship.evidence);
    }
    if (selectedGraphEdge) {
      if (selectedGraphEdge.evidence?.length) {
        return dedupeEvidence(selectedGraphEdge.evidence);
      }
      const matching = relationshipsPayload.relationships.filter(
        (relationship) =>
          relationship.id === selectedGraphEdge.id ||
          ((relationship.actor_id?.trim() || relationship.actor) === selectedGraphEdge.src_id &&
            (relationship.target_id?.trim() || relationship.target) === selectedGraphEdge.tgt_id)
      );
      return dedupeEvidence(matching.flatMap((relationship) => relationship.evidence ?? []));
    }
    if (selectedNodeId) {
      if (selectedNode?.evidence?.length) {
        return dedupeEvidence(selectedNode.evidence);
      }
      const matching = relationshipsPayload.relationships.filter((relationship) => {
        const actorId = relationship.actor_id?.trim() || relationship.actor;
        const targetId = relationship.target_id?.trim() || relationship.target;
        return actorId === selectedNodeId || targetId === selectedNodeId;
      });
      return dedupeEvidence(matching.flatMap((relationship) => relationship.evidence ?? []));
    }
    return [];
  }, [
    relationshipsPayload.relationships,
    selectedEntityDetails,
    selectedGraphEdge,
    selectedNode,
    selectedNodeId,
    selectedRelationship,
    selectedRelationshipDetails
  ]);

  const uploadReadyCount = uploadItems.filter(
    (item) =>
      item.selectedConfidence &&
      item.hashReady &&
      !item.hashError &&
      item.contentHashSha256 &&
      (item.duplicateMatches.length === 0 || item.duplicateChoice === "upload_new")
  ).length;
  const unresolvedUploadDuplicates = uploadItems.filter(
    (item) => item.duplicateMatches.length > 0 && item.duplicateChoice === null
  ).length;
  const uploadCanSubmit =
    uploadItems.length > 0 &&
    !uploadPreparing &&
    !uploading &&
    uploadReadyCount === uploadItems.length;

  const renderUploadItem = (item: UploadQueueItem, index: number) => {
    const selectedCode = item.selectedConfidence
      ? `${item.selectedConfidence.source}${item.selectedConfidence.validity}`
      : "--";

    return (
      <div key={item.clientId} className="rounded-xl border bg-muted/20 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground">
              {index + 1}. {item.file.name}
            </p>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span>{formatBytes(item.file.size)}</span>
              {item.estimatedPreflight ? (
                <span>
                  complexity: {item.estimatedPreflight.complexity_class.replace("_", " ")}
                </span>
              ) : null}
              {item.estimatedPreflight ? (
                <span>ETA: ~{Math.ceil(item.estimatedPreflight.eta_seconds / 60)} min</span>
              ) : null}
              {item.contentHashSha256 ? (
                <span className="font-mono">hash: {item.contentHashSha256.slice(0, 12)}...</span>
              ) : null}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {item.hashError ? (
              <Badge className="bg-destructive text-destructive-foreground">Hash failed</Badge>
            ) : item.hashReady ? (
              <Badge variant="outline">Hash ready</Badge>
            ) : (
              <Badge variant="muted">Preparing</Badge>
            )}
            {item.uploadState === "uploaded" ? (
              <Badge className="bg-emerald-600 text-white">Queued</Badge>
            ) : null}
            {item.uploadState === "failed" ? (
              <Badge className="bg-destructive text-destructive-foreground">Failed</Badge>
            ) : null}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() =>
                setUploadItems((prev) => prev.filter((row) => row.clientId !== item.clientId))
              }
              disabled={uploading}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {item.hashError ? (
          <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {item.hashError}
          </div>
        ) : null}

        {item.duplicateMatches.length > 0 ? (
          <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
            <p className="text-sm font-medium text-foreground">Duplicate detected in this case</p>
            <p className="mt-1 text-xs text-muted-foreground">
              This file matches existing evidence by SHA-256. You can upload it as a new evidence
              file, remove it from this batch, or cancel the dialog.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button
                type="button"
                variant={item.duplicateChoice === "upload_new" ? "default" : "outline"}
                size="sm"
                onClick={() =>
                  updateUploadItem(item.clientId, (current) => ({
                    ...current,
                    duplicateChoice: "upload_new"
                  }))
                }
              >
                Upload as new evidence
              </Button>
            </div>
            <div className="mt-3 grid gap-2">
              {item.duplicateMatches.map((match) => (
                <div
                  key={match.id}
                  className="rounded-md border bg-background/80 px-3 py-2"
                >
                  <p className="text-xs font-medium text-foreground">{match.original_filename}</p>
                  <p className="text-[11px] text-muted-foreground">
                    confidence {match.confidence_code} | model {formatModelLabel(match.ingest_model_name)} | added{" "}
                    {formatDate(match.created_at)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="mt-4 grid gap-4">
          <div className="grid gap-2">
            <label className="text-sm font-medium" htmlFor={`notes-${item.clientId}`}>
              Evidence notes
            </label>
            <textarea
              id={`notes-${item.clientId}`}
              value={item.notes}
              onChange={(event) =>
                updateUploadItem(item.clientId, (current) => ({
                  ...current,
                  notes: event.target.value
                }))
              }
              disabled={uploading}
              placeholder="Analyst context, provenance caveats, or handling notes"
              className="min-h-[84px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>

          <div className="grid gap-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Expert settings</label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  updateUploadItem(item.clientId, (current) => ({
                    ...current,
                    advancedOpen: !current.advancedOpen
                  }))
                }
              >
                {item.advancedOpen ? "Hide expert settings" : "Show expert settings"}
              </Button>
            </div>
            {item.advancedOpen ? (
              <div className="grid gap-3 rounded-xl border bg-background/70 p-4">
                <div className="grid gap-1">
                  <label className="text-xs uppercase text-muted-foreground">Extraction depth</label>
                  <Select
                    value={item.ingestProfile}
                    onValueChange={(value) =>
                      updateUploadItem(item.clientId, (current) => ({
                        ...current,
                        ingestProfile: value as IngestionProfile
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select depth" />
                    </SelectTrigger>
                    <SelectContent>
                      {ingestProfileOptions.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {ingestProfileOptions.find((option) => option.value === item.ingestProfile)?.hint}
                  </p>
                </div>
                <div className="grid gap-1">
                  <label className="text-xs uppercase text-muted-foreground">Visual analysis</label>
                  <Select
                    value={item.processingMode}
                    onValueChange={(value) =>
                      updateUploadItem(item.clientId, (current) => ({
                        ...current,
                        processingMode: value as ProcessingMode
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Visual analysis mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="multimodal">Full (images + text)</SelectItem>
                      <SelectItem value="text_first">Skip (text only)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {item.processingMode === "text_first"
                      ? "Skips visual analysis — faster for CSVs, text documents, and plain data."
                      : "Enables image and visual analysis for screenshots, photos, and scanned documents."}
                  </p>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="grid gap-1">
                    <label className="text-xs uppercase text-muted-foreground">OCR mode</label>
                    <Select
                      value={item.advanced.ocr_mode ?? "off"}
                      onValueChange={(value) =>
                        updateUploadItem(item.clientId, (current) => ({
                          ...current,
                          advanced: {
                            ...current.advanced,
                            ocr_mode: value as "off" | "auto" | "force"
                          }
                        }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="OCR mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="off">Off</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="force">Force</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-1">
                    <label className="text-xs uppercase text-muted-foreground">Queue priority</label>
                    <Select
                      value={item.advanced.queue_priority ?? "normal"}
                      onValueChange={(value) =>
                        updateUploadItem(item.clientId, (current) => ({
                          ...current,
                          advanced: {
                            ...current.advanced,
                            queue_priority: value as "low" | "normal" | "high"
                          }
                        }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Queue priority" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="high">High</SelectItem>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          <div className="grid gap-3">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium">Confidence (UNODC 4x4)</p>
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Selected: {selectedCode}
              </span>
            </div>
            <div className="grid gap-2 rounded-xl border bg-background/70 p-4">
              <div className="grid grid-cols-[5.75rem_repeat(4,minmax(0,1fr))] gap-2">
                <div />
                {confidenceColumns.map((column) => (
                  <div
                    key={column}
                    className="rounded-md border border-muted/70 bg-background/70 px-2 py-1 text-center"
                  >
                    <p className="text-xs font-semibold text-foreground">{column}</p>
                    <p className="text-[9px] leading-snug text-muted-foreground break-words whitespace-normal">
                      {confidenceValidityCompact[column]}
                    </p>
                  </div>
                ))}
                {confidenceRows.map((row) => (
                  <div key={`${item.clientId}-${row}`} className="contents">
                    <div className="rounded-md border border-muted/70 bg-background/70 px-2 py-1">
                      <p className="text-xs font-semibold text-foreground">{row}</p>
                      <p className="text-[9px] leading-snug text-muted-foreground break-words whitespace-normal">
                        {confidenceSourceCompact[row]}
                      </p>
                    </div>
                    {confidenceColumns.map((column) => {
                      const selected =
                        item.selectedConfidence?.source === row &&
                        item.selectedConfidence?.validity === column;
                      return (
                        <Button
                          key={`${item.clientId}-${row}${column}`}
                          type="button"
                          variant="outline"
                          className={`h-auto min-h-[5.75rem] w-full min-w-0 flex-col items-start justify-start border px-2 py-2 text-left whitespace-normal ${confidenceCellToneClass(
                            row,
                            column,
                            selected
                          )}`}
                          onClick={() =>
                            updateUploadItem(item.clientId, (current) => ({
                              ...current,
                              selectedConfidence: { source: row, validity: column }
                            }))
                          }
                        >
                          <span className="w-full text-xs font-semibold">
                            {row}
                            {column}
                          </span>
                          <span className="mt-1 w-full text-[9px] leading-snug opacity-90 break-words whitespace-normal">
                            {confidenceSourceCompact[row]}
                          </span>
                          <span className="w-full text-[9px] leading-snug opacity-90 break-words whitespace-normal">
                            {confidenceValidityCompact[column]}
                          </span>
                        </Button>
                      );
                      })}
                    </div>
                ))}
              </div>
            </div>
            <div className="grid gap-1 rounded-lg border border-dashed border-muted/60 bg-background/60 p-3 text-xs text-muted-foreground">
              <p className="font-medium text-foreground">How to read the matrix</p>
              <p>Rows represent source reliability (A/B/C/X). Columns represent information validity (1/2/3/4).</p>
              <p>Each cell displays the confidence code plus its reliability and validity meaning.</p>
            </div>
          </div>

          {item.uploadError ? (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {item.uploadError}
            </div>
          ) : null}
        </div>
      </div>
    );
  };

  const selectedGraphLabel = selectedNodeId
    ? selectedNode?.label ?? selectedNodeId
    : selectedRelationship
      ? `${selectedRelationship.actor} -> ${selectedRelationship.target}`
      : selectedEdge
        ? `${selectedEdge.src_id} -> ${selectedEdge.tgt_id}`
        : "No selection";

  const renderDetailsPane = () => {
    if (!selectedNodeId && !selectedEdge) {
      if (!caseSummary) {
        return (
          <div className="rounded-lg border border-dashed border-muted/60 p-4 text-sm text-muted-foreground">
            <div className="flex items-center justify-between gap-3">
              <span>Summary is not available yet.</span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleRefreshCaseSummary}
                disabled={caseSummaryRefreshing}
              >
                {caseSummaryRefreshing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCcw className="h-4 w-4" />
                )}
                Generate
              </Button>
            </div>
            {caseSummaryRefreshError ? (
              <p className="mt-2 text-xs text-destructive">{caseSummaryRefreshError}</p>
            ) : null}
          </div>
        );
      }

      const entityExportDisabled = (caseSummary.entity_count ?? 0) <= 0;
      const relationExportDisabled = (caseSummary.relationship_count ?? 0) <= 0;

      return (
        <>
          <div className="rounded-lg border bg-background/70 p-4">
            <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Case summary</p>
            <p className="mt-2 text-base font-semibold text-foreground">{caseSummary.case_name}</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    className="flex min-h-[38px] items-center justify-between gap-2 rounded-md border px-2 py-1 text-left transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
                    disabled={caseGraphExporting !== null}
                  >
                    <span>
                      <span className="block text-muted-foreground">Export</span>
                      <span className="block font-semibold text-foreground">CSV</span>
                    </span>
                    {caseGraphExporting ? (
                      <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
                    ) : (
                      <Download className="h-4 w-4 shrink-0 text-muted-foreground" />
                    )}
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    disabled={entityExportDisabled || caseGraphExporting !== null}
                    onClick={() => void handleExportGraphCsv("entities")}
                  >
                    Entities CSV
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={relationExportDisabled || caseGraphExporting !== null}
                    onClick={() => void handleExportGraphCsv("relations")}
                  >
                    Relations CSV
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <button
                type="button"
                className="flex min-h-[38px] items-center justify-between gap-2 rounded-md border px-2 py-1 text-left transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
                onClick={handleRefreshCaseSummary}
                disabled={caseSummaryRefreshing}
              >
                <span>
                  <span className="block text-muted-foreground">Summary</span>
                  <span className="block font-semibold text-foreground">Regenerate</span>
                </span>
                {caseSummaryRefreshing ? (
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
                ) : (
                  <RefreshCcw className="h-4 w-4 shrink-0 text-muted-foreground" />
                )}
              </button>
            </div>
            {caseSummaryRefreshError ? (
              <p className="mt-2 text-xs text-destructive">{caseSummaryRefreshError}</p>
            ) : null}
            {caseGraphExportError ? (
              <p className="mt-2 text-xs text-destructive">{caseGraphExportError}</p>
            ) : null}
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border px-2 py-1">
                <p className="text-muted-foreground">Entities</p>
                <p className="font-semibold text-foreground">{caseSummary.entity_count ?? 0}</p>
              </div>
              <div className="rounded-md border px-2 py-1">
                <p className="text-muted-foreground">Relationships</p>
                <p className="font-semibold text-foreground">{caseSummary.relationship_count ?? 0}</p>
              </div>
              <div className="rounded-md border px-2 py-1">
                <p className="text-muted-foreground">Evidence</p>
                <p className="font-semibold text-foreground">
                  {caseSummary.evidence_count ?? caseSummary.document_count ?? 0}
                </p>
              </div>
              <div className="rounded-md border px-2 py-1">
                <p className="text-muted-foreground">Completed</p>
                <p className="font-semibold text-foreground">{caseSummary.completed_document_count ?? 0}</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border bg-background/70 p-4">
            <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Intelligence POV</p>
            <p className="mt-2 text-sm text-muted-foreground">
              {caseSummary.intelligence_summary?.trim() || "No intelligence summary available yet."}
            </p>
          </div>

          <div className="rounded-lg border bg-background/70 p-4">
            <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Investigation POV</p>
            <p className="mt-2 text-sm text-muted-foreground">
              {caseSummary.investigation_summary?.trim() || "No investigation summary available yet."}
            </p>
          </div>

          <div className="rounded-lg border bg-background/70 p-4">
            <div className="flex items-center justify-between gap-2">
              <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">5W1H</p>
              <Badge variant="outline">Auto</Badge>
            </div>
            <div className="mt-2 grid gap-1 text-sm">
              {(
                [
                  ["Who", caseSummary.five_w_one_h?.who],
                  ["What", caseSummary.five_w_one_h?.what],
                  ["When", caseSummary.five_w_one_h?.when],
                  ["Where", caseSummary.five_w_one_h?.where],
                  ["Why", caseSummary.five_w_one_h?.why],
                  ["How", caseSummary.five_w_one_h?.how]
                ] as const
              ).map(([label, value]) => (
                <p key={label} className="text-muted-foreground">
                  <span className="font-medium text-foreground">{label}:</span> {value?.trim() ? value : "Unknown"}
                </p>
              ))}
            </div>
            {caseSummary.unknowns?.length ? (
              <div className="mt-3 border-t border-border/60 pt-2">
                <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Unknowns</p>
                <div className="mt-1 space-y-1">
                  {caseSummary.unknowns.map((item, index) => (
                    <p key={`${item}-${index}`} className="text-xs text-muted-foreground">
                      - {item}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}
            {caseSummary.last_refreshed_at ? (
              <p className="mt-3 text-[11px] text-muted-foreground">
                Last refreshed: {formatDateTime(caseSummary.last_refreshed_at)}
              </p>
            ) : null}
          </div>
        </>
      );
    }

    return (
      <>
        <div className="rounded-lg border bg-background/70 p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                {selectedNodeId ? "Entity details" : "Relationship details"}
              </p>
              <p className="mt-2 break-words text-base font-semibold text-foreground">{selectedGraphLabel}</p>
            </div>
            <Button variant="outline" size="sm" onClick={handleCanvasBackgroundSelect}>
              Clear
            </Button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline" className="font-medium">
              {selectedNodeId ? selectedNode?.entity_type ?? "Other" : relationshipType}
            </Badge>
            <span className="font-mono">
              {selectedNodeId ? selectedNode?.id ?? selectedNodeId : selectedGraphEdge?.id ?? selectedEdge?.id}
            </span>
          </div>
          <p className="mt-3 text-sm text-muted-foreground">
            {selectedNodeId ? selectedEntityDescription : selectedRelationshipDescription}
          </p>
          {selectionDetailsLoading ? (
            <p className="mt-2 text-xs text-muted-foreground">Loading grounded details...</p>
          ) : null}
          {selectionDetailsError ? (
            <p className="mt-2 text-xs text-destructive">{selectionDetailsError}</p>
          ) : null}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-3">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Supporting evidence</p>
          <Badge variant="muted">{activeEvidence.length}</Badge>
        </div>

        {activeEvidence.length === 0 ? (
          <div className="rounded-lg border bg-background/70 p-4 text-sm text-muted-foreground">
            No supporting evidence found.
          </div>
        ) : (
          <div className="space-y-3 overflow-x-hidden pr-1">
            {activeEvidence.map((evidence) => {
              const linkedDocument = resolveEvidenceDocument(evidence);
              const displayName = linkedDocument?.original_filename ?? pathBasename(evidence.file_path);
              return (
                <div
                  key={[
                    evidence.file_path,
                    evidence.reference_id,
                    evidence.document_id ?? "none",
                    evidence.source_id ?? "none",
                    evidence.confidence_code ?? "none"
                  ].join("-")}
                  className="overflow-hidden rounded-lg border bg-background/70 p-4"
                >
                  <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                    <p className="min-w-0 break-words text-sm font-medium text-foreground">{displayName}</p>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">
                        {evidence.confidence_code ?? "N/A"}
                      </Badge>
                      <Button
                        size="icon"
                        variant="outline"
                        className="h-7 w-7"
                        aria-label="Preview evidence"
                        title="Preview evidence"
                        disabled={!linkedDocument}
                        onClick={() => {
                          setEvidenceModalTarget(evidence);
                          setEvidenceModalOpen(true);
                        }}
                      >
                        <Eye className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <p className="mt-1 break-words text-[11px] font-mono text-muted-foreground">
                    ref: {evidence.reference_id}
                  </p>
                  {evidence.source_id ? (
                    <p className="mt-1 break-words text-[11px] font-mono text-muted-foreground">
                      source: {evidence.source_id}
                    </p>
                  ) : null}
                  {evidence.snippet ? (
                    <p className="mt-2 break-words text-xs text-muted-foreground">{evidence.snippet}</p>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
      </>
    );
  };

  const renderChatsPane = () => (
    <>
      {chatError ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {chatError}
        </div>
      ) : null}
      {multiSelectEntityLabels ? (
        <div className="flex items-center justify-between gap-2 rounded-md border bg-card px-3 py-2 text-xs">
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">{multiSelectEntityLabels}</span> selected in graph for context
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={handleMultiSelectClear}
          >
            Clear
          </Button>
        </div>
      ) : null}
      <div className="grid gap-2.5 md:flex md:min-h-0 md:flex-1 md:flex-col">
        <div
          ref={chatThreadRef}
          onScroll={handleChatThreadScroll}
          className="min-h-[10rem] overflow-y-auto rounded-xl border bg-background/40 p-2.5 lg:min-h-0 lg:flex-1"
        >
          {!activeChat ? (
            <div className="flex h-full min-h-[12rem] flex-col items-center justify-center gap-3 text-center">
              <p className="max-w-[24rem] text-sm text-muted-foreground">
                Select a chat from the picker or create a new one to begin.
              </p>
              <Button variant="outline" size="sm" onClick={() => void handleCreateChat()}>
                <Plus className="h-4 w-4" />
                Start new chat
              </Button>
              <Button variant="outline" size="sm" onClick={() => setChatListOpen(true)}>
                <ListChecks className="h-4 w-4" />
                Open chats
              </Button>
            </div>
          ) : activeChatMessages.length === 0 ? (
            <p className="text-sm text-muted-foreground">No messages in this chat yet.</p>
          ) : (
            <div className="space-y-3">
              {activeChatMessages.map((message) => {
                const isAssistant = message.role === "assistant";
                const metadata = extractAssistantMetadata(message);
                const messageHighlight = extractAssistantHighlight(message);
                const modelName = extractAssistantModelName(message);
                const references = messageHighlight?.references ?? [];
                const chunks = messageHighlight?.supporting_chunks ?? [];
                const referencesExpanded = Boolean(chatExpandedReferencesByMessageId[message.id]);
                const modeValue =
                  metadata && typeof metadata.mode === "string" ? metadata.mode.toUpperCase() : null;
                return (
                  <div
                    key={message.id}
                    className={[
                      "rounded-lg border px-3 py-2 text-sm",
                      isAssistant ? "bg-card/80" : "ml-8 border-primary/20 bg-primary/5",
                      isAssistant && selectedAssistantMessageId === message.id ? "border-primary/60 ring-1 ring-primary/25" : "",
                      isAssistant && messageHighlight ? "cursor-pointer transition-colors hover:border-primary/40" : ""
                    ].join(" ")}
                    onClick={() => {
                      if (!isAssistant || !messageHighlight) {
                        return;
                      }
                      applyAssistantMessageSelection(message);
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
                        {message.role}
                      </span>
                      <span className="text-[11px] text-muted-foreground">{formatDateTime(message.created_at)}</span>
                    </div>
                    {isAssistant ? (
                      <MarkdownText content={message.content} className="mt-2" />
                    ) : (
                      <p className="mt-1 whitespace-pre-wrap break-words text-sm text-foreground">
                        {message.content}
                      </p>
                    )}
                    {isAssistant ? (
                      <div className="mt-2 space-y-2">
                        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                          {modeValue ? (
                            <Badge variant="outline" className="font-mono">
                              {modeValue}
                            </Badge>
                          ) : null}
                          {modelName ? (
                            <Badge variant="outline" className="max-w-[16rem] truncate font-mono" title={modelName}>
                              {modelName}
                            </Badge>
                          ) : null}
                          <span>{references.length} references</span>
                          {references.length > 0 ? (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-5 px-2 text-[11px]"
                              onClick={(event) => {
                                event.stopPropagation();
                                setChatExpandedReferencesByMessageId((previous) => ({
                                  ...previous,
                                  [message.id]: !previous[message.id]
                                }));
                              }}
                            >
                              {referencesExpanded ? "Hide references" : "Show references"}
                            </Button>
                          ) : null}
                        </div>
                        {references.length > 0 && referencesExpanded ? (
                          <div className="space-y-2">
                            {references.map((reference) => {
                              const snippet =
                                chunks.find(
                                  (chunk) =>
                                    chunk.reference_id === reference.reference_id &&
                                    chunk.file_path === reference.file_path
                                )?.snippet ?? null;
                              return (
                                <div
                                  key={`${reference.reference_id}-${reference.file_path}`}
                                  className="min-w-0 overflow-hidden rounded-md border bg-background/70 px-2 py-1.5 text-xs"
                                >
                                  <div className="flex min-w-0 items-center justify-between gap-2">
                                    <span className="truncate font-medium">
                                      {pathBasename(reference.file_path)}
                                    </span>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 shrink-0 px-2 text-xs"
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        void openChatReferencePreview(reference, snippet);
                                      }}
                                    >
                                      Preview
                                    </Button>
                                  </div>
                                  <p className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
                                    {reference.reference_id}
                                  </p>
                                  {snippet ? (
                                    <p className="mt-1 break-words text-[11px] text-muted-foreground">
                                      {snippet}
                                    </p>
                                  ) : null}
                                </div>
                              );
                            })}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="grid gap-2 border-t border-border/60 pt-2">
          <textarea
            ref={chatComposerRef}
            value={chatComposer}
            onChange={(event) => setChatComposer(event.target.value)}
            onInput={() => adjustChatComposerHeight()}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void handleSendChatMessage();
              }
            }}
            disabled={chatSending || chatCreating || chatSendBlockedReason !== null}
            placeholder={chatSendBlockedReason ?? "Ask about this case..."}
            className="min-h-[80px] max-h-[144px] w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          />
          {chatSendBlockedReason ? (
            <p className="text-xs text-muted-foreground">{chatSendBlockedReason}</p>
          ) : null}
          <div className="flex items-center justify-end gap-2">
            {developerMode ? (
              <Select value={chatMode} onValueChange={(value) => setChatMode(value as ChatMode)}>
                <SelectTrigger className="w-[130px]">
                  <SelectValue placeholder="Mode" />
                </SelectTrigger>
                <SelectContent>
                  {chatModeOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}
            <Button
              onClick={() => void handleSendChatMessage()}
              disabled={!canSendChatMessage}
            >
              {chatSending ? "Sending..." : "Send"}
            </Button>
          </div>
        </div>
      </div>
    </>
  );

  const renderChatPickerDialog = () => (
    <Dialog open={chatListOpen} onOpenChange={setChatListOpen}>
      <DialogContent className="grid max-h-[80vh] max-w-xl grid-rows-[auto_minmax(0,1fr)] overflow-hidden p-0">
        <DialogHeader className="border-b border-border/60 px-6 py-4">
          <DialogTitle>Chats</DialogTitle>
          <DialogDescription>Switch between case chats without keeping the list pinned open.</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 overflow-y-auto overflow-x-hidden px-4 pb-4">
          {chatLoading && chats.length === 0 ? (
            <div className="py-3 text-xs text-muted-foreground">Loading chats...</div>
          ) : chats.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-3">
              <Button variant="outline" size="sm" onClick={() => void handleCreateChat()}>
                <Plus className="h-4 w-4" />
                Start new chat
              </Button>
              <div className="text-xs text-muted-foreground">No chats yet. Create one to start asking questions.</div>
            </div>
          ) : (
            <div className="grid gap-2 py-3">
              {chats.map((chat) => (
                <div key={chat.id} className="flex min-w-0 items-center gap-2 overflow-hidden">
                  <Button
                    variant={activeChatId === chat.id ? "default" : "ghost"}
                    size="sm"
                    className="h-auto min-w-0 flex-1 justify-start overflow-hidden px-3 py-2"
                    onClick={() => void handleSelectChat(chat.id)}
                    disabled={chatDeletingId === chat.id}
                  >
                    <div className="flex min-w-0 flex-1 items-center gap-3 overflow-hidden">
                      <span className="truncate text-left text-xs" title={chat.title}>
                        {chat.title}
                      </span>
                      <span className="shrink-0 text-[10px] text-muted-foreground">{formatDate(chat.updated_at)}</span>
                    </div>
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label={`Delete chat ${chat.title}`}
                    title="Delete chat"
                    disabled={Boolean(chatDeletingId) || chatSending || chatCreating}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDeleteChat(chat);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );

  const renderGraphFiltersMenu = () => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <SlidersHorizontal className="h-4 w-4" />
          
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[22rem] max-h-[min(75vh,28rem)] overflow-y-auto p-3">
        <div className="grid gap-4">
          <div className="grid gap-2">
            <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
              Date window (edge timestamp)
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              <Input
                type="date"
                value={graphDateFrom}
                onChange={(event) => {
                  setGraphPaneTab("focused");
                  setGraphDateFrom(event.target.value);
                }}
              />
              <Input
                type="date"
                value={graphDateTo}
                onChange={(event) => {
                  setGraphPaneTab("focused");
                  setGraphDateTo(event.target.value);
                }}
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {(graphDateFrom || graphDateTo) && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setGraphPaneTab("focused");
                    setGraphDateFrom("");
                    setGraphDateTo("");
                  }}
                >
                  Clear dates
                </Button>
              )}
              <Button
                variant={graphIncludeUndated ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setGraphPaneTab("focused");
                  setGraphIncludeUndated((previous) => !previous);
                }}
              >
                {graphIncludeUndated ? "Include undated" : "Exclude undated"}
              </Button>
            </div>
          </div>

          <div className="grid gap-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
                Focus hops
              </p>
              {neighborDepth > 0 && selectedNodeId ? (
                <span className="text-[11px] text-muted-foreground">
                  {neighborDepth}-hop on <span className="font-mono">{selectedNodeId}</span>
                </span>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-2">
              {[0, 1, 2].map((depth) => (
                <Button
                  key={`depth-${depth}`}
                  variant={neighborDepth === depth ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    if (depth > 0 && !selectedNodeId) {
                      return;
                    }
                    if (depth > 0) {
                      setGraphPaneTab("focused");
                    }
                    setNeighborDepth(depth as 0 | 1 | 2);
                  }}
                  disabled={depth > 0 && !selectedNodeId}
                >
                  {depth === 0 ? "Focus off" : `${depth}-hop`}
                </Button>
              ))}
            </div>
          </div>

          <div className="grid gap-2 border-t border-border/60 pt-3">
            <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
              Display
            </p>
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">Show relationship labels</span>
              <Button
                variant={graphShowLabels ? "default" : "outline"}
                size="sm"
                onClick={() => setGraphShowLabels((prev) => !prev)}
              >
                {graphShowLabels ? "On" : "Off"}
              </Button>
            </div>
          </div>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  return (
    <div
      className="relative min-h-screen px-3 py-4 sm:px-4 lg:px-5"
      onDragOver={handleCaseDragOver}
      onDragLeave={handleCaseDragLeave}
      onDrop={(event) => {
        void handleCaseDrop(event);
      }}
    >
      {caseDropActive ? (
        <div className="pointer-events-none fixed inset-3 z-40 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary bg-background/85 text-sm font-medium text-foreground shadow-soft backdrop-blur">
          Drop files to open the upload form
        </div>
      ) : null}
      <div className="mx-auto flex w-full max-w-[96rem] flex-col gap-4">
        <header className="shrink-0 rounded-2xl border bg-card/90 px-4 py-3 shadow-glow backdrop-blur lg:px-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                window.location.href = "/";
              }}
            >
              <ArrowLeft className="h-4 w-4" />
              Cases
            </Button>
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Case workspace</p>
              <div className="mt-0.5 flex items-center gap-2">
                <h1 className="text-xl font-semibold">{caseDetail?.name ?? "Loading case"}</h1>
                <span className="group relative z-20 inline-flex">
                  <button
                    type="button"
                    className="inline-flex h-7 w-7 items-center justify-center rounded-full border bg-background text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    aria-label="Case description"
                    title={caseDetail?.description ?? "Upload evidence and capture confidence before ingestion."}
                  >
                    <Info className="h-4 w-4" />
                  </button>
                  <span className="pointer-events-none absolute left-1/2 top-full z-[80] mt-2 hidden w-[min(28rem,calc(100vw-3rem))] -translate-x-1/2 rounded-lg border bg-popover px-3 py-2 text-xs leading-relaxed text-popover-foreground shadow-soft group-hover:block group-focus-within:block">
                    {caseDetail?.description ?? "Upload evidence and capture confidence before ingestion."}
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
            <Button variant="outline" size="sm" onClick={() => setRenameOpen(true)} disabled={!caseDetail || renamingCase}>
              <Pencil className="h-4 w-4" />
              Rename
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings2 className="h-4 w-4" />
              Settings
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                window.location.href = `/cases/${caseId}/jobs`;
              }}
            >
              <ListChecks className="h-4 w-4" />
              Jobs
            </Button>
            <Dialog open={uploadOpen} onOpenChange={handleUploadDialogOpenChange}>
              <DialogTrigger asChild>
                <Button className="gap-2">
                  <Upload className="h-4 w-4" />
                  Upload Evidence
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-4xl max-h-[92vh] overflow-hidden flex flex-col">
                <DialogHeader className="shrink-0">
                  <DialogTitle>Upload evidence</DialogTitle>
                  <DialogDescription>
                    Upload multiple evidence files, review case-local duplicates by hash, and assign per-file
                    ingestion settings.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-6 overflow-y-auto pr-1">
                  <div className="grid gap-2">
                    <label className="text-sm font-medium" htmlFor="evidence-file">
                      Evidence files
                    </label>
                    <Input
                      id="evidence-file"
                      type="file"
                      multiple
                      onChange={handleUploadFileSelection}
                      disabled={uploading || uploadPreparing}
                    />
                    <p className="text-xs text-muted-foreground">
                      Hashes are computed in the browser, then checked against existing evidence in this
                      case.
                    </p>
                  </div>

                  {uploadPreparing ? (
                    <div className="rounded-lg border border-dashed border-muted/60 bg-background/60 p-3 text-sm text-muted-foreground">
                      Preparing files, computing hashes, and checking for duplicates...
                    </div>
                  ) : null}

                  {uploadItems.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-muted/60 p-8 text-center text-sm text-muted-foreground">
                      No files selected yet.
                    </div>
                  ) : (
                    <div className="grid gap-4">{uploadItems.map(renderUploadItem)}</div>
                  )}
                </div>
                <DialogFooter className="shrink-0">
                  <div className="mr-auto text-xs text-muted-foreground">
                    {uploadItems.length > 0 ? (
                      <span>
                        Ready: {uploadReadyCount}/{uploadItems.length}
                        {unresolvedUploadDuplicates > 0
                          ? ` | unresolved duplicates: ${unresolvedUploadDuplicates}`
                          : ""}
                      </span>
                    ) : null}
                  </div>
                  <Button variant="outline" onClick={() => setUploadOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleUpload} disabled={!uploadCanSubmit}>
                    {uploadPreparing
                      ? "Preparing..."
                      : uploading
                        ? "Uploading..."
                        : `Upload ${uploadItems.length || ""}`.trim()}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
            <div className="ml-auto flex items-center gap-1 rounded-xl border bg-background/70 p-1">
              <Button
                variant={workspaceMode === "workbench" ? "default" : "ghost"}
                size="sm"
                onClick={() => {
                  setWorkspaceMode("workbench");
                }}
              >
                <Share2 className="h-4 w-4" />
                Graph
              </Button>
              <Button
                variant={workspaceMode === "analysis" ? "default" : "ghost"}
                size="sm"
                onClick={() => setWorkspaceMode("analysis")}
              >
                <Network className="h-4 w-4" />
                Analysis
              </Button>
            </div>
          </div>
          </div>
        </header>

        {error ? (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        ) : null}

        {loading ? (
          <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-muted/60 p-10 text-center text-sm text-muted-foreground">
            Loading case workspace...
          </div>
        ) : (
          <>
            {workspaceMode === "analysis" ? (
              <AnalysisWorkspace
                caseId={caseId}
                hasEvidence={hasCaseEvidence}
                onInspectGraphElement={handleAnalysisGraphInspect}
              />
            ) : (
            <div className="grid min-h-0 gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(20rem,1fr)]">
              <section className="hidden">
                <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Details</p>
                <div className="mt-4 space-y-4 overflow-x-hidden md:min-h-0 md:flex-1 md:overflow-y-auto md:pr-1">
                  {!selectedNodeId && !selectedEdge ? (
                    caseSummary ? (
                      <>
                        <div className="rounded-lg border bg-background/70 p-4">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                            Case summary
                          </p>
                          <p className="mt-2 text-base font-semibold text-foreground">{caseSummary.case_name}</p>
                          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                            <div className="rounded-md border px-2 py-1">
                              <p className="text-muted-foreground">Entities</p>
                              <p className="font-semibold text-foreground">{caseSummary.entity_count ?? 0}</p>
                            </div>
                            <div className="rounded-md border px-2 py-1">
                              <p className="text-muted-foreground">Relationships</p>
                              <p className="font-semibold text-foreground">{caseSummary.relationship_count ?? 0}</p>
                            </div>
                            <div className="rounded-md border px-2 py-1">
                              <p className="text-muted-foreground">Evidences</p>
                              <p className="font-semibold text-foreground">
                                {caseSummary.evidence_count ?? caseSummary.document_count ?? 0}
                              </p>
                            </div>
                            <div className="rounded-md border px-2 py-1">
                              <p className="text-muted-foreground">Completed</p>
                              <p className="font-semibold text-foreground">
                                {caseSummary.completed_document_count ?? 0}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="rounded-lg border bg-background/70 p-4">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                            Intelligence POV
                          </p>
                          <p className="mt-2 text-sm text-muted-foreground">
                            {caseSummary.intelligence_summary?.trim() ||
                              "No intelligence summary available yet."}
                          </p>
                        </div>

                        <div className="rounded-lg border bg-background/70 p-4">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                            Investigation POV
                          </p>
                          <p className="mt-2 text-sm text-muted-foreground">
                            {caseSummary.investigation_summary?.trim() ||
                              "No investigation summary available yet."}
                          </p>
                        </div>

                        <div className="rounded-lg border bg-background/70 p-4">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">5W1H</p>
                            <Badge variant="outline">Auto</Badge>
                          </div>
                          <div className="mt-2 grid gap-1 text-sm">
                            {(
                              [
                                ["Who", caseSummary.five_w_one_h?.who],
                                ["What", caseSummary.five_w_one_h?.what],
                                ["When", caseSummary.five_w_one_h?.when],
                                ["Where", caseSummary.five_w_one_h?.where],
                                ["Why", caseSummary.five_w_one_h?.why],
                                ["How", caseSummary.five_w_one_h?.how],
                              ] as const
                            ).map(([label, value]) => (
                              <p key={label} className="text-muted-foreground">
                                <span className="font-medium text-foreground">{label}:</span>{" "}
                                {value?.trim() ? value : "Unknown"}
                              </p>
                            ))}
                          </div>
                          {caseSummary.unknowns?.length ? (
                            <div className="mt-3 border-t border-border/60 pt-2">
                              <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Unknowns</p>
                              <div className="mt-1 space-y-1">
                                {caseSummary.unknowns.map((item, index) => (
                                  <p key={`${item}-${index}`} className="text-xs text-muted-foreground">
                                    - {item}
                                  </p>
                                ))}
                              </div>
                            </div>
                          ) : null}
                          {caseSummary.last_refreshed_at ? (
                            <p className="mt-3 text-[11px] text-muted-foreground">
                              Last refreshed: {formatDateTime(caseSummary.last_refreshed_at)}
                            </p>
                          ) : null}
                        </div>
                      </>
                    ) : (
                      <div className="rounded-lg border border-dashed border-muted/60 p-4 text-sm text-muted-foreground">
                        Summary is not available yet. Upload evidence and wait for ingestion to complete.
                      </div>
                    )
                  ) : (
                    <>
                      <div className="rounded-lg border bg-background/70 p-4">
                        <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                          {selectedNodeId ? "Entity details" : "Relationship details"}
                        </p>
                        <p className="mt-2 text-base font-semibold text-foreground">
                          {selectedNodeId
                            ? selectedNode?.label ?? selectedNodeId
                            : `${selectedRelationship?.actor ?? selectedEdge?.src_id ?? selectedGraphEdge?.src_id} -> ${selectedRelationship?.target ?? selectedEdge?.tgt_id ?? selectedGraphEdge?.tgt_id}`}
                        </p>
                        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                          <Badge variant="outline" className="font-medium">
                            {selectedNodeId
                              ? selectedNode?.entity_type ?? "Other"
                              : relationshipType}
                          </Badge>
                          <span className="font-mono">
                            {selectedNodeId
                              ? selectedNode?.id ?? selectedNodeId
                              : selectedGraphEdge?.id ?? selectedEdge?.id}
                          </span>
                        </div>
                        <p className="mt-3 text-sm text-muted-foreground">
                          {selectedNodeId ? selectedEntityDescription : selectedRelationshipDescription}
                        </p>
                        {selectionDetailsLoading ? (
                          <p className="mt-2 text-xs text-muted-foreground">Loading grounded details...</p>
                        ) : null}
                        {selectionDetailsError ? (
                          <p className="mt-2 text-xs text-destructive">{selectionDetailsError}</p>
                        ) : null}
                      </div>

                      <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                          Supporting evidence
                        </p>
                        <Badge variant="muted">{activeEvidence.length}</Badge>
                      </div>

                      {activeEvidence.length === 0 ? (
                        <div className="rounded-lg border bg-background/70 p-4 text-sm text-muted-foreground">
                          No supporting evidence found.
                        </div>
                      ) : (
                        <div className="space-y-3 overflow-x-hidden pr-1">
                          {activeEvidence.map((evidence) => {
                            const linkedDocument = resolveEvidenceDocument(evidence);
                            const displayName =
                              linkedDocument?.original_filename ?? pathBasename(evidence.file_path);
                            return (
                              <div
                                key={[
                                  evidence.file_path,
                                  evidence.reference_id,
                                  evidence.document_id ?? "none",
                                  evidence.source_id ?? "none",
                                  evidence.confidence_code ?? "none"
                                ].join("-")}
                                className="overflow-hidden rounded-lg border bg-background/70 p-4"
                              >
                                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                                  <p className="min-w-0 break-all text-sm font-medium text-foreground">
                                    {displayName}
                                  </p>
                                  <div className="flex items-center gap-2">
                                    <Badge variant="outline" className="font-mono">
                                      {evidence.confidence_code ?? "N/A"}
                                    </Badge>
                                    <Button
                                      size="icon"
                                      variant="outline"
                                      className="h-7 w-7"
                                      aria-label="Open document"
                                      title="Open document"
                                      disabled={!linkedDocument}
                                      onClick={() => openEvidenceDocument(evidence)}
                                    >
                                      <Eye className="h-3.5 w-3.5" />
                                    </Button>
                                  </div>
                                </div>
                                <p className="mt-1 break-all text-[11px] font-mono text-muted-foreground">
                                  ref: {evidence.reference_id}
                                </p>
                                {evidence.source_id ? (
                                  <p className="mt-1 break-all text-[11px] font-mono text-muted-foreground">
                                    source: {evidence.source_id}
                                  </p>
                                ) : null}
                                {evidence.snippet ? (
                                  <p className="mt-2 break-words text-xs text-muted-foreground">{evidence.snippet}</p>
                                ) : null}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </section>
              <section className="order-1 rounded-2xl border bg-card/90 p-3 shadow-soft md:p-4 lg:flex lg:h-[calc(100dvh-7rem)] lg:min-h-[34rem] lg:flex-col lg:overflow-hidden">
                {isCaseFresh ? (
                  <div className="flex h-full flex-col items-center justify-center gap-6 p-8 text-center">
                    <Loader2 className="h-10 w-10 animate-spin text-primary" />
                    <div>
                      <p className="text-base font-semibold text-foreground">Ingestion in progress</p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        Processing {documents.filter(d => d.ingestion_status === "complete" || d.ingestion_status === "completed_with_warnings").length} of {documents.length} evidence files
                      </p>
                    </div>
                    <div className="max-h-64 w-full max-w-md overflow-y-auto rounded-lg border bg-background/60 p-3 text-left">
                      <div className="space-y-1.5">
                        {sortedDocuments.map((doc) => {
                          const job = jobByDocumentId.get(doc.id);
                          const status = job?.status ?? doc.ingestion_status;
                          const processing = isProcessingStatus(status);
                          return (
                            <div key={doc.id} className="flex items-center gap-2 text-xs">
                              {processing ? (
                                <ProcessingIndicator label={`Processing ${doc.original_filename}`} />
                              ) : status === "complete" ? (
                                <span className="text-emerald-600 font-bold">&#10003;</span>
                              ) : status === "completed_with_warnings" ? (
                                <span className="text-amber-600 font-bold">&#9888;</span>
                              ) : status === "failed" ? (
                                <span className="text-destructive font-bold">&#10007;</span>
                              ) : (
                                <span className="text-muted-foreground">&#9679;</span>
                              )}
                              <span className="truncate">{doc.original_filename}</span>
                              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground uppercase">
                                {status}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                {ingestionBanner ? (
                  <div className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200 my-2">
                    <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                    Ingesting {ingestionBanner.completedCount} of {ingestionBanner.total} evidence files...
                  </div>
                ) : null}
                <div className="flex flex-wrap items-center gap-2 border-b border-border/60 pb-2">

                  <Button
                    variant={graphPaneTab === "original" ? "default" : "outline"}
                    size="sm"
                    onClick={resetGraphScope}
                  >
                    Original
                  </Button>
                  <Button
                    variant={graphPaneTab === "focused" ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      if (hasFocusedState) {
                        setGraphPaneTab("focused");
                      }
                    }}
                    disabled={!hasFocusedState}
                  >
                    Focused
                  </Button>
                  <span className="text-muted-foreground/40 select-none">|</span>
                  {([
                    { key: "person",       label: "Persons",       color: "#d97706" },
                    { key: "organization", label: "Organizations", color: "#2563eb" },
                    { key: "object",       label: "Objects",       color: "#6d28d9" },
                    { key: "location",     label: "Locations",     color: "#16a34a" },
                    { key: "event",        label: "Events",        color: "#dc2626" },
                  ] as const).map((poleType) => {
                    const isActive = poleTypeFilters.has(poleType.key);
                    return (
                      <Button
                        key={poleType.key}
                        variant={isActive ? "default" : "outline"}
                        size="sm"
                        onClick={() =>
                          setPoleTypeFilters((prev) => {
                            const next = new Set(prev);
                            if (next.has(poleType.key)) {
                              next.delete(poleType.key);
                            } else {
                              next.add(poleType.key);
                            }
                            return next;
                          })
                        }
                      >
                        {poleType.label}
                      </Button>
                    );
                  })}
                  <span className="text-muted-foreground/40 select-none">|</span>
                  <div className="relative">
                    <Input
                      value={graphSearchQuery}
                      onChange={(event) => handleGraphSearchChange(event.target.value)}
                      placeholder="Search entities..."
                      className="h-8 w-44 text-xs"
                    />
                    {graphSearchResults.length > 0 ? (
                      <div className="absolute left-0 top-full z-20 mt-1 max-h-60 w-80 overflow-y-auto rounded-md border bg-popover p-1 shadow-lg">
                        {graphSearchResults.map((result) => (
                          <button
                            key={result.id}
                            type="button"
                            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-xs hover:bg-accent hover:text-accent-foreground"
                            onClick={() => applySearchResult(result)}
                          >
                            <span className="truncate font-medium">{result.name}</span>
                            <span className="shrink-0 text-[10px] uppercase text-muted-foreground">
                              {result.entity_type}
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                  {renderGraphFiltersMenu()}
                  {graphData.truncated ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setGraphLimit((previous) => Math.min(previous + GRAPH_LIMIT_INCREMENT, MAX_AUTOLOAD_LIMIT))
                      }
                    >
                      Load more
                    </Button>
                  ) : null}
                  <Button variant="outline" size="sm" onClick={() => void refreshGraph()}>
                    <RefreshCcw className="h-4 w-4" />
                    
                  </Button>
           
                
                </div>
                {!graphControlsCollapsed ? (
                  <div className="mt-4 grid gap-3 rounded-xl border border-dashed border-muted/60 bg-background/40 p-3">
                  <div className="grid gap-2">
                    <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
                      Tag clusters
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {tagClusters.length === 0 ? (
                        <span className="text-xs text-muted-foreground">No clusters available.</span>
                      ) : (
                        tagClusters.map((cluster) => (
                          <Button
                            key={cluster.id}
                            variant={selectedClusters.includes(cluster.name) ? "default" : "outline"}
                            size="sm"
                            onClick={() =>
                              setSelectedClusters((prev) =>
                                prev.includes(cluster.name)
                                  ? prev.filter((value) => value !== cluster.name)
                                  : [...prev, cluster.name]
                              )
                            }
                          >
                            {cluster.name} ({cluster.tagCount})
                          </Button>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="grid gap-2">
                    <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
                      Relation type filters
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {graphRelationTypeOptions.length === 0 ? (
                        <span className="text-xs text-muted-foreground">No relation types available in this view.</span>
                      ) : (
                        graphRelationTypeOptions.map((relationType) => (
                          <Button
                            key={relationType}
                            variant={graphRelationTypeFilters.includes(relationType) ? "default" : "outline"}
                            size="sm"
                            onClick={() => toggleRelationTypeFilter(relationType)}
                          >
                            {relationType}
                          </Button>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="grid gap-2">
                    <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">
                      Expand neighbors
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant={neighborDepth === 0 ? "default" : "outline"}
                        size="sm"
                        onClick={() => setNeighborDepth(0)}
                      >
                        Off
                      </Button>
                      <Button
                        variant={neighborDepth === 1 ? "default" : "outline"}
                        size="sm"
                        onClick={() => {
                          if (!selectedNodeId) {
                            return;
                          }
                          setGraphPaneTab("focused");
                          setNeighborDepth(1);
                        }}
                        disabled={!selectedNodeId}
                      >
                        1-hop
                      </Button>
                      <Button
                        variant={neighborDepth === 2 ? "default" : "outline"}
                        size="sm"
                        onClick={() => {
                          if (!selectedNodeId) {
                            return;
                          }
                          setGraphPaneTab("focused");
                          setNeighborDepth(2);
                        }}
                        disabled={!selectedNodeId}
                      >
                        2-hop
                      </Button>
                    </div>
                    {neighborDepth > 0 && selectedNodeId ? (
                      <p className="text-xs text-muted-foreground">
                        Focus active on <span className="font-mono">{selectedNodeId}</span> at {neighborDepth}-hop.
                      </p>
                    ) : null}
                  </div>
                  </div>
                ) : null}
                {graphError ? (
                  <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                    {graphError}
                  </div>
                ) : null}
                <div className="mt-2 min-h-[30rem] sm:min-h-[34rem] md:min-h-0 md:flex-1">
                  {graphLoading ? (
                    <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-center text-sm text-muted-foreground">
                      Loading graph data...
                    </div>
                  ) : graphPaneTab === "focused" && hasHighlights && !hasFocusedGraphData ? (
                    <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-center text-sm text-muted-foreground">
                      No highlighted results yet for this chat.
                    </div>
                  ) : (
                    <GraphCanvas
                      className="mt-0 h-[30rem] min-h-[30rem] sm:h-[34rem] sm:min-h-[34rem] md:h-full md:min-h-0"
                      nodes={activeGraphData.nodes}
                      edges={activeGraphData.edges}
                      selectedNodeId={selectedNodeId}
                      selectedEdge={selectedEdge}
                    highlightedNodeIds={activeHighlightedNodeIds}
                    highlightedEdges={activeHighlightedEdges}
                    legendSelection={legendSelection}
                    showLabels={graphShowLabels}
                    multiSelectedNodeIds={multiSelectedNodeIds}
                    multiSelectedEdgeKeys={multiSelectedEdgeKeys}
                    onNodeSelect={handleNodeSelect}
                    onEdgeSelect={handleEdgeSelect}
                    onCanvasBackgroundSelect={handleCanvasBackgroundSelect}
                    onLegendSelect={handleLegendSelect}
                    onNodeMultiSelect={handleNodeMultiSelect}
                    onEdgeMultiSelect={handleEdgeMultiSelect}
                  />
                  )}
                </div>
                </>
                )}
              </section>
              <section
                ref={detailsPaneRef}
                className={[
                  "order-2 rounded-2xl border bg-card/90 p-3 shadow-soft transition-shadow md:p-4 lg:flex lg:h-[calc(100dvh-7rem)] lg:min-h-[34rem] lg:flex-col lg:overflow-hidden",
                  detailsPanePulse ? "rawabit-details-pulse" : ""
                ].join(" ").trim()}
              >
                <div className="flex items-center justify-between gap-2 border-b border-border/60 pb-2">
                  <div className="flex items-center gap-2">
                    <Button
                      variant={sidePaneTab === "details" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSidePaneTab("details")}
                    >
                      Details
                    </Button>
                    <Button
                      variant={sidePaneTab === "chats" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSidePaneTab("chats")}
                    >
                      Chats
                    </Button>
                  </div>
                  {sidePaneTab === "chats" ? (
                    <div className="ml-auto flex shrink-0 items-center gap-1.5">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 gap-2 px-3"
                        onClick={() => setChatListOpen(true)}
                        aria-label="Open chat list"
                        title="Open chat list"
                      >
                        <ListChecks className="h-4 w-4" />
                        History
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => void handleCreateChat()}
                        disabled={chatCreating || chatSending}
                        aria-label="New chat"
                        title="New chat"
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                  ) : null}
                </div>
                {sidePaneTab === "details" ? (
                  <div className="mt-3 space-y-4 overflow-x-hidden md:min-h-0 md:flex-1 md:overflow-y-auto md:pr-1">
                    {renderDetailsPane()}
                  </div>
                ) : (
                   <div className="mt-3 md:flex md:min-h-0 md:flex-1 md:flex-col">{renderChatsPane()}</div>
                )}
              </section>
            </div>
            )}

            {renderChatPickerDialog()}

            <Dialog
              open={Boolean(chatDeleteTarget)}
              onOpenChange={(open) => {
                if (!open) {
                  closeDeleteChatDialog();
                }
              }}
            >
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Delete chat</DialogTitle>
                  <DialogDescription>
                    Delete "{chatDeleteTarget?.title ?? "this chat"}" and all messages in it? This action cannot
                    be undone.
                  </DialogDescription>
                </DialogHeader>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={closeDeleteChatDialog}
                    disabled={Boolean(chatDeletingId)}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => void confirmDeleteChat()}
                    disabled={Boolean(chatDeletingId)}
                  >
                    {chatDeletingId ? "Deleting..." : "Delete chat"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog
              open={Boolean(documentPreviewTarget)}
              onOpenChange={(open) => {
                if (!open) {
                  setDocumentPreviewTarget(null);
                  setDocumentPreview(null);
                  setDocumentPreviewError(null);
                }
              }}
            >
              <DialogContent className="grid max-h-[82vh] max-w-4xl min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden">
                <DialogHeader className="min-w-0">
                  <DialogTitle className="truncate">
                    {documentPreview?.original_filename ??
                      documentPreviewTarget?.original_filename ??
                      "Evidence preview"}
                  </DialogTitle>
                  <DialogDescription>
                    Highlighted matches from the selected search result.
                  </DialogDescription>
                </DialogHeader>
                <div className="min-h-0 min-w-0 overflow-hidden rounded-xl border bg-background/80">
                  <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3 overflow-x-hidden">
                    {documentPreviewTarget ? (
                      <Badge variant={documentPreviewTarget.source_kind === "processed" ? "secondary" : "outline"}>
                        {documentPreviewTarget.source_kind}
                      </Badge>
                    ) : null}
                    {documentPreview?.confidence_code ?? documentPreviewTarget?.confidence_code ? (
                      <Badge variant="outline" className="font-mono shrink-0">
                        {documentPreview?.confidence_code ?? documentPreviewTarget?.confidence_code}
                      </Badge>
                    ) : null}
                    {documentPreviewTarget?.segment_key ? (
                      <span className="truncate font-mono text-[11px] text-muted-foreground min-w-0 max-w-full">
                        {documentPreviewTarget.segment_key}
                      </span>
                    ) : null}
                  </div>
                  <div className="max-h-[52vh] min-w-0 overflow-auto p-4">
                    {documentPreviewLoading ? (
                      <div className="grid gap-2">
                        <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
                        <div className="h-4 w-full animate-pulse rounded bg-muted" />
                        <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
                      </div>
                    ) : documentPreviewError ? (
                      <p className="text-sm text-destructive">{documentPreviewError}</p>
                    ) : documentPreview ? (
                      <div className="whitespace-pre-wrap break-words font-mono text-xs leading-6 text-foreground">
                        <HighlightedPreviewText
                          content={documentPreview.content}
                          ranges={documentPreview.match_ranges}
                        />
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No preview loaded.</p>
                    )}
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => {
                      const doc = documentPreviewTarget
                        ? documentsById.get(documentPreviewTarget.document_id)
                        : null;
                      if (doc) {
                        handleDocumentView(doc);
                      }
                    }}
                    disabled={!documentPreviewTarget || !documentsById.has(documentPreviewTarget.document_id)}
                  >
                    <Eye className="mr-2 h-4 w-4" />
                    Open raw file
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog
              open={evidenceModalOpen}
              onOpenChange={(open) => {
                if (!open) {
                  setEvidenceModalOpen(false);
                  setEvidenceModalTarget(null);
                }
              }}
            >
               <DialogContent className="grid max-h-[82vh] max-w-2xl min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden">
                <DialogHeader className="min-w-0">
                  <DialogTitle className="truncate">
                    {evidenceModalTarget
                      ? (() => {
                          const doc = resolveEvidenceDocument(evidenceModalTarget);
                          return doc?.original_filename ?? pathBasename(evidenceModalTarget.file_path);
                        })()
                      : "Evidence preview"}
                  </DialogTitle>
                  <DialogDescription>
                    Supporting evidence details for the selected entity or relationship.
                  </DialogDescription>
                </DialogHeader>
                <div className="min-h-0 min-w-0 overflow-hidden rounded-xl border bg-background/80">
                  <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3 overflow-x-hidden">
                    {evidenceModalTarget?.confidence_code ? (
                      <Badge variant="outline" className="font-mono shrink-0">
                        {evidenceModalTarget.confidence_code}
                      </Badge>
                    ) : null}
                    {evidenceModalTarget?.reference_id ? (
                      <span className="font-mono text-[11px] text-muted-foreground truncate min-w-0 max-w-full">
                        ref: {evidenceModalTarget.reference_id}
                      </span>
                    ) : null}
                    {evidenceModalTarget?.source_id ? (
                      <span className="font-mono text-[11px] text-muted-foreground truncate min-w-0 max-w-full">
                        source: {evidenceModalTarget.source_id}
                      </span>
                    ) : null}
                  </div>
                  <div className="max-h-[52vh] min-w-0 overflow-auto p-4">
                    {evidenceChunkLoading ? (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading full chunk content...
                      </div>
                    ) : evidenceChunkContent ? (
                      <div className="whitespace-pre-wrap break-words font-mono text-xs leading-6 text-foreground">
                        {evidenceChunkContent}
                      </div>
                    ) : evidenceModalTarget?.snippet ? (
                      <div className="whitespace-pre-wrap break-words font-mono text-xs leading-6 text-foreground">
                        {evidenceModalTarget.snippet}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No snippet available.</p>
                    )}
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (evidenceModalTarget) {
                        openEvidenceDocument(evidenceModalTarget);
                      }
                    }}
                    disabled={!evidenceModalTarget}
                  >
                    <Eye className="mr-2 h-4 w-4" />
                    Open raw file
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <RenameCaseDialog
              open={renameOpen}
              busy={renamingCase}
              initialName={caseDetail?.name}
              initialDescription={caseDetail?.description}
              onOpenChange={setRenameOpen}
              onSubmit={handleRenameCase}
            />

            <section className="rounded-2xl border bg-card/90 p-6 shadow-soft">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Documents</p>
                  <h2 className="mt-1 text-xl font-semibold">Evidence list</h2>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Re-ingest defaults to Standard multimodal. Use Deep for richer extraction.
                  </p>
                </div>
                <Badge variant="muted">{documents.length} files</Badge>
              </div>

              {documents.length === 0 ? (
                <div className="mt-4 rounded-xl border border-dashed border-muted/60 p-8 text-center text-sm text-muted-foreground">
                  No evidence uploaded yet. Use "Upload Evidence" to add documents.
                </div>
              ) : (
                <div className="mt-4">
                  <div className="mb-4 rounded-xl border border-muted/60 bg-background/80 p-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
                      <div className="relative min-w-0 flex-1">
                        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                        <Input
                          value={documentSearchQuery}
                          onChange={(event) => setDocumentSearchQuery(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              void runDocumentSearch();
                            }
                          }}
                          placeholder="Search raw and processed evidence text"
                          className="pl-9"
                        />
                      </div>
                      <Select
                        value={documentSearchSource}
                        onValueChange={(value) => setDocumentSearchSource(value as DocumentSearchSource)}
                      >
                        <SelectTrigger className="w-full lg:w-[150px]">
                          <SelectValue placeholder="Source" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All text</SelectItem>
                          <SelectItem value="raw">Raw</SelectItem>
                          <SelectItem value="processed">Processed</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        variant="outline"
                        onClick={() => void runDocumentSearch()}
                        disabled={documentSearchLoading || documentSearchQuery.trim().length === 0}
                      >
                        {documentSearchLoading ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Search className="mr-2 h-4 w-4" />
                        )}
                        Search
                      </Button>
                    </div>

                    {documentSearchError ? (
                      <p className="mt-3 text-sm text-destructive">{documentSearchError}</p>
                    ) : null}

                    {documentSearchResults.length > 0 ? (
                      <div className="mt-4 divide-y rounded-lg border">
                        {documentSearchResults.map((result) => {
                          const doc = documentsById.get(result.document_id);
                          return (
                            <div
                              key={`${result.document_id}-${result.source_kind}-${result.segment_key}`}
                              className="grid gap-3 p-3 md:grid-cols-[minmax(0,1fr)_auto]"
                            >
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="truncate text-sm font-medium">
                                    {result.original_filename}
                                  </span>
                                  <Badge variant={result.source_kind === "processed" ? "secondary" : "outline"}>
                                    {result.source_kind}
                                  </Badge>
                                  {result.confidence_code ? (
                                    <Badge variant="outline" className="font-mono">
                                      {result.confidence_code}
                                    </Badge>
                                  ) : null}
                                </div>
                                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                                  <HighlightedSnippet
                                    parts={result.snippet_parts}
                                    fallback={result.snippet || result.stored_file_path}
                                  />
                                </p>
                              </div>
                              <div className="flex items-center gap-2 md:justify-end">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => void openDocumentSearchPreview(result)}
                                  disabled={!doc}
                                >
                                  <Eye className="mr-2 h-4 w-4" />
                                  Preview
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => doc && void handleDocumentDownload(doc)}
                                  disabled={!doc || busyDocumentId === result.document_id}
                                >
                                  <Download className="mr-2 h-4 w-4" />
                                  Download
                                </Button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : documentSearchQuery.trim() && !documentSearchLoading ? (
                      <p className="mt-3 text-sm text-muted-foreground">No matching evidence text found.</p>
                    ) : null}
                  </div>

                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Filename</TableHead>
                        <TableHead>Uploaded</TableHead>
                        <TableHead>Model</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Duration</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedDocuments.map((doc) => {
                        const job = jobByDocumentId.get(doc.id);
                        const jobModelCandidate = job?.effective_config?.primary_ingest_model;
                        const jobModel = typeof jobModelCandidate === "string" ? jobModelCandidate : null;
                        const status = job?.status ?? doc.ingestion_status;
                        const error = job?.error ?? doc.ingestion_error;
                        const processing = isProcessingStatus(status);
                        return (
                          <TableRow key={doc.id}>
                            <TableCell className="font-medium">
                              <div className="flex items-center gap-2">
                                {processing ? (
                                  <ProcessingIndicator label={`Processing ${doc.original_filename}`} />
                                ) : null}
                                <span>{doc.original_filename}</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <span className="text-xs text-muted-foreground">{formatDateTime(doc.created_at)}</span>
                            </TableCell>
                            <TableCell>
                              <span className="text-xs text-muted-foreground">
                                {formatModelLabel(jobModel ?? doc.ingest_model_name)}
                              </span>
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline" className="font-mono">
                                {doc.confidence_code}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="grid gap-1">
                                <div className="flex items-center gap-1">
                                  {status === "completed_with_warnings" ? (
                                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600 shrink-0" />
                                  ) : null}
                                  <div title={status === "failed" && error ? `Error: ${error}` : status === "completed_with_warnings" ? "Completed with warnings — some extraction may have failed. Re-ingest if needed." : undefined}>
                                    {isProcessingStatus(status) && job?.current_stage
                                      ? <Badge variant="muted">{job.current_stage.replace(/_/g, " ")}</Badge>
                                      : statusBadge(status)}
                                  </div>
                                </div>
                                {typeof job?.progress === "number" ? (
                                  <span className="text-xs text-muted-foreground">{job.progress}%</span>
                                ) : null}
                                {job?.ingest_profile ? (
                                  <span className="text-xs text-muted-foreground">
                                    profile: {profileLabel(job.ingest_profile)}
                                  </span>
                                ) : null}
                                {job?.complexity_class ? (
                                  <span className="text-xs text-muted-foreground">
                                    complexity: {job.complexity_class.replace("_", " ")}
                                  </span>
                                ) : null}
                                {typeof job?.eta_seconds === "number" ? (
                                  <span className="text-xs text-muted-foreground">
                                    eta: ~{Math.ceil(job.eta_seconds / 60)} min
                                  </span>
                                ) : null}
                                {job?.queue_priority ? (
                                  <span className="text-xs text-muted-foreground">
                                    priority: {job.queue_priority}
                                  </span>
                                ) : null}
                                {job?.route_type ? (
                                  <span className="text-xs text-muted-foreground">
                                    route: {job.route_type.replace(/_/g, " ")}
                                  </span>
                                ) : null}
                                {typeof job?.effective_config?.entity_extract_max_gleaning === "number" ? (
                                  <span className="text-xs text-muted-foreground">
                                    gleaning: {job.effective_config.entity_extract_max_gleaning === 0 ? "off" : "on"}
                                  </span>
                                ) : null}
                                {status === "failed" && error ? (
                                  <span className="text-xs text-destructive" title={error}>
                                    {error}
                                  </span>
                                ) : null}
                              </div>
                            </TableCell>
                            <TableCell>
                              <span className="text-xs text-muted-foreground">
                                {formatDuration(job?.parse_duration_s, job?.insert_duration_s, job?.finalize_duration_s)}
                              </span>
                            </TableCell>
                            <TableCell className="text-right">
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuItem onClick={() => handleDocumentView(doc)}>
                                    <Eye className="mr-2 h-4 w-4" />
                                    View
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => handleDocumentDownload(doc)}
                                    disabled={busyDocumentId === doc.id}
                                  >
                                    <Download className="mr-2 h-4 w-4" />
                                    Download
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => handleDocumentReingest(doc)}
                                    disabled={busyDocumentId === doc.id}
                                  >
                                    <RefreshCcw className="mr-2 h-4 w-4" />
                                    Reingest (same settings)
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => {
                                      setEditReingestTarget(doc);
                                      setEditReingestSourceReliability(doc.confidence_source_reliability ?? "A");
                                      setEditReingestInfoValidity(doc.confidence_information_validity ?? "1");
                                      setEditReingestNotes(doc.notes ?? "");
                                      const job = jobByDocumentId.get(doc.id);
                                      const config = job?.effective_config as Record<string, unknown> | null;
                                      setEditReingestProfile((config?.ingest_profile as IngestionProfile) || "balanced_fast_intel");
                                      setEditReingestMode((config?.processing_mode as ProcessingMode) || "multimodal");
                                      setEditReingestAdvancedOverrides(config?.advanced_overrides ? JSON.stringify(config.advanced_overrides, null, 2) : "");
                                    }}
                                    disabled={busyDocumentId === doc.id}
                                  >
                                    <Pencil className="mr-2 h-4 w-4" />
                                    Edit and reingest
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => { setJobLogsDocumentTarget(doc.id); }}
                                  >
                                    <FileText className="mr-2 h-4 w-4" />
                                    View Logs
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    className="text-destructive"
                                    onClick={() => handleDocumentDelete(doc)}
                                    disabled={busyDocumentId === doc.id}
                                  >
                                    <Trash2 className="mr-2 h-4 w-4" />
                                    Delete
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              )}
            </section>

            <Dialog open={Boolean(editReingestTarget)} onOpenChange={(open) => { if (!open) setEditReingestTarget(null); }}>
              <DialogContent className="max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Edit & Re-ingest</DialogTitle>
                  <DialogDescription>
                    Modify confidence, notes, and expert settings for "{editReingestTarget?.original_filename ?? ""}", then re-ingest.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="grid gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">Source Reliability</label>
                      <Select value={editReingestSourceReliability} onValueChange={setEditReingestSourceReliability}>
                        <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                        <SelectContent>
                          {(["A", "B", "C", "X"] as const).map((v) => (
                            <SelectItem key={v} value={v}>{v} — {confidenceSourceCompact[v]}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">Information Validity</label>
                      <Select value={editReingestInfoValidity} onValueChange={setEditReingestInfoValidity}>
                        <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                        <SelectContent>
                          {(["1", "2", "3", "4"] as const).map((v) => (
                            <SelectItem key={v} value={v}>{v} — {confidenceValidityDescriptions[v]}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Notes</label>
                    <textarea
                      value={editReingestNotes}
                      onChange={(e) => setEditReingestNotes(e.target.value)}
                      className="min-h-[80px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      placeholder="Analyst context, provenance caveats, or LLM instructions"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="grid gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">Profile</label>
                      <Select value={editReingestProfile} onValueChange={(v) => setEditReingestProfile(v as IngestionProfile)}>
                        <SelectTrigger><SelectValue placeholder="Select profile" /></SelectTrigger>
                        <SelectContent>
                          {ingestProfileOptions.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">Processing Mode</label>
                      <Select value={editReingestMode} onValueChange={(v) => setEditReingestMode(v as ProcessingMode)}>
                        <SelectTrigger><SelectValue placeholder="Select mode" /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="multimodal">Multimodal</SelectItem>
                          <SelectItem value="text_first">Text First</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Advanced Overrides (JSON)</label>
                    <textarea
                      value={editReingestAdvancedOverrides}
                      onChange={(e) => setEditReingestAdvancedOverrides(e.target.value)}
                      className="min-h-[80px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 font-mono text-xs ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      placeholder='{"parse_method": "auto", "enable_vlm": true}'
                      spellCheck={false}
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setEditReingestTarget(null)} disabled={editReingestBusy}>Cancel</Button>
                  <Button onClick={handleEditReingest} disabled={editReingestBusy}>
                    {editReingestBusy ? "Re-ingesting..." : "Reingest"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={docLogsOpen} onOpenChange={(open) => { if (!open) { setDocLogsOpen(false); setJobLogsDocumentTarget(null); setDocLogs([]); setDocLogsError(null); } }}>
              <DialogContent className="max-h-[90vh] max-w-4xl overflow-hidden">
                <DialogHeader>
                  <DialogTitle>Ingestion Logs</DialogTitle>
                  <DialogDescription>
                    {jobLogsDocumentTarget && documentsById.get(jobLogsDocumentTarget) ? (
                      <span>{documentsById.get(jobLogsDocumentTarget)?.original_filename ?? "Unknown"}</span>
                    ) : <span>Job logs</span>}
                  </DialogDescription>
                </DialogHeader>
                {docLogsError ? (
                  <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">{docLogsError}</div>
                ) : null}
                <div className="rounded-lg border bg-slate-950/95 p-3">
                  {docLogsLoading ? (
                    <div className="h-80 overflow-auto text-xs text-slate-200">Loading logs...</div>
                  ) : docLogs.length === 0 ? (
                    <div className="h-80 overflow-auto text-xs text-slate-200">No logs available yet.</div>
                  ) : (
                    <pre ref={docLogsPreRef} className="h-80 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-slate-100">
                      {docLogs.map((entry) => `[${formatDateTime(entry.created_at)}] ${entry.level.toUpperCase()}: ${entry.message}`).join("\n")}
                    </pre>
                  )}
                </div>
                <DialogFooter>
                  <Button variant="outline" size="sm" onClick={() => { setDocLogsOpen(false); setJobLogsDocumentTarget(null); setDocLogs([]); }}>Close</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen}>
              <div className="grid gap-3">
                <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">Chat</p>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground">Developer mode</span>
                  <Button
                    variant={developerMode ? "default" : "outline"}
                    size="sm"
                    onClick={() => setDeveloperMode((prev) => !prev)}
                  >
                    {developerMode ? "On" : "Off"}
                  </Button>
                </div>
                <div className="grid gap-1">
                  <label className="text-xs text-muted-foreground">Default chat mode (when dev mode is on)</label>
                  <Select value={chatMode} onValueChange={(value) => setChatMode(value as ChatMode)}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="hybrid" />
                    </SelectTrigger>
                    <SelectContent>
                      {chatModeOptions.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </SettingsDialog>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const jobMatch = useMemo(() => {
    return window.location.pathname.match(/^\/cases\/([^/]+)\/jobs\/?$/);
  }, []);

  if (jobMatch) {
    return <CaseJobsView caseId={jobMatch[1]} />;
  }

  const caseId = useMemo(() => {
    const match = window.location.pathname.match(/^\/cases\/([^/]+)\/?$/);
    return match?.[1] ?? null;
  }, []);

  if (caseId) {
    return <CaseWorkspace caseId={caseId} />;
  }

  return <CasesDirectory />;
}
