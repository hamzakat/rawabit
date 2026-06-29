import { useCallback, useEffect, useRef, useState } from "react";
import {
  CalendarClock,
  FileText,
  GitBranch,
  ListChecks,
  Loader2,
  Network,
  Plus,
  RefreshCcw
} from "lucide-react";

import { AnalysisCreateDialog } from "@/components/analysis/analysis-create-dialog";
import { AnalysisListDialog } from "@/components/analysis/analysis-list-dialog";
import { MermaidChartViewer } from "@/components/analysis/mermaid-chart-viewer";
import { GraphCanvas, type GraphLegendSelection } from "@/components/graph/graph-canvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import { MarkdownText } from "@/components/ui/markdown";
import type {
  AnalysisChart,
  AnalysisRecord,
  AnalysisType,
  GraphEdge,
  GraphNode,
  HighlightRelationship
} from "@/lib/api";
import {
  createAnalysis,
  deleteAnalysis,
  fetchAnalyses,
  fetchAnalysis,
  repairAnalysisChart,
  retryAnalysis
} from "@/lib/api";

const analysisIcons: Record<AnalysisType, typeof Network> = {
  link: Network,
  flow: GitBranch,
  event: CalendarClock
};

const analysisLabels: Record<AnalysisType, string> = {
  link: "Link Analysis",
  flow: "Flow Analysis",
  event: "Event Analysis"
};

const activeStatuses = new Set(["queued", "generating", "repair_queued", "repairing"]);

interface AnalysisWorkspaceProps {
  caseId: string;
  hasEvidence: boolean;
  onInspectGraphElement?: (
    target:
      | { kind: "node"; node: GraphNode }
      | { kind: "edge"; edge: GraphEdge }
  ) => void;
}

export function AnalysisWorkspace({ caseId, hasEvidence, onInspectGraphElement }: AnalysisWorkspaceProps) {
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingAnalysisId, setDeletingAnalysisId] = useState<string | null>(null);
  const [retryingAnalysisId, setRetryingAnalysisId] = useState<string | null>(null);
  const [repairingChartId, setRepairingChartId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [listOpen, setListOpen] = useState(false);
  const [groundingOpen, setGroundingOpen] = useState(false);
  const [groundingLegendSelection, setGroundingLegendSelection] = useState<GraphLegendSelection>(null);
  const [error, setError] = useState<string | null>(null);
  const repairCodeRef = useRef<string | null>(null);

  const refreshAnalyses = useCallback(
    async (preferredId?: string | null) => {
      setLoading(true);
      setError(null);
      try {
        const list = await fetchAnalyses(caseId);
        setAnalyses(list);
        const next =
          (preferredId ? list.find((analysis) => analysis.id === preferredId) : null) ?? list[0] ?? null;
        setActiveAnalysis(next);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load analyses.");
      } finally {
        setLoading(false);
      }
    },
    [caseId]
  );

  useEffect(() => {
    setAnalyses([]);
    setActiveAnalysis(null);
    void refreshAnalyses();
  }, [caseId, refreshAnalyses]);

  useEffect(() => {
    if (!analyses.some((analysis) => activeStatuses.has(analysis.status))) {
      return;
    }
    const timer = window.setInterval(() => {
      void fetchAnalyses(caseId)
        .then((list) => {
          setAnalyses(list);
          setActiveAnalysis((current) => {
            if (!current) return list[0] ?? null;
            return list.find((item) => item.id === current.id) ?? list[0] ?? null;
          });
          if (!list.some((analysis) => activeStatuses.has(analysis.status))) {
            repairCodeRef.current = null;
            setRepairingChartId(null);
          }
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Unable to refresh analysis.");
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [analyses, caseId]);

  const handleCreateAnalysis = useCallback(
    async ({ prompt, analysisType }: { prompt: string; analysisType: AnalysisType }) => {
      setCreating(true);
      setError(null);
      try {
        const analysis = await createAnalysis(caseId, {
          prompt,
          analysis_type: analysisType
        });
        setAnalyses((previous) => [analysis, ...previous.filter((item) => item.id !== analysis.id)]);
        setActiveAnalysis(analysis);
        setCreateOpen(false);
        setListOpen(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to create analysis.");
      } finally {
        setCreating(false);
      }
    },
    [caseId]
  );

  const handleSelectAnalysis = useCallback(
    async (analysis: AnalysisRecord) => {
      setListOpen(false);
      setActiveAnalysis(analysis);
      setError(null);
      try {
        const detail = await fetchAnalysis(caseId, analysis.id);
        setActiveAnalysis(detail);
        setAnalyses((previous) => previous.map((item) => (item.id === detail.id ? detail : item)));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load analysis.");
      }
    },
    [caseId]
  );

  const handleDeleteAnalysis = useCallback(
    async (analysis: AnalysisRecord) => {
      setDeletingAnalysisId(analysis.id);
      setError(null);
      try {
        await deleteAnalysis(caseId, analysis.id);
        const remaining = analyses.filter((item) => item.id !== analysis.id);
        setAnalyses(remaining);
        if (activeAnalysis?.id === analysis.id) {
          repairCodeRef.current = null;
          setActiveAnalysis(remaining[0] ?? null);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to delete analysis.";
        setError(message);
        throw err instanceof Error ? err : new Error(message);
      } finally {
        setDeletingAnalysisId(null);
      }
    },
    [activeAnalysis?.id, analyses, caseId]
  );

  const handleRetryAnalysis = useCallback(
    async (analysis: AnalysisRecord) => {
      if (activeStatuses.has(analysis.status)) return;
      setRetryingAnalysisId(analysis.id);
      setError(null);
      try {
        const retried = await retryAnalysis(caseId, analysis.id);
        setAnalyses((previous) =>
          previous.map((item) => (item.id === retried.id ? retried : item))
        );
        setActiveAnalysis(retried);
        setListOpen(false);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to retry analysis.";
        setError(message);
        throw err instanceof Error ? err : new Error(message);
      } finally {
        setRetryingAnalysisId(null);
      }
    },
    [caseId]
  );

  const handleRenderError = useCallback(
    (chart: AnalysisChart, message: string, brokenCode: string) => {
      if (!activeAnalysis || repairingChartId || repairCodeRef.current === brokenCode) return;
      if (chart.repair_attempts >= 5) {
        setError(`Mermaid repair limit reached for "${chart.title}".`);
        return;
      }
      repairCodeRef.current = brokenCode;
      setRepairingChartId(chart.id);
      setError(null);
      void repairAnalysisChart(caseId, activeAnalysis.id, {
        chart_id: chart.id,
        error: message,
        mermaid_code: brokenCode
      })
        .then((repaired) => {
          setActiveAnalysis(repaired);
          setAnalyses((previous) => previous.map((item) => (item.id === repaired.id ? repaired : item)));
        })
        .catch((err) => {
          repairCodeRef.current = null;
          setRepairingChartId(null);
          setError(err instanceof Error ? err.message : "Unable to repair Mermaid chart.");
        });
    },
    [activeAnalysis, caseId, repairingChartId]
  );

  const ActiveIcon = activeAnalysis ? analysisIcons[activeAnalysis.analysis_type] ?? Network : Network;
  const summaryText = activeAnalysis?.summary_text?.trim() || "No narrative summary generated yet.";
  const evidenceCount = activeAnalysis?.references.length ?? 0;
  const chunkCount = activeAnalysis?.chunks.length ?? 0;
  const subgraphNodes = activeAnalysis?.subgraph.nodes ?? [];
  const subgraphEdges = activeAnalysis?.subgraph.edges ?? [];
  const subgraphHighlights: HighlightRelationship[] = subgraphEdges.map((edge) => ({
    edge_id: edge.id,
    relation_type: edge.relation_type,
    src_id: edge.src_id,
    tgt_id: edge.tgt_id
  }));

  const handleInspectNode = useCallback(
    (node: GraphNode) => {
      setGroundingOpen(false);
      onInspectGraphElement?.({ kind: "node", node });
    },
    [onInspectGraphElement]
  );

  const handleInspectEdge = useCallback(
    (edge: GraphEdge) => {
      setGroundingOpen(false);
      onInspectGraphElement?.({ kind: "edge", edge });
    },
    [onInspectGraphElement]
  );

  return (
    <section className="rounded-2xl border bg-card/90 p-3 shadow-soft md:p-4 xl:flex xl:h-[calc(100dvh-7rem)] xl:min-h-[34rem] xl:flex-col xl:overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/60 pb-3">
        <div className="flex min-w-0 flex-1 basis-full items-start gap-3 xl:basis-[28rem]">
          <span className="rounded-lg border bg-background p-2">
            <ActiveIcon className="h-5 w-5 text-muted-foreground" />
          </span>
          <div className="min-w-0">
            <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">AI Analyzer</p>
            <h2 className="max-w-full break-words text-lg font-semibold leading-snug text-foreground">
              {activeAnalysis?.prompt ?? "Analyses"}
            </h2>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {activeAnalysis ? (
            <Badge variant="outline">{analysisLabels[activeAnalysis.analysis_type]}</Badge>
          ) : null}
          <Button variant="outline" size="sm" onClick={() => void refreshAnalyses(activeAnalysis?.id)}>
            <RefreshCcw className="h-4 w-4" />
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={() => setListOpen(true)}>
            <ListChecks className="h-4 w-4" />
            History
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)} disabled={!hasEvidence || creating}>
            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            New analysis
          </Button>
        </div>
      </div>

      {error ? (
        <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      {!hasEvidence ? (
        <div className="mt-4 flex min-h-[24rem] flex-1 items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-center">
          <div>
            <Network className="mx-auto h-8 w-8 text-muted-foreground" />
            <p className="mt-3 text-sm font-medium text-foreground">No evidence available</p>
            <p className="mt-1 max-w-md text-sm text-muted-foreground">
              Upload and ingest evidence before creating visual analyses.
            </p>
          </div>
        </div>
      ) : loading && analyses.length === 0 ? (
        <div className="mt-4 flex min-h-[24rem] flex-1 items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-sm text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Loading analyses...
        </div>
      ) : !activeAnalysis ? (
        <div className="mt-4 flex min-h-[24rem] flex-1 items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-center">
          <div>
            <Network className="mx-auto h-8 w-8 text-muted-foreground" />
            <p className="mt-3 text-sm font-medium text-foreground">No analyses yet</p>
            <p className="mt-1 max-w-md text-sm text-muted-foreground">
              Create a link, flow, or event analysis from this case workspace.
            </p>
            <Button className="mt-4" onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4" />
              New analysis
            </Button>
          </div>
        </div>
      ) : activeStatuses.has(activeAnalysis.status) ? (
        <div className="mt-4 flex min-h-[24rem] flex-1 items-center justify-center rounded-xl border border-dashed border-muted/60 p-8 text-center">
          <div>
            <Loader2 className="mx-auto h-7 w-7 animate-spin text-muted-foreground" />
            <p className="mt-3 text-sm font-medium text-foreground">Working on analysis...</p>
            <p className="mt-1 text-sm text-muted-foreground">
              The job continues in the background. This view refreshes automatically.
            </p>
          </div>
        </div>
      ) : activeAnalysis.status === "failed" ? (
        <div className="mt-4 flex min-h-[24rem] flex-1 items-center justify-center rounded-xl border border-destructive/30 bg-destructive/5 p-8 text-center">
          <div className="max-w-lg">
            <p className="text-sm font-medium text-foreground">Analysis failed</p>
            <p className="mt-2 text-sm text-muted-foreground">
              {activeAnalysis.error || "The analysis could not be completed."}
            </p>
            <Button
              className="mt-4"
              onClick={() => void handleRetryAnalysis(activeAnalysis)}
              disabled={retryingAnalysisId === activeAnalysis.id}
            >
              {retryingAnalysisId === activeAnalysis.id ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCcw className="h-4 w-4" />
              )}
              Retry
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-4 grid min-h-[32rem] gap-4 overflow-visible xl:min-h-0 xl:flex-1 xl:grid-cols-[minmax(0,2fr)_minmax(20rem,1fr)] xl:overflow-hidden">
          <div className="min-h-[28rem] xl:min-h-0">
            <MermaidChartViewer
              charts={activeAnalysis.charts}
              title={activeAnalysis.title}
              repairingChartId={repairingChartId}
              onRenderError={handleRenderError}
            />
          </div>
          <aside className="grid content-start gap-4 rounded-xl border bg-background/60 p-4 xl:overflow-y-auto">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Summary</p>
              <MarkdownText content={summaryText} className="mt-3 text-sm" />
            </div>
            <div className="grid grid-cols-2 gap-2 border-t border-border/60 pt-4 text-xs">
              <button
                type="button"
                className="rounded-md border bg-card px-3 py-2 text-left transition-colors hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                onClick={() => setGroundingOpen(true)}
              >
                <p className="flex items-center gap-1 text-muted-foreground">
                  <FileText className="h-3.5 w-3.5" />
                  References
                </p>
                <p className="mt-1 text-base font-semibold text-foreground">{evidenceCount}</p>
              </button>
              <button
                type="button"
                className="rounded-md border bg-card px-3 py-2 text-left transition-colors hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                onClick={() => setGroundingOpen(true)}
              >
                <p className="text-muted-foreground">Chunks</p>
                <p className="mt-1 text-base font-semibold text-foreground">{chunkCount}</p>
              </button>
              <div className="rounded-md border bg-card px-3 py-2">
                <p className="text-muted-foreground">Charts</p>
                <p className="mt-1 text-base font-semibold text-foreground">
                  {activeAnalysis.charts.length}
                </p>
              </div>
              <div className="rounded-md border bg-card px-3 py-2">
                <p className="text-muted-foreground">Model</p>
                <p className="mt-1 truncate text-xs font-medium text-foreground" title={activeAnalysis.model_name ?? ""}>
                  {activeAnalysis.model_name ?? "Unknown"}
                </p>
              </div>
            </div>
            <div className="border-t border-border/60 pt-4">
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Question</p>
              <p className="mt-2 whitespace-pre-wrap break-words text-sm text-muted-foreground">
                {activeAnalysis.prompt}
              </p>
            </div>
          </aside>
        </div>
      )}

      <AnalysisCreateDialog
        open={createOpen}
        busy={creating}
        hasEvidence={hasEvidence}
        onOpenChange={setCreateOpen}
        onSubmit={handleCreateAnalysis}
      />
      <AnalysisListDialog
        open={listOpen}
        analyses={analyses}
        activeAnalysisId={activeAnalysis?.id ?? null}
        loading={loading}
        deletingAnalysisId={deletingAnalysisId}
        retryingAnalysisId={retryingAnalysisId}
        onOpenChange={setListOpen}
        onSelect={handleSelectAnalysis}
        onDelete={handleDeleteAnalysis}
        onRetry={handleRetryAnalysis}
        onNewAnalysis={() => {
          setListOpen(false);
          setCreateOpen(true);
        }}
      />
      <Dialog open={groundingOpen} onOpenChange={setGroundingOpen}>
        <DialogContent className="grid max-h-[90vh] w-[calc(100vw-2rem)] max-w-5xl grid-rows-[auto_minmax(0,1fr)] overflow-hidden">
          <DialogHeader>
            <DialogTitle>Analysis grounding</DialogTitle>
            <DialogDescription>
              References, supporting chunks, and subgraph context used for this analysis.
            </DialogDescription>
          </DialogHeader>
          <div className="grid min-h-0 gap-4 overflow-y-auto overflow-x-hidden pr-1">
            <section className="grid h-[min(34rem,58vh)] min-h-[22rem] min-w-0 grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-xl border bg-background/60 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm font-semibold text-foreground">Relevant subgraph</p>
                <div className="flex gap-2">
                  <Badge variant="outline">{subgraphNodes.length} nodes</Badge>
                  <Badge variant="outline">{subgraphEdges.length} edges</Badge>
                </div>
              </div>
              {subgraphNodes.length || subgraphEdges.length ? (
                <GraphCanvas
                  className="!mt-3 !h-full !min-h-0 overflow-hidden border-solid"
                  nodes={subgraphNodes}
                  edges={subgraphEdges}
                  selectedNodeId={null}
                  selectedEdge={null}
                  highlightedNodeIds={subgraphNodes.map((node) => node.id)}
                  highlightedEdges={subgraphHighlights}
                  legendSelection={groundingLegendSelection}
                  showLabels
                  multiSelectedNodeIds={new Set()}
                  multiSelectedEdgeKeys={new Set()}
                  onNodeSelect={handleInspectNode}
                  onEdgeSelect={handleInspectEdge}
                  onCanvasBackgroundSelect={() => setGroundingLegendSelection(null)}
                  onLegendSelect={setGroundingLegendSelection}
                  onNodeMultiSelect={handleInspectNode}
                  onEdgeMultiSelect={handleInspectEdge}
                />
              ) : (
                <p className="mt-3 rounded-md border border-dashed border-muted/60 p-3 text-sm text-muted-foreground">
                  No relevant subgraph was returned for this analysis.
                </p>
              )}
            </section>
            <div className="grid content-start gap-4">
              <section className="rounded-xl border bg-background/60 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold text-foreground">References</p>
                  <Badge variant="outline">{evidenceCount}</Badge>
                </div>
                <div className="mt-3 grid gap-2">
                  {activeAnalysis?.references.length ? (
                    activeAnalysis.references.map((reference, index) => (
                      <div key={`${reference.reference_id}-${index}`} className="rounded-md border bg-card px-3 py-2 text-xs">
                        <p className="font-mono text-foreground">{reference.reference_id}</p>
                        <p className="mt-1 break-words text-muted-foreground">{reference.file_path}</p>
                      </div>
                    ))
                  ) : (
                    <p className="rounded-md border border-dashed border-muted/60 p-3 text-sm text-muted-foreground">
                      No references were returned for this analysis.
                    </p>
                  )}
                </div>
              </section>

              <section className="rounded-xl border bg-background/60 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold text-foreground">Supporting chunks</p>
                  <Badge variant="outline">{chunkCount}</Badge>
                </div>
                <div className="mt-3 grid gap-2">
                  {activeAnalysis?.chunks.length ? (
                    activeAnalysis.chunks.map((chunk, index) => (
                      <div key={`${chunk.reference_id}-${index}`} className="rounded-md border bg-card px-3 py-2 text-xs">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-foreground">{chunk.reference_id}</span>
                        </div>
                        <p className="mt-1 break-words text-muted-foreground">{chunk.file_path}</p>
                        {chunk.snippet ? (
                          <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                            {chunk.snippet}
                          </p>
                        ) : null}
                      </div>
                    ))
                  ) : (
                    <p className="rounded-md border border-dashed border-muted/60 p-3 text-sm text-muted-foreground">
                      No chunk snippets were returned for this analysis.
                    </p>
                  )}
                </div>
              </section>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
}
