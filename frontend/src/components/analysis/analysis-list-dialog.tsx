import { useState } from "react";
import { CalendarClock, GitBranch, ListChecks, Loader2, Network, Plus, RefreshCcw, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import type { AnalysisRecord, AnalysisType } from "@/lib/api";

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

function formatAnalysisTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

interface AnalysisListDialogProps {
  open: boolean;
  analyses: AnalysisRecord[];
  activeAnalysisId: string | null;
  loading: boolean;
  deletingAnalysisId: string | null;
  retryingAnalysisId: string | null;
  onOpenChange: (open: boolean) => void;
  onSelect: (analysis: AnalysisRecord) => void;
  onDelete: (analysis: AnalysisRecord) => Promise<void>;
  onRetry: (analysis: AnalysisRecord) => Promise<void>;
  onNewAnalysis: () => void;
}

export function AnalysisListDialog({
  open,
  analyses,
  activeAnalysisId,
  loading,
  deletingAnalysisId,
  retryingAnalysisId,
  onOpenChange,
  onSelect,
  onDelete,
  onRetry,
  onNewAnalysis
}: AnalysisListDialogProps) {
  const [deleteTarget, setDeleteTarget] = useState<AnalysisRecord | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleConfirmDelete = async () => {
    if (!deleteTarget) {
      return;
    }
    setDeleteError(null);
    try {
      await onDelete(deleteTarget);
      setDeleteTarget(null);
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : "Unable to delete analysis.");
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="grid max-h-[82vh] max-w-3xl grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden">
          <DialogHeader>
            <DialogTitle>Analysis history</DialogTitle>
            <DialogDescription>
              Reopen or delete generated charts and summaries for this case.
            </DialogDescription>
          </DialogHeader>

          <div className="min-h-0 overflow-y-auto pr-1">
            {loading ? (
              <div className="rounded-xl border border-dashed border-muted/60 p-8 text-center text-sm text-muted-foreground">
                Loading analyses...
              </div>
            ) : analyses.length === 0 ? (
              <div className="flex min-h-[14rem] flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-muted/60 p-8 text-center">
                <ListChecks className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">No analyses have been generated for this case.</p>
                <Button size="sm" onClick={onNewAnalysis}>
                  <Plus className="h-4 w-4" />
                  New analysis
                </Button>
              </div>
            ) : (
              <div className="divide-y rounded-xl border bg-background/60">
                {analyses.map((analysis) => {
                  const Icon = analysisIcons[analysis.analysis_type] ?? Network;
                  const active = analysis.id === activeAnalysisId;
                  const deleting = deletingAnalysisId === analysis.id;
                  const retrying = retryingAnalysisId === analysis.id;
                  const running = ["queued", "generating", "repair_queued", "repairing"].includes(
                    analysis.status
                  );
                  return (
                    <div
                      key={analysis.id}
                      className={["flex min-w-0 items-center", active ? "bg-primary/5" : ""].join(" ")}
                    >
                      <button
                        type="button"
                        className="min-w-0 flex-1 px-4 py-3 text-left transition-colors hover:bg-accent"
                        onClick={() => onSelect(analysis)}
                        disabled={deleting}
                      >
                        <div className="flex min-w-0 items-start justify-between gap-3">
                          <div className="flex min-w-0 items-start gap-3">
                            <span className="mt-0.5 rounded-md border bg-card p-1.5">
                              <Icon className="h-4 w-4 text-muted-foreground" />
                            </span>
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-foreground">{analysis.title}</p>
                              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{analysis.prompt}</p>
                              {analysis.status === "failed" && analysis.error ? (
                                <p className="mt-2 line-clamp-2 text-xs text-destructive">{analysis.error}</p>
                              ) : null}
                            </div>
                          </div>
                          <div className="flex shrink-0 flex-col items-end gap-1">
                            <div className="flex items-center gap-1">
                              <Badge variant={active ? "default" : "outline"}>
                                {analysisLabels[analysis.analysis_type]}
                              </Badge>
                              {analysis.status !== "complete" ? (
                                <Badge variant="outline">{analysis.status.replace("_", " ")}</Badge>
                              ) : null}
                            </div>
                            <span className="text-[11px] text-muted-foreground">
                              {formatAnalysisTime(analysis.updated_at)}
                            </span>
                          </div>
                        </div>
                      </button>
                      {analysis.status === "failed" ? (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-9 w-9 shrink-0 text-muted-foreground"
                          onClick={() => void onRetry(analysis)}
                          disabled={retrying || deleting || running}
                          aria-label={`Retry ${analysis.title}`}
                          title="Retry analysis"
                        >
                          {retrying ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <RefreshCcw className="h-4 w-4" />
                          )}
                        </Button>
                      ) : null}
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="mr-2 h-9 w-9 shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => {
                          setDeleteError(null);
                          setDeleteTarget(analysis);
                        }}
                        disabled={deleting}
                        aria-label={`Delete ${analysis.title}`}
                        title="Delete analysis"
                      >
                        {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            <Button onClick={onNewAnalysis}>
              <Plus className="h-4 w-4" />
              New analysis
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(nextOpen) => {
          if (!nextOpen && !deletingAnalysisId) {
            setDeleteTarget(null);
            setDeleteError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete analysis</DialogTitle>
            <DialogDescription>
              Delete "{deleteTarget?.title ?? "this analysis"}"? The generated chart, summary, and grounding data will be removed permanently.
            </DialogDescription>
          </DialogHeader>
          {deleteError ? (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {deleteError}
            </div>
          ) : null}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={Boolean(deletingAnalysisId)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => void handleConfirmDelete()}
              disabled={Boolean(deletingAnalysisId)}
            >
              {deletingAnalysisId ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              {deletingAnalysisId ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
