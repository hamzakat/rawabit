import { useEffect, useState } from "react";
import { CalendarClock, GitBranch, Loader2, Network } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { AnalysisType } from "@/lib/api";

const analysisTypeOptions: Array<{
  value: AnalysisType;
  label: string;
  description: string;
  icon: typeof Network;
}> = [
  {
    value: "link",
    label: "Link analysis",
    description: "Associations, roles, ownership, and hidden connectors.",
    icon: Network
  },
  {
    value: "flow",
    label: "Flow analysis",
    description: "Movement of money, goods, influence, actions, or communications.",
    icon: GitBranch
  },
  {
    value: "event",
    label: "Event analysis",
    description: "Chronology, sequence, hypothesized events, and actor lanes.",
    icon: CalendarClock
  }
];

interface AnalysisCreateDialogProps {
  open: boolean;
  busy: boolean;
  hasEvidence: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: { prompt: string; analysisType: AnalysisType }) => Promise<void>;
}

export function AnalysisCreateDialog({
  open,
  busy,
  hasEvidence,
  onOpenChange,
  onSubmit
}: AnalysisCreateDialogProps) {
  const [prompt, setPrompt] = useState("");
  const [analysisType, setAnalysisType] = useState<AnalysisType>("link");
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setFormError(null);
    }
  }, [open]);

  const selectedOption = analysisTypeOptions.find((option) => option.value === analysisType) ?? analysisTypeOptions[0];
  const SelectedIcon = selectedOption.icon;
  const canSubmit = hasEvidence && prompt.trim().length > 0 && !busy;

  const handleSubmit = async () => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      setFormError("Enter an analysis question.");
      return;
    }
    if (!hasEvidence) {
      setFormError("Upload evidence before creating an analysis.");
      return;
    }
    setFormError(null);
    await onSubmit({ prompt: trimmed, analysisType });
    setPrompt("");
    setAnalysisType("link");
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>New analysis</DialogTitle>
          <DialogDescription>
            Ask an investigative question and choose the UNODC analysis method for the generated chart.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-5">
          <div className="grid gap-2">
            <label className="text-sm font-medium" htmlFor="analysis-prompt">
              Analysis question
            </label>
            <textarea
              id="analysis-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Example: show the key actors and money movement around the warehouse transfer"
              className="min-h-[132px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm leading-6 text-foreground shadow-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              disabled={busy}
            />
            <p className="text-xs text-muted-foreground">
              The system will query the case RAG, extract the highlighted subgraph, and generate a chart plus narrative.
            </p>
          </div>

          <div className="grid gap-2">
            <label className="text-sm font-medium" htmlFor="analysis-type">
              Analysis type
            </label>
            <Select
              value={analysisType}
              onValueChange={(value) => setAnalysisType(value as AnalysisType)}
              disabled={busy}
            >
              <SelectTrigger id="analysis-type">
                <SelectValue placeholder="Select analysis type" />
              </SelectTrigger>
              <SelectContent>
                {analysisTypeOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="rounded-lg border bg-background/70 p-3">
              <div className="flex items-start gap-3">
                <SelectedIcon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium text-foreground">{selectedOption.label}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{selectedOption.description}</p>
                </div>
              </div>
            </div>
          </div>

          {!hasEvidence ? (
            <div className="rounded-md border border-amber-300/70 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100">
              Upload evidence before creating analyses.
            </div>
          ) : null}

          {formError ? <p className="text-sm text-destructive">{formError}</p> : null}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={() => void handleSubmit()} disabled={!canSubmit}>
            {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Generate
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
