import { useCallback, useEffect, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { fetchSettings, updateSettings, type SettingsPayload } from "@/lib/api";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children?: ReactNode;
};

export function SettingsDialog({ open, onOpenChange, children }: Props) {
  const [data, setData] = useState<SettingsPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await fetchSettings();
      setData(result);
      setOverrides((result.overrides as Record<string, string>) ?? {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load settings.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      void load();
    } else {
      setData(null);
      setError(null);
    }
  }, [open, load]);

  const handleSave = useCallback(async () => {
    try {
      setSaving(true);
      setError(null);
      await updateSettings(overrides);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save settings.");
    } finally {
      setSaving(false);
    }
  }, [overrides, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Override system defaults. Changes take effect immediately — no restart needed.
          </DialogDescription>
        </DialogHeader>
        {loading ? (
          <div className="py-4 text-sm text-muted-foreground">Loading settings...</div>
        ) : error ? (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div>
        ) : (
          <div className="grid gap-6">
            <div className="grid gap-3">
              <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">Models</p>
              {(["rag_llm_model", "rag_vlm_model", "rag_embedding_model"] as const).map((key) => {
                const label = key === "rag_llm_model" ? "LLM Model" : key === "rag_vlm_model" ? "VLM Model" : "Embedding Model";
                const hint =
                  key === "rag_llm_model"
                    ? "OpenRouter model ID for entity extraction, Q&A, summarization, and evidence normalization."
                    : key === "rag_vlm_model"
                      ? "OpenRouter model ID for vision-language tasks — image description, visual analysis, OCR."
                      : "OpenRouter model ID for text and graph vector embeddings.";
                const defaultValue = (data?.effective?.[key] as string) ?? "";
                return (
                  <div key={key} className="grid gap-1">
                    <label className="text-xs text-muted-foreground">{label}</label>
                    <Input
                      value={overrides[key] ?? ""}
                      placeholder={defaultValue || "Default"}
                      onChange={(e) => setOverrides((prev) => ({ ...prev, [key]: e.target.value }))}
                    />
                    <p className="text-[0.7rem] leading-tight text-muted-foreground/80">{hint}</p>
                  </div>
                );
              })}
              <div className="grid gap-1">
                <label className="text-xs text-muted-foreground">Embedding Dim Hint</label>
                <Input
                  type="number"
                  value={overrides["rag_embedding_dim_hint"] ?? ""}
                  placeholder={data?.effective?.rag_embedding_dim_hint != null ? String(data.effective.rag_embedding_dim_hint) : "1536"}
                  onChange={(e) => setOverrides((prev) => ({ ...prev, rag_embedding_dim_hint: e.target.value }))}
                />
                <p className="text-[0.7rem] leading-tight text-muted-foreground/80">Vector dimensionality matching the chosen embedding model. Must align with the model spec or ingestion fails.</p>
              </div>
            </div>
            {children}
            <div className="grid gap-3">
              <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">Retrieval</p>
              {(["rag_cosine_threshold", "rag_default_top_k", "rag_default_chunk_top_k"] as const).map((key) => {
                const label = key === "rag_cosine_threshold" ? "Cosine Threshold"
                  : key === "rag_default_top_k" ? "Retrieval Top-K"
                  : "Retrieval Chunk Top-K";
                const hint =
                  key === "rag_cosine_threshold"
                    ? "Minimum similarity for a retrieval result to be considered relevant. Raise to filter out weak matches, lower to surface more candidates."
                    : key === "rag_default_top_k"
                      ? "Default number of graph entities and relationships to retrieve per query. Higher = more context but slower."
                      : "Default number of supporting text chunks to retrieve per query. Higher = richer evidence but more tokens.";
                const step = key === "rag_cosine_threshold" ? "0.01" : undefined;
                return (
                  <Field key={key} label={label} value={overrides[key]} onValueChange={(v) => setOverrides((prev) => ({ ...prev, [key]: v }))} data={data} field={key} step={step} hint={hint} />
                );
              })}
            </div>
            <div className="grid gap-3">
              <p className="text-xs font-medium uppercase tracking-[0.15em] text-muted-foreground">System</p>
              {(["rag_llm_temperature"] as const).map((key) => {
                const label = "LLM Temperature";
                const hint = "Controls response determinism. Near 0 = factual and consistent. Higher = more varied and creative.";
                return (
                  <Field key={key} label={label} value={overrides[key]} onValueChange={(v) => setOverrides((prev) => ({ ...prev, [key]: v }))} data={data} field={key} step="0.01" hint={hint} />
                );
              })}
              {(["rag_llm_max_async", "rag_embedding_max_async", "rag_llm_timeout_seconds", "rag_lightrag_max_parallel_insert", "ingestion_worker_concurrency", "rag_llm_max_tokens"] as const).map((key) => {
                const label = key === "rag_llm_max_async" ? "LLM Max Async"
                  : key === "rag_embedding_max_async" ? "Embedding Max Async"
                  : key === "rag_llm_timeout_seconds" ? "LLM Timeout (s)"
                  : key === "rag_lightrag_max_parallel_insert" ? "Max Parallel Insert"
                  : key === "ingestion_worker_concurrency" ? "Ingestion Workers"
                  : "LLM Max Tokens";
                const hint =
                  key === "rag_llm_max_async"
                    ? "Maximum concurrent LLM API calls during ingestion. Increase for faster processing if provider rate limits allow."
                    : key === "rag_embedding_max_async"
                      ? "Maximum concurrent embedding API calls during ingestion. Higher = faster graph construction."
                      : key === "rag_llm_timeout_seconds"
                        ? "Request timeout for each LLM API call. Increase for slow providers or large extraction jobs."
                        : key === "rag_lightrag_max_parallel_insert"
                          ? "Maximum parallel insert workers per document during LightRAG ingestion."
                          : key === "ingestion_worker_concurrency"
                            ? "Number of documents processed concurrently. Increase if you have spare API capacity."
                            : "Maximum output tokens for LLM calls — extraction, normalization, and generation.";
                return (
                  <Field key={key} label={label} value={overrides[key]} onValueChange={(v) => setOverrides((prev) => ({ ...prev, [key]: v }))} data={data} field={key} hint={hint} />
                );
              })}
            </div>
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={handleSave} disabled={saving || loading}>{saving ? "Saving..." : "Save"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  value,
  onValueChange,
  data,
  field,
  step,
  hint,
}: {
  label: string;
  value: string | undefined;
  onValueChange: (v: string) => void;
  data: SettingsPayload | null;
  field: string;
  step?: string;
  hint?: string;
}) {
  const defaultValue = data?.effective?.[field];
  return (
    <div className="grid gap-1">
      <label className="text-xs text-muted-foreground">{label}</label>
      <Input
        type="number"
        step={step}
        value={value ?? ""}
        placeholder={defaultValue != null ? String(defaultValue) : ""}
        onChange={(e) => onValueChange(e.target.value)}
      />
      {hint ? <p className="text-[0.7rem] leading-tight text-muted-foreground/80">{hint}</p> : null}
    </div>
  );
}
