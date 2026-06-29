import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent,
  type WheelEvent
} from "react";
import { Download, Expand, Loader2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import type { AnalysisChart } from "@/lib/api";

let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;
const PNG_EXPORT_FONT = "Arial, Helvetica, sans-serif";

function getMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "base",
        fontFamily: '"Space Grotesk", ui-sans-serif, system-ui, sans-serif',
        flowchart: {
          htmlLabels: false,
          useMaxWidth: false,
          curve: "basis"
        },
        timeline: {
          disableMulticolor: true,
          useMaxWidth: false
        },
        themeVariables: {
          background: "#fbf8f3",
          primaryColor: "#ffffff",
          primaryBorderColor: "#cbd5e1",
          primaryTextColor: "#151b28",
          lineColor: "#64748b",
          tertiaryColor: "#f1f5f9",
          fontSize: "16px"
        },
        themeCSS: `
      .person > rect,.person > circle,.person > polygon,.person > path { fill:#fef3c7!important;stroke:#d97706!important;stroke-width:2px!important }
      .organization > rect,.organization > circle,.organization > polygon,.organization > path { fill:#dbeafe!important;stroke:#2563eb!important;stroke-width:2px!important }
      .object > rect,.object > circle,.object > polygon,.object > path { fill:#ede9fe!important;stroke:#6d28d9!important;stroke-width:2px!important }
      .location > rect,.location > circle,.location > polygon,.location > path { fill:#dcfce7!important;stroke:#16a34a!important;stroke-width:2px!important }
      .event > rect,.event > circle,.event > polygon,.event > path { fill:#fee2e2!important;stroke:#dc2626!important;stroke-width:2px!important }
      .process > rect,.process > circle,.process > polygon,.process > path { fill:#f1f5f9!important;stroke:#475569!important;stroke-width:2px!important }
      .unknown > rect,.unknown > circle,.unknown > polygon,.unknown > path,.hypothesis > rect,.hypothesis > circle,.hypothesis > polygon,.hypothesis > path { stroke-dasharray:5 4!important }
      .label,.nodeLabel,.edgeLabel { color:#151b28!important;fill:#151b28!important }
      .edgeLabel rect { fill:#fbf8f3!important;opacity:.94!important }
    `
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

function sanitizeFilename(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "analysis-chart";
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function stripUnsafeCssReferences(value: string) {
  return value
    .replace(/@import[^;]+;/gi, "")
    .replace(/@font-face\s*\{[^}]*\}/gi, "")
    .replace(/url\((?!\s*['"]?#)[^)]+\)/gi, "none")
    .replace(/"Space Grotesk"|Space Grotesk|ui-sans-serif|system-ui/gi, PNG_EXPORT_FONT);
}

function isExternalReference(value: string) {
  const trimmed = value.trim();
  if (!trimmed || trimmed.startsWith("#")) return false;
  if (/^url\(\s*['"]?#/i.test(trimmed)) return false;
  return /^(https?:|data:|blob:|file:|\/\/)/i.test(trimmed) || /^url\(/i.test(trimmed);
}

function prepareSvg(svg: string, options: { rasterSafe?: boolean } = {}) {
  const documentNode = new DOMParser().parseFromString(svg, "image/svg+xml");
  const element = documentNode.documentElement;
  if (element.nodeName.toLowerCase() === "parsererror") {
    throw new Error("Unable to serialize chart.");
  }
  element.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  element.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  const viewBox = (element.getAttribute("viewBox") || "0 0 1600 900")
    .split(/\s+/)
    .map(Number);
  const width = viewBox[2] > 0 ? viewBox[2] : 1600;
  const height = viewBox[3] > 0 ? viewBox[3] : 900;
  element.setAttribute("width", String(Math.ceil(width)));
  element.setAttribute("height", String(Math.ceil(height)));

  if (options.rasterSafe) {
    element.querySelectorAll("script, iframe, object, embed, image").forEach((node) => node.remove());
    element.querySelectorAll("style").forEach((style) => {
      style.textContent = stripUnsafeCssReferences(style.textContent ?? "");
    });
    element.querySelectorAll("*").forEach((node) => {
      Array.from(node.attributes).forEach((attribute) => {
        if (/^on/i.test(attribute.name)) {
          node.removeAttribute(attribute.name);
          return;
        }
        if ((attribute.name === "href" || attribute.name === "xlink:href") && isExternalReference(attribute.value)) {
          node.removeAttribute(attribute.name);
          return;
        }
        if (attribute.name === "style") {
          node.setAttribute("style", stripUnsafeCssReferences(attribute.value));
          return;
        }
        if (/font-family/i.test(attribute.name)) {
          node.setAttribute(attribute.name, PNG_EXPORT_FONT);
        }
      });
    });

    const exportStyle = documentNode.createElementNS("http://www.w3.org/2000/svg", "style");
    exportStyle.textContent = `*{font-family:${PNG_EXPORT_FONT}!important}`;
    element.insertBefore(exportStyle, element.firstChild);
  }

  return {
    svg: new XMLSerializer().serializeToString(element),
    width,
    height
  };
}

interface MermaidChartViewerProps {
  charts: AnalysisChart[];
  title: string;
  repairingChartId: string | null;
  onRenderError: (chart: AnalysisChart, message: string, code: string) => void;
}

export function MermaidChartViewer({
  charts,
  title,
  repairingChartId,
  onRenderError
}: MermaidChartViewerProps) {
  const [activeId, setActiveId] = useState(charts[0]?.id ?? "");
  const [svgCache, setSvgCache] = useState<Record<string, string>>({});
  const [renderingId, setRenderingId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const reportedRef = useRef<string | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);
  const activeChart = charts.find((chart) => chart.id === activeId) ?? charts[0] ?? null;
  const cacheKey = activeChart ? `${activeChart.id}:${activeChart.mermaid_code}` : "";
  const svg = cacheKey ? svgCache[cacheKey] ?? "" : "";
  const renderPrefix = useMemo(() => Math.random().toString(36).slice(2), []);

  useEffect(() => {
    if (!charts.some((chart) => chart.id === activeId)) {
      setActiveId(charts[0]?.id ?? "");
    }
  }, [activeId, charts]);

  const fitViewport = useCallback(() => {
    const surface = surfaceRef.current;
    if (!surface || !svg) return;
    const chartSvg = surface.querySelector<SVGSVGElement>("svg");
    if (!chartSvg) return;
    const box = chartSvg.viewBox.baseVal;
    const width = box.width || chartSvg.getBoundingClientRect().width || 1600;
    const height = box.height || chartSvg.getBoundingClientRect().height || 900;
    const rect = surface.getBoundingClientRect();
    const nextZoom = Math.min(1, Math.max(0.2, Math.min((rect.width - 32) / width, (rect.height - 32) / height)));
    setZoom(nextZoom);
    setPan({
      x: (rect.width - width * nextZoom) / 2,
      y: (rect.height - height * nextZoom) / 2
    });
  }, [svg]);

  useEffect(() => {
    if (!activeChart || svg || repairingChartId === activeChart.id) return;
    let cancelled = false;
    const code = activeChart.mermaid_code.trim();
    const reportKey = `${activeChart.id}:${code}`;
    setRenderingId(activeChart.id);
    setDownloadError(null);
    void getMermaid()
      .then(async (mermaid) => {
        await mermaid.parse(code);
        return mermaid.render(`rawabit-analysis-${renderPrefix}-${activeChart.id}-${Date.now()}`, code);
      })
      .then(({ svg: rendered }) => {
        if (!cancelled) {
          reportedRef.current = null;
          setSvgCache((previous) => ({ ...previous, [cacheKey]: rendered }));
        }
      })
      .catch((error: unknown) => {
        if (!cancelled && reportedRef.current !== reportKey) {
          reportedRef.current = reportKey;
          onRenderError(
            activeChart,
            error instanceof Error ? error.message : String(error || "Mermaid render error"),
            code
          );
        }
      })
      .finally(() => {
        if (!cancelled) setRenderingId(null);
      });
    return () => {
      cancelled = true;
    };
  }, [activeChart, cacheKey, onRenderError, renderPrefix, repairingChartId, svg]);

  useEffect(() => {
    if (!svg) return;
    const frame = window.requestAnimationFrame(fitViewport);
    return () => window.cancelAnimationFrame(frame);
  }, [fitViewport, svg]);

  const handleWheel = useCallback(
    (event: WheelEvent<HTMLDivElement>) => {
      if (!svg) return;
      event.preventDefault();
      const rect = event.currentTarget.getBoundingClientRect();
      const nextZoom = Math.min(3.5, Math.max(0.2, zoom * (event.deltaY < 0 ? 1.12 : 0.88)));
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const chartX = (pointerX - pan.x) / zoom;
      const chartY = (pointerY - pan.y) / zoom;
      setZoom(nextZoom);
      setPan({ x: pointerX - chartX * nextZoom, y: pointerY - chartY * nextZoom });
    },
    [pan.x, pan.y, svg, zoom]
  );

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!svg || event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: pan.x,
        originY: pan.y
      };
      setDragging(true);
    },
    [pan.x, pan.y, svg]
  );

  const handlePointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    setPan({
      x: drag.originX + event.clientX - drag.startX,
      y: drag.originY + event.clientY - drag.startY
    });
  }, []);

  const handlePointerEnd = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setDragging(false);
  }, []);

  const handleDownloadSvg = useCallback(() => {
    if (!svg || !activeChart) return;
    try {
      setDownloadError(null);
      const prepared = prepareSvg(svg);
      downloadBlob(
        new Blob([prepared.svg], { type: "image/svg+xml;charset=utf-8" }),
        `${sanitizeFilename(title)}-${sanitizeFilename(activeChart.title)}.svg`
      );
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Unable to download chart.");
    }
  }, [activeChart, svg, title]);

  const handleDownloadPng = useCallback(async () => {
    if (!svg || !activeChart) return;
    setDownloadError(null);
    try {
      const prepared = prepareSvg(svg, { rasterSafe: true });
      const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(prepared.svg)}`;
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("Unable to rasterize chart."));
        image.src = url;
      });
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(prepared.width * scale);
      canvas.height = Math.ceil(prepared.height * scale);
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Unable to prepare PNG export.");
      context.fillStyle = "#fbf8f3";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.scale(scale, scale);
      context.drawImage(image, 0, 0, prepared.width, prepared.height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Unable to create PNG export.")), "image/png");
      });
      downloadBlob(blob, `${sanitizeFilename(title)}-${sanitizeFilename(activeChart.title)}.png`);
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Unable to download chart.");
    }
  }, [activeChart, svg, title]);

  const busy = Boolean(activeChart && (renderingId === activeChart.id || repairingChartId === activeChart.id));

  const renderSurface = (expandedSurface = false) => (
    <div
      ref={surfaceRef}
      className={[
        "relative min-h-0 flex-1 overflow-hidden select-none",
        svg ? (dragging ? "cursor-grabbing" : "cursor-grab") : "",
        expandedSurface ? "rounded-xl border bg-background/70" : ""
      ].join(" ")}
      style={{ touchAction: "none" }}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
    >
      {busy && !svg ? (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-2 rounded-md border bg-card px-3 py-2 text-sm text-muted-foreground shadow-sm">
            <Loader2 className="h-4 w-4 animate-spin" />
            Working on diagram...
          </div>
        </div>
      ) : null}
      {svg ? (
        <div
          className="analysis-mermaid pointer-events-none absolute left-0 top-0 min-w-max origin-top-left [&_*]:select-none [&_svg]:h-auto [&_svg]:max-w-none"
          style={{
            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
            transformOrigin: "top left"
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : null}
      {!busy && !svg ? (
        <div className="flex h-full min-h-[20rem] items-center justify-center text-sm text-muted-foreground">
          No chart is available.
        </div>
      ) : null}
    </div>
  );

  return (
    <>
      <div className="flex h-full min-h-[24rem] flex-col rounded-xl border bg-background/60">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
          <div className="flex min-w-0 flex-wrap gap-1" role="tablist" aria-label="Analysis charts">
            {charts.map((chart) => (
              <button
                key={chart.id}
                type="button"
                role="tab"
                aria-selected={chart.id === activeChart?.id}
                className={[
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  chart.id === activeChart?.id
                    ? "bg-primary text-primary-foreground"
                    : "border bg-card text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                ].join(" ")}
                onClick={() => setActiveId(chart.id)}
              >
                {chart.title}
              </button>
            ))}
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" className="h-8 w-8" disabled={!svg} aria-label="Download chart" title="Download chart">
                  <Download className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={handleDownloadSvg}>Download SVG</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => void handleDownloadPng()}>Download PNG</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => setZoom((value) => Math.min(3.5, value + 0.15))} disabled={!svg} aria-label="Zoom in" title="Zoom in">
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => setZoom((value) => Math.max(0.2, value - 0.15))} disabled={!svg} aria-label="Zoom out" title="Zoom out">
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" className="h-8 w-8" onClick={fitViewport} disabled={!svg} aria-label="Reset view" title="Reset view">
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => setExpanded(true)} disabled={!svg} aria-label="Expand chart" title="Expand chart">
              <Expand className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {downloadError ? (
          <div className="border-b border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {downloadError}
          </div>
        ) : null}
        {renderSurface()}
      </div>

      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="grid h-[92vh] max-w-[96vw] grid-rows-[auto_minmax(0,1fr)] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{activeChart?.title ?? title}</DialogTitle>
            <DialogDescription>Drag to pan and use the mouse wheel to zoom.</DialogDescription>
          </DialogHeader>
          {renderSurface(true)}
        </DialogContent>
      </Dialog>
    </>
  );
}
