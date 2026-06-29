import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { LocateFixed, ZoomIn, ZoomOut } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { GraphEdge, GraphNode, HighlightRelationship } from "@/lib/api";

type SelectedEdge = {
  id: string;
  src_id: string;
  tgt_id: string;
};

export type GraphLegendSelection =
  | {
      kind: "node";
      value: string;
    }
  | {
      kind: "edge";
      value: string;
    }
  | null;

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  className?: string;
  selectedNodeId: string | null;
  selectedEdge: SelectedEdge | null;
  highlightedNodeIds: string[];
  highlightedEdges: HighlightRelationship[];
  legendSelection: GraphLegendSelection;
  showLabels: boolean;
  multiSelectedNodeIds: Set<string>;
  multiSelectedEdgeKeys: Set<string>;
  onNodeSelect: (node: GraphNode) => void;
  onEdgeSelect: (edge: GraphEdge) => void;
  onCanvasBackgroundSelect: () => void;
  onLegendSelect: (selection: GraphLegendSelection) => void;
  onNodeMultiSelect: (node: GraphNode) => void;
  onEdgeMultiSelect: (edge: GraphEdge) => void;
};

type NodeDatum = GraphNode & {
  relationshipCount: number;
  radius: number;
  color: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

type LinkDatum = GraphEdge & {
  source: string | NodeDatum;
  target: string | NodeDatum;
  pair_key: string;
  directed_flow_hint: boolean;
};

type NodeTypeLegendItem = {
  key: string;
  label: string;
  color: string;
  count: number;
};

type EdgeTypeLegendItem = {
  key: string;
  label: string;
  count: number;
  directedCount: number;
  color: string;
};

const MIN_ZOOM = 0.35;
const MAX_ZOOM = 4;
const NODE_LABEL_FONT_SIZE = 10;
const NODE_LABEL_HALO_WIDTH = 2;
const NODE_LABEL_VISIBLE_ZOOM = 1.5;

const RELATION_COLORS = [
  "#0f766e",
  "#2563eb",
  "#7c3aed",
  "#b45309",
  "#be123c",
  "#047857",
  "#0e7490",
  "#c2410c",
  "#4338ca",
  "#525252",
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function pairKey(srcId: string, tgtId: string) {
  return `${srcId}=>${tgtId}`;
}

function colorForEntityType(entityType: string) {
  const normalized = normalizeEntityType(entityType);
  if (normalized === "person") {
    return "#d97706";
  }
  if (normalized === "organization") {
    return "#2563eb";
  }
  if (normalized === "object") {
    return "#6d28d9";
  }
  if (normalized === "location") {
    return "#16a34a";
  }
  if (normalized === "event") {
    return "#dc2626";
  }
  return "#64748b";
}

function emojiForEntityType(entityType: string) {
  const normalized = normalizeEntityType(entityType);
  if (normalized === "person") {
    return "👤";
  }
  if (normalized === "organization") {
    return "👥";
  }
  if (normalized === "location") {
    return "🗺️";
  }
  if (normalized === "event") {
    return "🕑";
  }
  if (normalized === "object") {
    return "📦";
  }
  return "";
}

function normalizeEntityType(entityType: string | null | undefined) {
  const normalized = (entityType ?? "").trim().toLowerCase();
  return normalized.length > 0 ? normalized : "other";
}

function formatEntityTypeLabel(entityType: string) {
  return entityType
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ") || "Other";
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function asNode(value: string | NodeDatum): NodeDatum | null {
  if (typeof value === "string") {
    return null;
  }
  return value;
}

function stableHash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function colorForRelationType(relationType: string) {
  const normalized = normalizeEntityType(relationType);
  const index = stableHash(normalized) % RELATION_COLORS.length;
  return RELATION_COLORS[index] ?? "#64748b";
}

export function GraphCanvas({
  nodes,
  edges,
  className,
  selectedNodeId,
  selectedEdge,
  highlightedNodeIds,
  highlightedEdges,
  legendSelection,
  showLabels,
  multiSelectedNodeIds,
  multiSelectedEdgeKeys,
  onNodeSelect,
  onEdgeSelect,
  onCanvasBackgroundSelect,
  onLegendSelect,
  onNodeMultiSelect,
  onEdgeMultiSelect,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const svgSelectionRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null);
  const graphRootRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const nodeSelectionRef = useRef<d3.Selection<SVGGElement, NodeDatum, SVGGElement, unknown> | null>(null);
  const linkSelectionRef = useRef<d3.Selection<SVGGElement, LinkDatum, SVGGElement, unknown> | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const simulationRef = useRef<d3.Simulation<NodeDatum, LinkDatum> | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const nodePositionCacheRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const transformCacheRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const dataSignatureRef = useRef<string>("");
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);

  const relationshipCountMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const node of nodes) {
      map.set(node.id, 0);
    }
    for (const edge of edges) {
      map.set(edge.src_id, (map.get(edge.src_id) ?? 0) + 1);
      map.set(edge.tgt_id, (map.get(edge.tgt_id) ?? 0) + 1);
    }
    return map;
  }, [nodes, edges]);

  const nodeData = useMemo<NodeDatum[]>(
    () => {
      const maxRelationships = Math.max(
        1,
        ...nodes.map((node) => relationshipCountMap.get(node.id) ?? 0)
      );
      const radiusScale = d3
        .scaleSqrt<number, number>()
        .domain([1, maxRelationships])
        .range([6, 22])
        .clamp(true);

      return nodes.map((node) => {
        const relationshipCount = relationshipCountMap.get(node.id) ?? 0;
        return {
          ...node,
          relationshipCount,
          radius: relationshipCount <= 0 ? 5 : radiusScale(relationshipCount),
          color: colorForEntityType(node.entity_type),
        };
      });
    },
    [nodes, relationshipCountMap]
  );

  const nodeTypeLegendItems = useMemo<NodeTypeLegendItem[]>(() => {
    const counts = new Map<string, number>();
    nodes.forEach((node) => {
      const key = normalizeEntityType(node.entity_type);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()]
      .map(([key, count]) => ({
        key,
        label: formatEntityTypeLabel(key),
        color: colorForEntityType(key),
        count,
      }))
      .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
  }, [nodes]);

  const edgeTypeLegendItems = useMemo<EdgeTypeLegendItem[]>(() => {
    const counts = new Map<string, number>();
    edges.forEach((edge) => {
      const key = (edge.relation_type || edge.label || "ASSOCIATED_WITH").trim() || "ASSOCIATED_WITH";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()]
      .map(([key, count]) => ({
        key,
        label: formatEntityTypeLabel(key),
        count,
        directedCount: 0,
        color: colorForRelationType(key),
      }))
      .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))
      .slice(0, 12);
  }, [edges]);

  const linkData = useMemo<LinkDatum[]>(
    () =>
      edges.map((edge) => ({
        ...edge,
        source: edge.src_id,
        target: edge.tgt_id,
        pair_key: pairKey(edge.src_id, edge.tgt_id),
        directed_flow_hint: false,
      })),
    [edges]
  );

  const highlightedNodeSet = useMemo(() => new Set(highlightedNodeIds), [highlightedNodeIds]);
  const highlightedEdgeIdSet = useMemo(
    () => new Set(highlightedEdges.map((item) => item.edge_id).filter((value): value is string => Boolean(value))),
    [highlightedEdges]
  );
  const highlightedEdgePairSet = useMemo(
    () =>
      new Set(
        highlightedEdges.flatMap((item) => [
          pairKey(item.src_id, item.tgt_id),
          pairKey(item.tgt_id, item.src_id),
        ])
      ),
    [highlightedEdges]
  );
  const hasHighlight = highlightedNodeSet.size > 0 || highlightedEdgePairSet.size > 0;
  const hasMultiSelect = multiSelectedNodeIds.size > 0 || multiSelectedEdgeKeys.size > 0;
  const selectedEdgeKey = selectedEdge ? pairKey(selectedEdge.src_id, selectedEdge.tgt_id) : null;
  const dataSignature = useMemo(() => {
    const nodeIds = nodes.map((node) => node.id).sort();
    const edgeIds = edges
      .map((edge) => `${edge.id}|${edge.src_id}|${edge.tgt_id}|${edge.relation_type}`)
      .sort();
    return `${nodeIds.join(",")}::${edgeIds.join(",")}`;
  }, [edges, nodes]);

  const hideTooltip = useCallback(() => {
    const tooltip = tooltipRef.current;
    if (!tooltip) {
      return;
    }
    tooltip.style.opacity = "0";
  }, []);

  const moveTooltip = useCallback((event: MouseEvent) => {
    const tooltip = tooltipRef.current;
    const container = containerRef.current;
    if (!tooltip || !container) {
      return;
    }
    const rect = container.getBoundingClientRect();
    const tooltipWidth = tooltip.offsetWidth || 180;
    const tooltipHeight = tooltip.offsetHeight || 56;
    const maxLeft = Math.max(4, rect.width - tooltipWidth - 8);
    const maxTop = Math.max(4, rect.height - tooltipHeight - 8);
    const left = clamp(event.clientX - rect.left + 12, 4, maxLeft);
    const top = clamp(event.clientY - rect.top - tooltipHeight - 10, 4, maxTop);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }, []);

  const showTooltip = useCallback(
    (html: string, event: MouseEvent) => {
      const tooltip = tooltipRef.current;
      if (!tooltip) {
        return;
      }
      tooltip.innerHTML = html;
      tooltip.style.opacity = "1";
      moveTooltip(event);
    },
    [moveTooltip]
  );

  const fitToView = useCallback(() => {
    const root = graphRootRef.current?.node();
    const svg = svgSelectionRef.current;
    const zoomBehavior = zoomBehaviorRef.current;
    if (!root || !svg || !zoomBehavior || viewport.width <= 0 || viewport.height <= 0) {
      return;
    }
    const bbox = root.getBBox();
    if (!Number.isFinite(bbox.width) || !Number.isFinite(bbox.height) || bbox.width <= 0 || bbox.height <= 0) {
      return;
    }
    const padding = 26;
    const scale = clamp(
      Math.min(
        (viewport.width - padding * 2) / Math.max(1, bbox.width),
        (viewport.height - padding * 2) / Math.max(1, bbox.height)
      ),
      MIN_ZOOM,
      MAX_ZOOM
    );
    const translateX = viewport.width / 2 - (bbox.x + bbox.width / 2) * scale;
    const translateY = viewport.height / 2 - (bbox.y + bbox.height / 2) * scale;
    const transform = d3.zoomIdentity.translate(translateX, translateY).scale(scale);
    svg.transition().duration(260).call(zoomBehavior.transform as any, transform);
  }, [viewport.height, viewport.width]);

  const adjustZoom = useCallback((multiplier: number) => {
    const svg = svgSelectionRef.current;
    const zoomBehavior = zoomBehaviorRef.current;
    if (!svg || !zoomBehavior) {
      return;
    }
    svg.transition().duration(160).call(zoomBehavior.scaleBy as any, multiplier);
  }, []);

  const applyVisualState = useCallback(() => {
    const nodeSelection = nodeSelectionRef.current;
    const linkSelection = linkSelectionRef.current;
    if (!nodeSelection || !linkSelection) {
      return;
    }

    linkSelection
      .select<SVGLineElement>("line.link-visible")
      .attr("stroke", (link) => {
        const multiSelected = multiSelectedEdgeKeys.has(link.pair_key);
        const highlighted =
          highlightedEdgeIdSet.has(link.id) || highlightedEdgePairSet.has(link.pair_key);
        const selected =
          (selectedEdge?.id && selectedEdge.id === link.id) ||
          (selectedEdgeKey !== null && selectedEdgeKey === link.pair_key);
        if (multiSelected) {
          return "#7e22ce";
        }
        if (selected) {
          return "#b91c1c";
        }
        if (highlighted) {
          return "#d97706";
        }
        return "#334155";
      })
      .attr("stroke-opacity", (link) => {
        const multiSelected = multiSelectedEdgeKeys.has(link.pair_key);
        const highlighted =
          highlightedEdgeIdSet.has(link.id) || highlightedEdgePairSet.has(link.pair_key);
        const selected =
          (selectedEdge?.id && selectedEdge.id === link.id) ||
          (selectedEdgeKey !== null && selectedEdgeKey === link.pair_key);
        if (multiSelected) {
          return 0.95;
        }
        if (hasHighlight && !highlighted && !selected) {
          return 0.2;
        }
        return selected ? 0.98 : highlighted ? 0.92 : 0.74;
      })
      .attr("stroke-width", (link) => {
        const multiSelected = multiSelectedEdgeKeys.has(link.pair_key);
        const highlighted =
          highlightedEdgeIdSet.has(link.id) || highlightedEdgePairSet.has(link.pair_key);
        const selected =
          (selectedEdge?.id && selectedEdge.id === link.id) ||
          (selectedEdgeKey !== null && selectedEdgeKey === link.pair_key);
        if (multiSelected) {
          return 2.8;
        }
        if (selected) {
          return 3.1;
        }
        if (highlighted) {
          return 2.5;
        }
        return 1.6;
      })
      .attr("stroke-dasharray", (link) => {
        const multiSelected = multiSelectedEdgeKeys.has(link.pair_key);
        const highlighted =
          highlightedEdgeIdSet.has(link.id) || highlightedEdgePairSet.has(link.pair_key);
        if (multiSelected) {
          return "4 3";
        }
        if (highlighted) {
          return "5 4";
        }
        return null;
      })
      .attr("marker-end", null);

    const nodeOpacity = (node: NodeDatum) => {
        if (multiSelectedNodeIds.has(node.id)) {
          return 1;
        }
        if (hasHighlight && node.id !== selectedNodeId && !highlightedNodeSet.has(node.id)) {
          return 0.26;
        }
        return 1;
      };

    nodeSelection
      .select<SVGCircleElement>("circle.node-shape")
      .attr("fill", (node) => {
        if (multiSelectedNodeIds.has(node.id)) {
          return "#7e22ce";
        }
        if (node.id === selectedNodeId) {
          return "#b91c1c";
        }
        if (highlightedNodeSet.has(node.id)) {
          return "#f59e0b";
        }
        return node.color;
      })
      .attr("opacity", nodeOpacity)
      .attr("stroke", (node) => {
        if (multiSelectedNodeIds.has(node.id)) {
          return "#581c87";
        }
        return node.id === selectedNodeId ? "#7f1d1d" : "#0f172a";
      })
      .attr("stroke-width", (node) => {
        if (multiSelectedNodeIds.has(node.id)) {
          return 2.4;
        }
        if (node.id === selectedNodeId) {
          return 2.4;
        }
        return highlightedNodeSet.has(node.id) ? 1.8 : 1;
      })
      .attr("stroke-dasharray", (node) => {
        if (multiSelectedNodeIds.has(node.id)) {
          return "4 3";
        }
        return null;
      });

    nodeSelection
      .select<SVGTextElement>("text.node-emoji")
      .attr("opacity", nodeOpacity);

    nodeSelection
      .select<SVGTextElement>("text.node-label")
      .style("display", (node) => {
        if (node.id === selectedNodeId || highlightedNodeSet.has(node.id) || multiSelectedNodeIds.has(node.id)) {
          return "block";
        }
        return zoom >= NODE_LABEL_VISIBLE_ZOOM ? "block" : "none";
      })
      .attr("transform", (node) => {
        const labelScale = 1 / clamp(zoom, 1, MAX_ZOOM);
        const x = node.radius + 7 * labelScale;
        return `translate(${x}, 0) scale(${labelScale})`;
      });
  }, [
    hasHighlight,
    highlightedEdgeIdSet,
    highlightedEdgePairSet,
    highlightedNodeSet,
    multiSelectedEdgeKeys,
    multiSelectedNodeIds,
    selectedEdge,
    selectedEdgeKey,
    selectedNodeId,
    zoom,
  ]);
  const applyVisualStateRef = useRef(applyVisualState);
  useEffect(() => {
    applyVisualStateRef.current = applyVisualState;
  }, [applyVisualState]);

  useEffect(() => {
    const linkSelection = linkSelectionRef.current;
    if (!linkSelection) {
      return;
    }
    linkSelection
      .select<SVGTextElement>("text.edge-label")
      .style("display", showLabels ? "block" : "none");
  }, [showLabels]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      setViewport({
        width: Math.floor(entry.contentRect.width),
        height: Math.floor(entry.contentRect.height),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || tooltipRef.current) {
      return;
    }
    const tooltip = document.createElement("div");
    tooltip.style.position = "absolute";
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    tooltip.style.opacity = "0";
    tooltip.style.pointerEvents = "none";
    tooltip.style.zIndex = "30";
    tooltip.style.padding = "8px 10px";
    tooltip.style.borderRadius = "8px";
    tooltip.style.border = "1px solid rgba(148, 163, 184, 0.35)";
    tooltip.style.background = "rgba(15, 23, 42, 0.93)";
    tooltip.style.color = "#e2e8f0";
    tooltip.style.fontSize = "12px";
    tooltip.style.lineHeight = "1.35";
    tooltip.style.maxWidth = "320px";
    tooltip.style.boxShadow = "0 8px 20px rgba(2, 6, 23, 0.45)";
    tooltip.style.transition = "opacity 80ms linear";
    container.appendChild(tooltip);
    tooltipRef.current = tooltip;
    return () => {
      tooltip.remove();
      tooltipRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!svgRef.current || viewport.width <= 0 || viewport.height <= 0) {
      return;
    }
    const shouldAutoFit = dataSignatureRef.current !== dataSignature;
    dataSignatureRef.current = dataSignature;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    hideTooltip();

    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "rawabit-flow-arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 12)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L9,0L0,4")
      .attr("fill", "#334155");

    const graphRoot = svg.append("g");

    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([MIN_ZOOM, MAX_ZOOM])
      .on("zoom", (event) => {
        graphRoot.attr("transform", event.transform.toString());
        transformCacheRef.current = event.transform;
        setZoom(clamp(event.transform.k, MIN_ZOOM, MAX_ZOOM));
      });

    svg.call(zoomBehavior as any);
    const initialTransform = shouldAutoFit
      ? d3.zoomIdentity.translate(viewport.width / 2, viewport.height / 2).scale(1)
      : transformCacheRef.current;
    svg.call(zoomBehavior.transform as any, initialTransform);

    const simulationNodes: NodeDatum[] = nodeData.map((node): NodeDatum => {
      const cached = nodePositionCacheRef.current.get(node.id);
      if (!cached) {
        const seed = stableHash(node.id);
        const angle = (seed % 360) * (Math.PI / 180);
        const radius = 20 + (seed % 80);
        return {
          ...node,
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          vx: 0,
          vy: 0,
        };
      }
      return {
        ...node,
        x: cached.x,
        y: cached.y,
        vx: 0,
        vy: 0,
      };
    });
    const nodeById = new Map(simulationNodes.map((node) => [node.id, node]));
    const simulationLinks: LinkDatum[] = linkData.map((link) => ({
      ...link,
      source: nodeById.get(link.src_id) ?? link.src_id,
      target: nodeById.get(link.tgt_id) ?? link.tgt_id,
    }));
    const linkSelection = graphRoot
      .append("g")
      .attr("class", "graph-links")
      .selectAll<SVGGElement, LinkDatum>("g")
      .data(simulationLinks, (link) => link.id)
      .join("g");

    linkSelection.append("line").attr("class", "link-visible");
    linkSelection
      .append("line")
      .attr("class", "link-hit")
      .attr("stroke", "transparent")
      .attr("stroke-width", 14)
      .style("cursor", "pointer");
    linkSelection
      .append("text")
      .attr("class", "edge-label")
      .attr("font-size", 9)
      .attr("fill", "#334155")
      .attr("paint-order", "stroke")
      .attr("stroke", "rgba(248, 250, 252, 0.94)")
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .style("pointer-events", "none")
      .style("user-select", "none")
      .style("display", "none")
      .text((link) => link.relation_type.replace(/_/g, " "));

    const nodeSelection = graphRoot
      .append("g")
      .attr("class", "graph-nodes")
      .selectAll<SVGGElement, NodeDatum>("g")
      .data(simulationNodes, (node) => node.id)
      .join("g")
      .style("cursor", "pointer");

    nodeSelection
      .append("circle")
      .attr("class", "node-shape")
      .attr("r", (node) => node.radius);

    nodeSelection
      .append("text")
      .attr("class", "node-emoji")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("font-size", (node) => `${Math.max(7, node.radius * 0.85)}px`)
      .style("pointer-events", "none")
      .style("user-select", "none")
      .text((node) => emojiForEntityType(node.entity_type));

    nodeSelection
      .append("text")
      .attr("class", "node-label")
      .attr("x", 0)
      .attr("y", 4)
      .attr("font-size", NODE_LABEL_FONT_SIZE)
      .attr("fill", "#0f172a")
      .attr("paint-order", "stroke")
      .attr("stroke", "rgba(248, 250, 252, 0.95)")
      .attr("stroke-width", NODE_LABEL_HALO_WIDTH)
      .attr("stroke-linejoin", "round")
      .style("pointer-events", "none")
      .style("user-select", "none")
      .text((node) => node.label);

    const simulation = d3
      .forceSimulation<NodeDatum>(simulationNodes)
      .force(
        "link",
        d3
          .forceLink<NodeDatum, LinkDatum>(simulationLinks)
          .id((node) => node.id)
          .distance(28)
          .strength(0.75)
      )
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(0, 0))
      .force(
        "x",
        d3.forceX<NodeDatum>(() => 0).strength(0.08)
      )
      .force(
        "y",
        d3.forceY<NodeDatum>(() => 0).strength(0.08)
      )
      .force("collision", d3.forceCollide<NodeDatum>().radius((node) => node.radius + 6).iterations(6))
      .alphaDecay(0.018)
      .alphaMin(0.02);
    const renderTick = () => {
      linkSelection
        .select<SVGLineElement>("line.link-visible")
        .attr("x1", (link) => asNode(link.source)?.x ?? 0)
        .attr("y1", (link) => asNode(link.source)?.y ?? 0)
        .attr("x2", (link) => asNode(link.target)?.x ?? 0)
        .attr("y2", (link) => asNode(link.target)?.y ?? 0);

      linkSelection
        .select<SVGLineElement>("line.link-hit")
        .attr("x1", (link) => asNode(link.source)?.x ?? 0)
        .attr("y1", (link) => asNode(link.source)?.y ?? 0)
        .attr("x2", (link) => asNode(link.target)?.x ?? 0)
        .attr("y2", (link) => asNode(link.target)?.y ?? 0);

      linkSelection
        .select<SVGTextElement>("text.edge-label")
        .attr("x", (link) => {
          const source = asNode(link.source);
          const target = asNode(link.target);
          return ((source?.x ?? 0) + (target?.x ?? 0)) / 2;
        })
        .attr("y", (link) => {
          const source = asNode(link.source);
          const target = asNode(link.target);
          return ((source?.y ?? 0) + (target?.y ?? 0)) / 2 - 4;
        });

      nodeSelection.attr("transform", (node) => `translate(${node.x ?? 0}, ${node.y ?? 0})`);
      for (const node of simulationNodes) {
        if (typeof node.x === "number" && typeof node.y === "number") {
          nodePositionCacheRef.current.set(node.id, { x: node.x, y: node.y });
        }
      }
    };
    simulation.on("tick", renderTick);
    simulation.on("end", () => {
      if (shouldAutoFit) {
        fitToView();
      }
    });
    if (!shouldAutoFit) {
      // Preserve the existing layout on interaction-driven rerenders.
      simulationNodes.forEach((node) => {
        node.vx = 0;
        node.vy = 0;
      });
      simulation.stop();
      renderTick();
    }

    nodeSelection.call(
      d3
        .drag<SVGGElement, NodeDatum>()
        .on("start", (event, node) => {
          if (!event.active) {
            simulation.alphaTarget(0.2).restart();
          }
          node.fx = node.x ?? 0;
          node.fy = node.y ?? 0;
        })
        .on("drag", (event, node) => {
          node.fx = event.x;
          node.fy = event.y;
        })
        .on("end", (event, node) => {
          if (!event.active) {
            simulation.alphaTarget(0);
          }
          node.fx = null;
          node.fy = null;
        }) as any
    );

    nodeSelection
      .on("mouseover", (event, node) => {
        const html = [
          `<strong>${escapeHtml(node.label)}</strong>`,
          `Type: ${escapeHtml(node.entity_type || "Other")}`,
          `Relationships: ${node.relationshipCount}`,
        ].join("<br/>");
        showTooltip(html, event as MouseEvent);
      })
      .on("mousemove", (event) => moveTooltip(event as MouseEvent))
      .on("mouseout", hideTooltip)
      .on("click", (event, node) => {
        event.stopPropagation();
        if (event.ctrlKey || event.metaKey) {
          onNodeMultiSelect(node);
        } else {
          onNodeSelect(node);
        }
      });

    linkSelection
      .select<SVGLineElement>("line.link-hit")
      .on("mouseover", (event, link) => {
        const html = [
          `<strong>${escapeHtml(link.relation_type || "ASSOCIATED_WITH")}</strong>`,
          `${escapeHtml(link.src_id)} &rarr; ${escapeHtml(link.tgt_id)}`,
          escapeHtml(link.label || "No relationship description"),
        ].join("<br/>");
        showTooltip(html, event as MouseEvent);
      })
      .on("mousemove", (event) => moveTooltip(event as MouseEvent))
      .on("mouseout", hideTooltip)
      .on("click", (event, link) => {
        event.stopPropagation();
        if (event.ctrlKey || event.metaKey) {
          onEdgeMultiSelect(link);
        } else {
          onEdgeSelect(link);
        }
      });

    svg.on("click", () => {
      hideTooltip();
      onCanvasBackgroundSelect();
    });

    svgSelectionRef.current = svg;
    graphRootRef.current = graphRoot;
    nodeSelectionRef.current = nodeSelection;
    linkSelectionRef.current = linkSelection;
    zoomBehaviorRef.current = zoomBehavior;
    simulationRef.current = simulation;
    applyVisualStateRef.current();

    if (shouldAutoFit) {
      window.setTimeout(() => fitToView(), 260);
    }

    return () => {
      simulation.stop();
      hideTooltip();
      svg.on(".zoom", null);
      simulationRef.current = null;
      nodeSelectionRef.current = null;
      linkSelectionRef.current = null;
      graphRootRef.current = null;
      svgSelectionRef.current = null;
      zoomBehaviorRef.current = null;
    };
  }, [
    fitToView,
    hideTooltip,
    linkData,
    moveTooltip,
    nodeData,
    onEdgeSelect,
    onCanvasBackgroundSelect,
    onNodeSelect,
    showTooltip,
    viewport.height,
    viewport.width,
    dataSignature,
  ]);

  useEffect(() => {
    applyVisualState();
  }, [applyVisualState]);

  return (
    <div
      className={[
        "relative mt-4 h-[520px] rounded-xl border border-dashed border-muted/70 bg-background/60",
        className ?? "",
      ].join(" ").trim()}
    >
      <div className="absolute left-3 top-3 z-10 rounded-lg border bg-card/95 px-3 py-2 text-xs text-muted-foreground">
        <p>Scroll to zoom, drag nodes/canvas, click node/edge for details, Ctrl+click to select for context.</p>
      </div>
      <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-lg border bg-card/95 p-2">
        <Button size="icon" variant="outline" onClick={() => adjustZoom(1.15)}>
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button size="icon" variant="outline" onClick={() => adjustZoom(0.88)}>
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button size="icon" variant="outline" onClick={fitToView}>
          <LocateFixed className="h-4 w-4" />
        </Button>
      </div>
      <div className="absolute bottom-3 right-3 z-10 rounded-md border bg-card/95 px-2 py-1 text-xs text-muted-foreground">
        Zoom {Math.round(zoom * 100)}%
      </div>
      {(nodeTypeLegendItems.length > 0 || edgeTypeLegendItems.length > 0) ? (
        <div className="group absolute bottom-3 left-3 z-10 w-[min(16rem,calc(100%-1.5rem))] max-h-[calc(100%-4.5rem)] overflow-hidden rounded-lg border bg-card/95 px-1.5 py-1 text-[10px] text-muted-foreground shadow-soft backdrop-blur">
          <div className="flex items-center justify-between gap-1 border-b border-border/60 pb-0.5">
            <div className="min-w-0">
              <p className="font-medium uppercase tracking-[0.12em] text-foreground text-[9px]">Legend</p>
            </div>
            {legendSelection ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-4 px-1 text-[8px]"
                onClick={() => onLegendSelect(null)}
              >
                Clear
              </Button>
            ) : null}
          </div>
          <div className="mt-1 grid gap-1 overflow-y-auto pr-0.5">
            {nodeTypeLegendItems.length > 0 ? (
              <div className="grid gap-0.5">
                <div className="hidden group-hover:flex items-center gap-1 text-[8px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
                  <span className="h-px w-1.5 bg-muted-foreground/50" />
                  Nodes
                </div>
                <div className="flex flex-wrap gap-0.5">
                  {nodeTypeLegendItems.map((item) => {
                    const active =
                      legendSelection?.kind === "node" && legendSelection.value === item.key;
                    return (
                      <Button
                        key={item.key}
                        type="button"
                        variant="outline"
                        size="sm"
                        aria-pressed={active}
                        title={`${item.label}: ${item.count}`}
                        className={[
                          "h-4 min-w-0 gap-0.5 rounded-full px-1 text-[9px] leading-none group-hover:h-5 group-hover:px-1.5 group-hover:text-[10px]",
                          active ? "border-foreground/60 bg-background text-foreground" : "bg-background/80",
                        ].join(" ")}
                        style={{
                          borderColor: active ? item.color : undefined,
                          boxShadow: active ? `inset 0 0 0 1px ${item.color}` : undefined,
                        }}
                        onClick={() =>
                          onLegendSelect(active ? null : { kind: "node", value: item.key })
                        }
                      >
                        <span
                          className="h-1.5 w-1.5 shrink-0 rounded-full ring-1 ring-black/10 group-hover:h-2 group-hover:w-2"
                          style={{ backgroundColor: item.color }}
                          aria-hidden="true"
                        />
                        <span className="truncate font-medium">{item.label}</span>
                        <span className="font-mono text-[8px] text-muted-foreground group-hover:text-[9px]">{item.count}</span>
                      </Button>
                    );
                  })}
                </div>
              </div>
            ) : null}
            {edgeTypeLegendItems.length > 0 ? (
              <div className="grid gap-0.5 border-t border-border/60 pt-0.5">
                <div className="hidden group-hover:flex items-center gap-1 text-[8px] font-medium uppercase tracking-[0.1em] text-muted-foreground">
                  <span className="h-px w-1.5 bg-muted-foreground/50" />
                  Relationships
                </div>
                <div className="flex flex-wrap gap-0.5">
                  {edgeTypeLegendItems.map((item) => {
                    const normalizedValue = normalizeEntityType(item.key);
                    const active =
                      legendSelection?.kind === "edge" && legendSelection.value === normalizedValue;
                    return (
                      <Button
                        key={item.key}
                        type="button"
                        variant="outline"
                        size="sm"
                        aria-pressed={active}
                        title={`${item.label}: ${item.count}`}
                        className={[
                          "h-4 min-w-0 gap-0.5 rounded-full px-1 text-[9px] leading-none group-hover:h-5 group-hover:px-1.5 group-hover:text-[10px]",
                          active ? "border-foreground/60 bg-background text-foreground" : "bg-background/80",
                        ].join(" ")}
                        style={{
                          borderColor: active ? item.color : undefined,
                          boxShadow: active ? `inset 0 0 0 1px ${item.color}` : undefined,
                        }}
                        onClick={() =>
                          onLegendSelect(active ? null : { kind: "edge", value: normalizedValue })
                        }
                      >
                        <span
                          className="h-0.5 w-2 shrink-0 rounded-full group-hover:h-0.5 group-hover:w-3"
                          style={{ backgroundColor: item.color }}
                          aria-hidden="true"
                        />
                        <span className="truncate font-medium">{item.label}</span>
                        <span className="font-mono text-[8px] text-muted-foreground group-hover:text-[9px]">{item.count}</span>
                      </Button>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <div ref={containerRef} className="h-full w-full overflow-hidden">
        {nodes.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            No graph data available.
          </div>
        ) : (
          <svg ref={svgRef} className="h-full w-full" />
        )}
      </div>
    </div>
  );
}
