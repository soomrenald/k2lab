import { useRef, useState } from "react";
import { Icon } from "./Icon";

export type StudioMode = "generation" | "edit" | "face";
export type RegionLayer = "generation" | "reference" | "targets";

export interface RegionBox {
  id: string;
  name: string;
  layer: RegionLayer;
  x: number;
  y: number;
  width: number;
  height: number;
  prompt: string;
  enabled: boolean;
}

type ResizeEdge = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface DragState {
  kind: "draw" | "move" | "resize";
  regionId?: string;
  edge?: ResizeEdge;
  startX: number;
  startY: number;
  initial?: RegionBox;
}

interface Props {
  mode: StudioMode;
  activeLayer: RegionLayer;
  sourceUrl: string | null;
  sourceName: string;
  resultUrl: string | null;
  regions: RegionBox[];
  selectedId: string | null;
  drawMode: boolean;
  comparePosition: number;
  onComparePosition: (value: number) => void;
  onSelect: (id: string | null) => void;
  onRegions: (regions: RegionBox[]) => void;
  onDrawMode: (value: boolean) => void;
  onLoadImage: (file: File) => void;
}

const canvasSize = 1000;
const minimumSize = 24;

export function RegionCanvas({
  mode,
  activeLayer,
  sourceUrl,
  sourceName,
  resultUrl,
  regions,
  selectedId,
  drawMode,
  comparePosition,
  onComparePosition,
  onSelect,
  onRegions,
  onDrawMode,
  onLoadImage,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);

  const visibleRegions = mode === "face"
    ? []
    : regions.filter((region) => region.layer === activeLayer);

  function point(event: React.PointerEvent<SVGSVGElement | SVGElement>) {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvasSize, (event.clientX - rect.left) / rect.width * canvasSize)),
      y: Math.max(0, Math.min(canvasSize, (event.clientY - rect.top) / rect.height * canvasSize)),
    };
  }

  function beginDraw(event: React.PointerEvent<SVGSVGElement>) {
    if (!drawMode || mode === "face" || event.target !== event.currentTarget) return;
    const start = point(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    const region: RegionBox = {
      id: crypto.randomUUID(),
      name: `Region ${regions.filter((item) => item.layer === activeLayer).length + 1}`,
      layer: activeLayer,
      x: start.x,
      y: start.y,
      width: 1,
      height: 1,
      prompt: "",
      enabled: true,
    };
    onRegions([...regions, region]);
    onSelect(region.id);
    setDrag({ kind: "draw", regionId: region.id, startX: start.x, startY: start.y });
  }

  function beginMove(event: React.PointerEvent<SVGRectElement>, region: RegionBox) {
    event.stopPropagation();
    onSelect(region.id);
    if (drawMode || selectedId !== region.id) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({ kind: "move", regionId: region.id, startX: start.x, startY: start.y, initial: region });
  }

  function beginResize(event: React.PointerEvent<SVGRectElement>, region: RegionBox, edge: ResizeEdge) {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({ kind: "resize", regionId: region.id, edge, startX: start.x, startY: start.y, initial: region });
  }

  function movePointer(event: React.PointerEvent<SVGSVGElement>) {
    if (!drag || !drag.regionId) return;
    const current = point(event);
    onRegions(regions.map((region) => {
      if (region.id !== drag.regionId) return region;
      if (drag.kind === "draw") {
        return {
          ...region,
          x: Math.min(drag.startX, current.x),
          y: Math.min(drag.startY, current.y),
          width: Math.abs(current.x - drag.startX),
          height: Math.abs(current.y - drag.startY),
        };
      }
      const initial = drag.initial!;
      const dx = current.x - drag.startX;
      const dy = current.y - drag.startY;
      if (drag.kind === "move") {
        return {
          ...region,
          x: Math.max(0, Math.min(canvasSize - initial.width, initial.x + dx)),
          y: Math.max(0, Math.min(canvasSize - initial.height, initial.y + dy)),
        };
      }
      return resized(initial, drag.edge!, dx, dy);
    }));
  }

  function endPointer() {
    if (drag?.kind === "draw" && drag.regionId) {
      const region = regions.find((item) => item.id === drag.regionId);
      if (region && (region.width < minimumSize || region.height < minimumSize)) {
        onRegions(regions.filter((item) => item.id !== region.id));
        onSelect(null);
      }
      onDrawMode(false);
    }
    setDrag(null);
  }

  return (
    <div className="canvas-column">
      <div className="canvas-toolbar">
        <div className="canvas-title">
          <span className="status-dot online" />
          <span>{mode === "edit" ? (activeLayer === "reference" ? "Reference layout" : "Edit targets") : mode === "face" ? "Face refinement source" : "Generation canvas"}</span>
          <small>{sourceName || "1024 × 1024"}</small>
        </div>
        <div className="canvas-actions">
          <label className="quiet-button file-button">
            <Icon name="upload" /> {sourceUrl ? "Replace image" : "Load image"}
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onLoadImage(file);
              event.target.value = "";
            }} />
          </label>
          {mode !== "face" && (
            <button className={`quiet-button ${drawMode ? "active" : ""}`} onClick={() => onDrawMode(!drawMode)}>
              <Icon name="plus" /> {drawMode ? "Drawing…" : "Draw region"}
            </button>
          )}
        </div>
      </div>
      <div className={`image-stage ${drawMode ? "drawing" : ""}`}>
        <div className="image-frame">
          {sourceUrl ? (
            <img className="canvas-image" src={sourceUrl} alt="Loaded source" draggable={false} />
          ) : (
            <div className="empty-canvas">
              <div className="empty-orbit"><Icon name={mode === "edit" ? "edit" : mode === "face" ? "face" : "spark"} /></div>
              <strong>{mode === "generation" ? "Start from an open canvas" : "Load an image to begin"}</strong>
              <span>{mode === "generation" ? "Draw regions or add a reference image" : "PNG, JPEG, or WebP"}</span>
            </div>
          )}
          {sourceUrl && resultUrl && (
            <div className="result-clip" style={{ clipPath: `inset(0 ${100 - comparePosition * 100}% 0 0)` }}>
              <img className="canvas-image result-image" src={resultUrl} alt="Generation result" draggable={false} />
            </div>
          )}
          <svg
            ref={svgRef}
            className="region-overlay"
            viewBox={`0 0 ${canvasSize} ${canvasSize}`}
            preserveAspectRatio="none"
            onPointerDown={beginDraw}
            onPointerMove={movePointer}
            onPointerUp={endPointer}
            onPointerCancel={endPointer}
            onClick={(event) => { if (!drawMode && event.target === event.currentTarget) onSelect(null); }}
          >
            {visibleRegions.map((region) => (
              <g className={`region-group ${region.id === selectedId ? "selected" : ""} ${!region.enabled ? "disabled" : ""}`} key={region.id}>
                <rect className="region-fill" x={region.x} y={region.y} width={region.width} height={region.height}
                  onPointerDown={(event) => beginMove(event, region)} />
                <rect className="region-outline" x={region.x} y={region.y} width={region.width} height={region.height} />
                <g className="region-label" transform={`translate(${region.x}, ${Math.max(0, region.y - 32)})`}>
                  <rect width={Math.max(112, region.name.length * 13 + 24)} height="28" rx="8" />
                  <text x="12" y="19">{region.name}</text>
                </g>
                {region.id === selectedId && resizeHandles(region).map((handle) => (
                  <rect key={handle.edge} className={`resize-handle edge-${handle.edge}`}
                    x={handle.x} y={handle.y} width={handle.width} height={handle.height} rx="4"
                    onPointerDown={(event) => beginResize(event, region, handle.edge)} />
                ))}
              </g>
            ))}
          </svg>
          {sourceUrl && resultUrl && comparePosition > 0 && comparePosition < 1 && (
            <div className="compare-line" style={{ left: `${comparePosition * 100}%` }} />
          )}
          {sourceUrl && resultUrl && (
            <div className="compare-control">
              <span>Source</span>
              <input type="range" min="0" max="1" step="0.01" value={comparePosition}
                onChange={(event) => onComparePosition(Number(event.target.value))} />
              <span>Result</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function resized(region: RegionBox, edge: ResizeEdge, dx: number, dy: number): RegionBox {
  let left = region.x;
  let top = region.y;
  let right = region.x + region.width;
  let bottom = region.y + region.height;
  if (edge.includes("w")) left = Math.max(0, Math.min(right - minimumSize, left + dx));
  if (edge.includes("e")) right = Math.min(canvasSize, Math.max(left + minimumSize, right + dx));
  if (edge.includes("n")) top = Math.max(0, Math.min(bottom - minimumSize, top + dy));
  if (edge.includes("s")) bottom = Math.min(canvasSize, Math.max(top + minimumSize, bottom + dy));
  return { ...region, x: left, y: top, width: right - left, height: bottom - top };
}

function resizeHandles(region: RegionBox) {
  const size = 18;
  const half = size / 2;
  const edge = 12;
  return [
    { edge: "nw" as const, x: region.x - half, y: region.y - half, width: size, height: size },
    { edge: "n" as const, x: region.x + edge, y: region.y - half, width: Math.max(1, region.width - edge * 2), height: size },
    { edge: "ne" as const, x: region.x + region.width - half, y: region.y - half, width: size, height: size },
    { edge: "e" as const, x: region.x + region.width - half, y: region.y + edge, width: size, height: Math.max(1, region.height - edge * 2) },
    { edge: "se" as const, x: region.x + region.width - half, y: region.y + region.height - half, width: size, height: size },
    { edge: "s" as const, x: region.x + edge, y: region.y + region.height - half, width: Math.max(1, region.width - edge * 2), height: size },
    { edge: "sw" as const, x: region.x - half, y: region.y + region.height - half, width: size, height: size },
    { edge: "w" as const, x: region.x - half, y: region.y + edge, width: size, height: Math.max(1, region.height - edge * 2) },
  ];
}
