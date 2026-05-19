import { memo } from 'react'
import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from '@xyflow/react'
import type { PipelineEdgeData } from '../../lib/pipelineGraph'

function PipelineEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}: EdgeProps<Edge<PipelineEdgeData>>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  })

  const cap = data?.capacity ?? 32
  const policy = data?.drop_policy ?? 'drop_oldest'

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: selected ? '#22d3ee' : '#465669',
          strokeWidth: selected ? 2 : 1.5,
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-none text-[9px] font-mono px-1 py-0.5 rounded bg-bg-overlay border border-border text-ink-muted"
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
        >
          {cap} · {policy}
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

export const PipelineEdge = memo(PipelineEdgeComponent)
