import { memo } from 'react'
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { PipelineNodeData } from '../../lib/pipelineGraph'
import { getCategoryColor, getHandleConfig } from '../../lib/pipelineHandles'
import { getNodeTypeDef } from '../../lib/nodeTypes'

function PipelineNodeComponent({ data, selected }: NodeProps<Node<PipelineNodeData>>) {
  const stageType = data.stageType || '—'
  const stageId = data.stageId || '—'
  const handles = getHandleConfig(stageType)
  const color = getCategoryColor(stageType)
  const category = getNodeTypeDef(stageType)?.category ?? ''

  return (
    <div
      className={`
        rounded-lg border bg-bg-elevated shadow-sm min-w-[180px] max-w-[220px]
        ${selected ? 'border-accent ring-1 ring-accent/40' : 'border-border'}
      `}
      style={{ borderTopWidth: 3, borderTopColor: color }}
    >
      {handles.inputs !== 'none' && (
        <Handle
          type="target"
          position={Position.Left}
          id="in"
          className="!w-2.5 !h-2.5 !bg-ink-muted !border-2 !border-bg-elevated"
        />
      )}
      <div className="px-3 py-2">
        <div className="text-[10px] uppercase tracking-wide text-ink-muted truncate">
          {category || 'node'}
        </div>
        <div className="font-mono text-[12px] text-ink-primary truncate" title={stageId}>
          {stageId}
        </div>
        <div className="font-mono text-[10px] text-ink-secondary truncate" title={stageType}>
          {stageType}
        </div>
      </div>
      {handles.outputs !== 'none' && (
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          className="!w-2.5 !h-2.5 !bg-accent !border-2 !border-bg-elevated"
        />
      )}
    </div>
  )
}

export const PipelineNode = memo(PipelineNodeComponent)
