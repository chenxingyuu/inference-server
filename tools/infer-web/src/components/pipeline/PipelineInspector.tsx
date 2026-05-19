import { useEffect, useState } from 'react'
import type { Edge, Node } from '@xyflow/react'
import { Field, Input, Select } from '../ui/Field'
import { useT } from '../../lib/i18n'
import { NODE_CATEGORIES, NODE_TYPE_DEFS, getNodeTypeDef } from '../../lib/nodeTypes'
import type { PipelineEdgeData, PipelineNodeData } from '../../lib/pipelineGraph'
import { pairsToWith } from '../../lib/pipelineGraph'
import type { DropPolicy } from '../../types'

const CUSTOM = '__custom__'

type WithPair = { k: string; v: string }

interface PipelineInspectorProps {
  selectedNode: Node<PipelineNodeData> | null
  selectedEdge: Edge<PipelineEdgeData> | null
  onUpdateNode: (nodeId: string, data: PipelineNodeData) => void
  onRenameNode: (oldId: string, newId: string) => void
  onUpdateEdge: (edgeId: string, data: PipelineEdgeData) => void
}

export function PipelineInspector({
  selectedNode,
  selectedEdge,
  onUpdateNode,
  onRenameNode,
  onUpdateEdge,
}: PipelineInspectorProps) {
  const { t } = useT()

  if (!selectedNode && !selectedEdge) {
    return (
      <aside className="w-64 flex-shrink-0 border-l border-border bg-bg-surface p-4">
        <p className="text-[12px] text-ink-muted">{t('pipelines.editor.inspector_empty')}</p>
      </aside>
    )
  }

  if (selectedEdge) {
    return (
      <aside className="w-64 flex-shrink-0 border-l border-border bg-bg-surface overflow-y-auto p-4 space-y-4">
        <h3 className="label">{t('pipelines.editor.inspector_edge')}</h3>
        <Field label={t('pipelines.col.from')}>
          <Input value={selectedEdge.data?.from ?? selectedEdge.source} disabled className="opacity-70" />
        </Field>
        <Field label={t('pipelines.col.to')}>
          <Input value={selectedEdge.data?.to ?? selectedEdge.target} disabled className="opacity-70" />
        </Field>
        <Field label={t('pipelines.col.capacity')}>
          <Input
            type="number"
            min={1}
            value={selectedEdge.data?.capacity ?? 32}
            onChange={(e) => {
              const capacity = Math.max(1, Number(e.target.value) || 1)
              onUpdateEdge(selectedEdge.id, {
                from: selectedEdge.data?.from ?? selectedEdge.source,
                to: selectedEdge.data?.to ?? selectedEdge.target,
                capacity,
                drop_policy: selectedEdge.data?.drop_policy ?? 'drop_oldest',
              })
            }}
          />
        </Field>
        <Field label={t('pipelines.col.drop_policy')}>
          <Select
            value={selectedEdge.data?.drop_policy ?? 'drop_oldest'}
            onChange={(e) => {
              onUpdateEdge(selectedEdge.id, {
                from: selectedEdge.data?.from ?? selectedEdge.source,
                to: selectedEdge.data?.to ?? selectedEdge.target,
                capacity: selectedEdge.data?.capacity ?? 32,
                drop_policy: e.target.value as DropPolicy,
              })
            }}
          >
            <option value="block">block</option>
            <option value="drop_oldest">drop_oldest</option>
            <option value="drop_newest">drop_newest</option>
          </Select>
        </Field>
      </aside>
    )
  }

  if (!selectedNode) return null

  return (
    <aside className="w-64 flex-shrink-0 border-l border-border bg-bg-surface overflow-y-auto p-4">
      <h3 className="label mb-3">{t('pipelines.editor.inspector_node')}</h3>
      <NodeInspectorForm
        node={selectedNode}
        onUpdateNode={onUpdateNode}
        onRenameNode={onRenameNode}
      />
    </aside>
  )
}

function NodeInspectorForm({
  node,
  onUpdateNode,
  onRenameNode,
}: {
  node: Node<PipelineNodeData>
  onUpdateNode: (nodeId: string, data: PipelineNodeData) => void
  onRenameNode: (oldId: string, newId: string) => void
}) {
  const { t } = useT()
  const data = node.data
  const [pairs, setPairs] = useState<WithPair[]>(
    () => Object.entries(data.with ?? {}).map(([k, v]) => ({ k, v })),
  )
  const [customType, setCustomType] = useState(() =>
    getNodeTypeDef(data.stageType) ? '' : data.stageType,
  )
  const [localId, setLocalId] = useState(data.stageId)

  useEffect(() => {
    setPairs(Object.entries(data.with ?? {}).map(([k, v]) => ({ k, v })))
    setCustomType(getNodeTypeDef(data.stageType) ? '' : data.stageType)
    setLocalId(data.stageId)
  }, [node.id, data.stageId, data.stageType, data.with])

  const selectValue = getNodeTypeDef(data.stageType) ? data.stageType : (data.stageType ? CUSTOM : '')

  const applyData = (patch: Partial<PipelineNodeData>) => {
    onUpdateNode(node.id, { ...data, ...patch })
  }

  const handlePairsChange = (ps: WithPair[]) => {
    setPairs(ps)
    applyData({ with: pairsToWith(ps) })
  }

  const handleTypeSelect = (val: string) => {
    if (val === CUSTOM) {
      setCustomType('')
      applyData({ stageType: '' })
      return
    }
    const def = getNodeTypeDef(val)!
    const newPairs = def.withTemplate.map((p) => ({ ...p }))
    setPairs(newPairs)
    const autoId = data.stageId || val.replaceAll('.', '_')
    setLocalId(autoId)
    applyData({
      stageType: val,
      stageId: autoId,
      with: pairsToWith(newPairs),
    })
    if (autoId !== node.id) onRenameNode(node.id, autoId)
  }

  const commitId = () => {
    const trimmed = localId.trim()
    if (trimmed && trimmed !== data.stageId) {
      onRenameNode(node.id, trimmed)
    } else {
      setLocalId(data.stageId)
    }
  }

  const updatePair = (i: number, patch: Partial<WithPair>) =>
    handlePairsChange(pairs.map((p, j) => (j === i ? { ...p, ...patch } : p)))

  return (
    <div className="space-y-3">
      <Field label={t('pipelines.col.node_id')}>
        <Input
          value={localId}
          onChange={(e) => setLocalId(e.target.value)}
          onBlur={commitId}
          onKeyDown={(e) => { if (e.key === 'Enter') commitId() }}
        />
      </Field>
      <Field label={t('pipelines.col.type')}>
        <Select value={selectValue} onChange={(e) => handleTypeSelect(e.target.value)}>
          <option value="" disabled>— {t('pipelines.col.type')} —</option>
          {NODE_CATEGORIES.map((cat) => {
            const items = NODE_TYPE_DEFS.filter((d) => d.category === cat)
            return (
              <optgroup key={cat} label={cat}>
                {items.map((d) => (
                  <option key={d.type} value={d.type}>{d.type}</option>
                ))}
              </optgroup>
            )
          })}
          <option value={CUSTOM}>{t('pipelines.col.type_custom')}</option>
        </Select>
      </Field>

      {selectValue === CUSTOM && (
        <Field label={t('pipelines.col.type')}>
          <Input
            placeholder="custom.type"
            value={customType}
            onChange={(e) => {
              setCustomType(e.target.value)
              applyData({ stageType: e.target.value })
            }}
            className="text-[11px]"
          />
        </Field>
      )}

      <div>
        <span className="label block mb-2">{t('pipelines.editor.params')}</span>
        <div className="space-y-2">
          {pairs.map((p, i) => (
            <div key={i} className="flex gap-1">
              <Input
                placeholder={t('pipelines.col.key')}
                value={p.k}
                onChange={(e) => updatePair(i, { k: e.target.value })}
                className="flex-1 text-[11px]"
              />
              <Input
                placeholder={t('pipelines.col.value')}
                value={p.v}
                onChange={(e) => updatePair(i, { v: e.target.value })}
                className="flex-1 text-[11px]"
              />
              <button
                type="button"
                onClick={() => handlePairsChange(pairs.filter((_, j) => j !== i))}
                className="btn-icon text-danger/50 hover:text-danger"
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => handlePairsChange([...pairs, { k: '', v: '' }])}
            className="text-[10px] text-ink-muted hover:text-accent"
          >
            {t('pipelines.add_param')}
          </button>
        </div>
      </div>
    </div>
  )
}
