import { useCallback, useEffect, useRef, useState, type MutableRefObject } from 'react'
import type { Edge, Node } from '@xyflow/react'
import { Field, Input, Select } from '../ui/Field'
import { ParamKeyCombobox } from '../ui/ParamKeyCombobox'
import { ParamValueInput } from '../ui/ParamValueInput'
import { useModels } from '../../hooks/queries'
import { useT } from '../../lib/i18n'
import { NODE_CATEGORIES, NODE_TYPE_DEFS, getNodeTypeDef } from '../../lib/nodeTypes'
import {
  INFER_ENGINE_THRESHOLD_KEYS,
  isValidInferEngineThreshold,
  pairsToWith,
  type InferEngineThresholdKey,
  type PipelineEdgeData,
  type PipelineNodeData,
} from '../../lib/pipelineGraph'
import type { DropPolicy } from '../../types'

const CUSTOM = '__custom__'

type WithPair = { k: string; v: string }

interface PipelineInspectorProps {
  selectedNode: Node<PipelineNodeData> | null
  selectedEdge: Edge<PipelineEdgeData> | null
  /** Bumps on undo/redo so inspector re-syncs from graph without watching every data.with change. */
  graphRevision?: number
  onUpdateNode: (
    nodeId: string,
    data: PipelineNodeData,
    options?: { commit?: 'immediate' | 'debounced' },
  ) => void
  onRenameNode: (oldId: string, newId: string) => void
  onUpdateEdge: (edgeId: string, data: PipelineEdgeData) => void
  inspectorFlushRef?: MutableRefObject<(() => void) | null>
}

export function PipelineInspector({
  selectedNode,
  selectedEdge,
  graphRevision = 0,
  onUpdateNode,
  onRenameNode,
  onUpdateEdge,
  inspectorFlushRef,
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
        key={`${selectedNode.id}-${graphRevision}`}
        node={selectedNode}
        onUpdateNode={onUpdateNode}
        onRenameNode={onRenameNode}
        inspectorFlushRef={inspectorFlushRef}
      />
    </aside>
  )
}

function withToPairs(withRecord: Record<string, string> | undefined): WithPair[] {
  return Object.entries(withRecord ?? {}).map(([k, v]) => ({ k, v }))
}

function InferEngineThresholdFields({
  params,
  onChange,
}: {
  params: Record<string, string> | undefined
  onChange: (key: InferEngineThresholdKey, value: string) => void
}) {
  const { t } = useT()
  const { data: models } = useModels()
  const model = models?.find(({ id }) => id === params?.model_id)
  const supportsNms = model?.model_type === 'detector' && ['v5', 'v8', 'v11'].includes(model.version)
  const keys = supportsNms ? INFER_ENGINE_THRESHOLD_KEYS : (['conf_thresh'] as const)

  return (
    <details className="rounded border border-border px-3 py-2">
      <summary className="cursor-pointer text-xs text-ink-secondary">
        {t('pipelines.editor.infer_advanced')}
      </summary>
      <div className="mt-3 space-y-3">
        {keys.map((key) => {
          const label = t(key === 'conf_thresh' ? 'pipelines.editor.confidence_threshold' : 'pipelines.editor.nms_threshold')
          const modelDefault = model?.[key]
          const value = params?.[key] ?? ''
          const placeholder = typeof modelDefault === 'number'
            ? t('pipelines.editor.model_default_value', { value: Number(modelDefault.toPrecision(6)) })
            : t('pipelines.editor.model_default')

          return (
            <Field key={key} label={label}>
              <Input
                aria-label={label}
                aria-invalid={!isValidInferEngineThreshold(value)}
                className="text-xs"
                inputMode="decimal"
                placeholder={placeholder}
                value={value}
                onChange={(e) => onChange(key, e.target.value)}
              />
            </Field>
          )
        })}
      </div>
    </details>
  )
}

function NodeInspectorForm({
  node,
  onUpdateNode,
  onRenameNode,
  inspectorFlushRef,
}: {
  node: Node<PipelineNodeData>
  onUpdateNode: (
    nodeId: string,
    data: PipelineNodeData,
    options?: { commit?: 'immediate' | 'debounced' },
  ) => void
  onRenameNode: (oldId: string, newId: string) => void
  inspectorFlushRef?: MutableRefObject<(() => void) | null>
}) {
  const { t } = useT()
  const data = node.data
  const [pairs, setPairs] = useState<WithPair[]>(() => withToPairs(data.with))
  const [customType, setCustomType] = useState(() =>
    getNodeTypeDef(data.stageType) ? '' : data.stageType,
  )
  const [localId, setLocalId] = useState(data.stageId)
  const pairsRef = useRef(pairs)
  pairsRef.current = pairs

  const syncPairsToGraph = useCallback(
    (ps: WithPair[], options?: { commit?: 'immediate' | 'debounced' }) => {
      onUpdateNode(node.id, { ...data, with: pairsToWith(ps) }, options)
    },
    [node.id, data, onUpdateNode],
  )
  const syncPairsToGraphRef = useRef(syncPairsToGraph)
  syncPairsToGraphRef.current = syncPairsToGraph

  useEffect(() => {
    if (!inspectorFlushRef) return
    inspectorFlushRef.current = () => {
      syncPairsToGraphRef.current(pairsRef.current, { commit: 'immediate' })
    }
    return () => {
      inspectorFlushRef.current = null
    }
  }, [inspectorFlushRef])

  const selectValue = getNodeTypeDef(data.stageType) ? data.stageType : (data.stageType ? CUSTOM : '')
  const typeDef = getNodeTypeDef(data.stageType)

  const applyData = (
    patch: Partial<PipelineNodeData>,
    options?: { commit?: 'immediate' | 'debounced' },
  ) => {
    onUpdateNode(node.id, { ...data, ...patch }, options)
  }

  const handlePairsChange = (ps: WithPair[]) => {
    setPairs(ps)
    syncPairsToGraph(ps, { commit: 'debounced' })
  }

  const handleTypeSelect = (val: string) => {
    if (val === CUSTOM) {
      setCustomType('')
      applyData({ stageType: '' }, { commit: 'immediate' })
      return
    }
    const def = getNodeTypeDef(val)!
    const newPairs = def.withTemplate.map((p) => ({ ...p }))
    setPairs(newPairs)
    const autoId = data.stageId || val.replaceAll('.', '_')
    setLocalId(autoId)
    applyData(
      {
        stageType: val,
        stageId: autoId,
        with: pairsToWith(newPairs),
      },
      { commit: 'immediate' },
    )
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

  const setThreshold = (key: InferEngineThresholdKey, value: string) => {
    const next = pairs.filter((pair) => pair.k !== key)
    handlePairsChange(value === '' ? next : [...next, { k: key, v: value }])
  }

  const isInferEngine = data.stageType === 'infer.engine'

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
              applyData({ stageType: e.target.value }, { commit: 'debounced' })
            }}
            className="text-[11px]"
          />
        </Field>
      )}

      <div>
        <span className="label block mb-2">{t('pipelines.editor.params')}</span>
        <div className="space-y-2">
          {pairs.map((p, i) => {
            if (isInferEngine && INFER_ENGINE_THRESHOLD_KEYS.some((key) => key === p.k)) return null
            const paramDef = typeDef?.withTemplate.find((d) => d.k === p.k)
            return (
              <div key={i} className="flex gap-1">
                <ParamKeyCombobox
                  value={p.k}
                  options={typeDef?.withTemplate ?? []}
                  onChange={(k) => updatePair(i, { k })}
                  placeholder={t('pipelines.col.key')}
                  className="flex-1"
                />
                <ParamValueInput
                  paramDef={paramDef}
                  value={p.v}
                  onChange={(v) => updatePair(i, { v })}
                  placeholder={t('pipelines.col.value')}
                  className="flex-1"
                />
                <button
                  type="button"
                  onClick={() => handlePairsChange(pairs.filter((_, j) => j !== i))}
                  className="btn-icon text-danger/50 hover:text-danger"
                >
                  ×
                </button>
              </div>
            )
          })}
          <button
            type="button"
            onClick={() => handlePairsChange([...pairs, { k: '', v: '' }])}
            className="text-[10px] text-ink-muted hover:text-accent"
          >
            {t('pipelines.add_param')}
          </button>
        </div>
      </div>
      {isInferEngine && <InferEngineThresholdFields params={pairsToWith(pairs)} onChange={setThreshold} />}
    </div>
  )
}
