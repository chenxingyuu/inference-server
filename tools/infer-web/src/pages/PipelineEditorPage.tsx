import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
} from '@xyflow/react'
import toast from 'react-hot-toast'
import { Field, Input } from '../components/ui/Field'
import { NodePalette } from '../components/pipeline/NodePalette'
import { PipelineCanvas } from '../components/pipeline/PipelineCanvas'
import { PipelineInspector } from '../components/pipeline/PipelineInspector'
import { useAddPipeline, usePipelines, useUpdatePipeline } from '../hooks/queries'
import { useT } from '../lib/i18n'
import {
  applyDagreLayout,
  createStageNode,
  fromPipelineInfo,
  loadLayout,
  needsInitialAutoLayout,
  renameNodeId,
  saveLayout,
  toPipelineCreate,
  validatePipelineGraph,
  type PipelineEdgeData,
  type PipelineNodeData,
  type ValidationResult,
} from '../lib/pipelineGraph'

function formatValidationErrors(
  t: ReturnType<typeof useT>['t'],
  result: ValidationResult,
): string {
  if (result.ok) return ''
  return result.errors
    .map((code) => {
      const suffix = code.split(':')[0]
      const key = `pipelines.editor.error.${suffix}` as Parameters<ReturnType<typeof useT>['t']>[0]
      const msg = t(key)
      return msg !== key ? msg : code
    })
    .join(' · ')
}

function PipelineEditorInner({ mode }: { mode: 'create' | 'edit' }) {
  const { t } = useT()
  const navigate = useNavigate()
  const { id: routeId } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const copyFrom = searchParams.get('copyFrom')

  const { data: pipelines = [], isLoading } = usePipelines()
  const add = useAddPipeline()
  const update = useUpdatePipeline()

  const isEdit = mode === 'edit'
  const [pipelineId, setPipelineId] = useState('')
  const [initialized, setInitialized] = useState(false)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<PipelineNodeData>>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge<PipelineEdgeData>>([])
  const [selectedNode, setSelectedNode] = useState<Node<PipelineNodeData> | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<Edge<PipelineEdgeData> | null>(null)
  const [fitViewRequest, setFitViewRequest] = useState(0)

  const applyInitialGraph = useCallback(
    (
      graph: { nodes: Node<PipelineNodeData>[]; edges: Edge<PipelineEdgeData>[] },
      layout: Record<string, { x: number; y: number }> | null,
    ) => {
      const nodeIds = graph.nodes.map((n) => n.id)
      const nodes = needsInitialAutoLayout(layout, nodeIds)
        ? applyDagreLayout(graph.nodes, graph.edges)
        : graph.nodes
      setNodes(nodes)
      setEdges(graph.edges)
      if (nodes.length > 0) {
        setFitViewRequest((n) => n + 1)
      }
    },
    [setNodes, setEdges],
  )

  useEffect(() => {
    if (initialized || isLoading) return
    if (isEdit && routeId) {
      const p = pipelines.find((x) => x.id === routeId)
      if (!p) return
      const layout = loadLayout(p.id)
      applyInitialGraph(fromPipelineInfo(p, layout), layout)
      setPipelineId(p.id)
      setInitialized(true)
      return
    }
    if (!isEdit && copyFrom) {
      const p = pipelines.find((x) => x.id === copyFrom)
      if (p) {
        applyInitialGraph(fromPipelineInfo(p, null), null)
        setPipelineId(`${p.id}-copy`)
      }
      setInitialized(true)
      return
    }
    if (!isEdit) {
      setInitialized(true)
    }
  }, [
    initialized,
    isLoading,
    isEdit,
    routeId,
    copyFrom,
    pipelines,
    applyInitialGraph,
  ])

  const validation = useMemo(
    () => validatePipelineGraph(pipelineId, nodes, edges),
    [pipelineId, nodes, edges],
  )

  const validationMessage = useMemo(
    () => formatValidationErrors(t, validation),
    [t, validation],
  )

  const handleAddType = useCallback(
    (nodeType: string) => {
      const existingIds = new Set(nodes.map((n) => n.id))
      const offset = nodes.length
      const newNode = createStageNode(
        nodeType,
        { x: 120 + (offset % 5) * 40, y: 120 + Math.floor(offset / 5) * 90 },
        existingIds,
      )
      setNodes((nds) => [...nds, newNode])
    },
    [nodes, setNodes],
  )

  const handleAutoLayout = useCallback(() => {
    setNodes((nds) => applyDagreLayout(nds, edges))
    setFitViewRequest((n) => n + 1)
  }, [edges, setNodes])

  const handleUpdateNodeData = useCallback(
    (nodeId: string, data: PipelineNodeData) => {
      setNodes((nds) =>
        nds.map((n) => (n.id === nodeId ? { ...n, data } : n)),
      )
      if (selectedNode?.id === nodeId) {
        setSelectedNode((prev) => (prev ? { ...prev, data } : null))
      }
    },
    [setNodes, selectedNode?.id],
  )

  const handleRenameNode = useCallback(
    (oldId: string, newId: string) => {
      const trimmed = newId.trim()
      if (!trimmed || trimmed === oldId) return
      if (nodes.some((n) => n.id === trimmed)) {
        toast.error(t('pipelines.editor.error.duplicate_node_id'))
        return
      }
      const { nodes: nextNodes, edges: nextEdges } = renameNodeId(nodes, edges, oldId, trimmed)
      setNodes(nextNodes)
      setEdges(nextEdges)
      if (selectedNode?.id === oldId) {
        setSelectedNode(nextNodes.find((n) => n.id === trimmed) ?? null)
      }
    },
    [nodes, edges, setNodes, setEdges, selectedNode?.id, t],
  )

  const handleUpdateEdgeData = useCallback(
    (edgeId: string, data: PipelineEdgeData) => {
      setEdges((eds) =>
        eds.map((e) => (e.id === edgeId ? { ...e, data } : e)),
      )
      if (selectedEdge?.id === edgeId) {
        setSelectedEdge((prev) => (prev ? { ...prev, data } : null))
      }
    },
    [setEdges, selectedEdge?.id],
  )

  const handleSave = () => {
    const result = validatePipelineGraph(pipelineId, nodes, edges)
    if (!result.ok) {
      toast.error(formatValidationErrors(t, result))
      return
    }
    const body = toPipelineCreate(pipelineId, nodes, edges)
    if (isEdit && routeId) {
      update.mutate(
        { id: routeId, body },
        {
          onSuccess: () => {
            saveLayout(pipelineId, nodes)
            navigate('/pipelines')
          },
        },
      )
    } else {
      add.mutate(body, {
        onSuccess: () => {
          saveLayout(pipelineId, nodes)
          navigate('/pipelines')
        },
      })
    }
  }

  const pending = add.isPending || update.isPending

  if (isEdit && !isLoading && initialized && !pipelines.find((p) => p.id === routeId)) {
    return (
      <div className="p-8">
        <p className="text-danger">{t('pipelines.editor.not_found')}</p>
        <Link to="/pipelines" className="text-accent text-sm mt-2 inline-block">
          {t('pipelines.editor.back')}
        </Link>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-[calc(100vh-0px)] -m-0">
      <header className="flex-shrink-0 flex items-center gap-4 px-4 py-3 border-b border-border bg-bg-surface">
        <Link to="/pipelines" className="text-[12px] text-ink-muted hover:text-accent">
          ← {t('pipelines.editor.back')}
        </Link>
        <div className="flex-1 max-w-xs">
        <Field label={t('pipelines.field.id')}>
          <Input
            placeholder="detection-pipeline"
            value={pipelineId}
            onChange={(e) => setPipelineId(e.target.value)}
            disabled={isEdit}
            className={isEdit ? 'opacity-60 cursor-not-allowed' : ''}
          />
        </Field>
        </div>
        <button type="button" onClick={handleAutoLayout} className="btn-ghost text-[12px]">
          {t('pipelines.editor.auto_layout')}
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={!validation.ok || pending}
          className="btn-primary"
        >
          {pending ? t('pipelines.saving') : t('pipelines.editor.save')}
        </button>
      </header>

      {!validation.ok && validationMessage && (
        <div className="px-4 py-2 text-[11px] text-danger bg-danger/10 border-b border-danger/30">
          {validationMessage}
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <NodePalette onAddType={handleAddType} />
        <PipelineCanvas
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          setNodes={setNodes}
          setEdges={setEdges}
          fitViewRequest={fitViewRequest}
          onSelectionChange={(node, edge) => {
            setSelectedNode(node)
            setSelectedEdge(edge)
          }}
        />
        <PipelineInspector
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          onUpdateNode={handleUpdateNodeData}
          onRenameNode={handleRenameNode}
          onUpdateEdge={handleUpdateEdgeData}
        />
      </div>
    </div>
  )
}

export function PipelineEditorPage({ mode }: { mode: 'create' | 'edit' }) {
  return (
    <ReactFlowProvider>
      <PipelineEditorInner mode={mode} />
    </ReactFlowProvider>
  )
}
