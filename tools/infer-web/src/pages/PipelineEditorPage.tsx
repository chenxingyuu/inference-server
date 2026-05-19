import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ReactFlowProvider, type Edge, type Node } from '@xyflow/react'
import toast from 'react-hot-toast'
import { Input } from '../components/ui/Field'
import { NodePalette } from '../components/pipeline/NodePalette'
import { PipelineCanvas } from '../components/pipeline/PipelineCanvas'
import { PipelineInspector } from '../components/pipeline/PipelineInspector'
import { usePipelineGraphHistory } from '../hooks/usePipelineGraphHistory'
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

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    target.isContentEditable
  )
}

function useDebouncedCommit(commitGraph: () => void, delayMs: number) {
  const commitRef = useRef(commitGraph)
  commitRef.current = commitGraph
  const timerRef = useRef<ReturnType<typeof setTimeout>>()

  const debouncedCommit = useCallback(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => commitRef.current(), delayMs)
  }, [delayMs])

  const flushCommit = useCallback(() => {
    clearTimeout(timerRef.current)
    commitRef.current()
  }, [])

  useEffect(() => () => clearTimeout(timerRef.current), [])

  return { debouncedCommit, flushCommit }
}

function PipelineEditorInner({ mode }: { mode: 'create' | 'edit' }) {
  const { t } = useT()
  const navigate = useNavigate()
  const { id: routeId } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const copyFrom = searchParams.get('copyFrom')
  const editorRef = useRef<HTMLDivElement>(null)

  const { data: pipelines = [], isLoading } = usePipelines()
  const add = useAddPipeline()
  const update = useUpdatePipeline()

  const isEdit = mode === 'edit'
  const [pipelineId, setPipelineId] = useState('')
  const [initialized, setInitialized] = useState(false)
  const [selectedNode, setSelectedNode] = useState<Node<PipelineNodeData> | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<Edge<PipelineEdgeData> | null>(null)
  const [fitViewRequest, setFitViewRequest] = useState(0)

  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    onNodesChange,
    onEdgesChange,
    setGraph,
    resetHistory,
    commitGraph,
    scheduleCommit,
    undo,
    redo,
    canUndo,
    canRedo,
    onNodeDragStart,
    onNodeDragStop,
    graphRevision,
  } = usePipelineGraphHistory()

  const { debouncedCommit, flushCommit } = useDebouncedCommit(commitGraph, 400)
  const inspectorFlushRef = useRef<(() => void) | null>(null)
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  nodesRef.current = nodes
  edgesRef.current = edges

  useEffect(() => {
    if (selectedNode) {
      const next = nodes.find((n) => n.id === selectedNode.id)
      if (!next) {
        setSelectedNode(null)
      } else if (
        next.position.x !== selectedNode.position.x ||
        next.position.y !== selectedNode.position.y ||
        JSON.stringify(next.data) !== JSON.stringify(selectedNode.data)
      ) {
        setSelectedNode(next)
      }
    }
    if (selectedEdge) {
      const next = edges.find((e) => e.id === selectedEdge.id)
      if (!next) {
        setSelectedEdge(null)
      } else if (JSON.stringify(next.data) !== JSON.stringify(selectedEdge.data)) {
        setSelectedEdge(next)
      }
    }
  }, [nodes, edges, selectedNode, selectedEdge])

  const applyInitialGraph = useCallback(
    (
      graph: { nodes: Node<PipelineNodeData>[]; edges: Edge<PipelineEdgeData>[] },
      layout: Record<string, { x: number; y: number }> | null,
    ) => {
      const nodeIds = graph.nodes.map((n) => n.id)
      const nextNodes = needsInitialAutoLayout(layout, nodeIds)
        ? applyDagreLayout(graph.nodes, graph.edges)
        : graph.nodes
      setGraph(nextNodes, graph.edges)
      resetHistory(nextNodes, graph.edges)
      if (nextNodes.length > 0) {
        setFitViewRequest((n) => n + 1)
      }
    },
    [setGraph, resetHistory],
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

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!editorRef.current?.contains(document.activeElement) && document.activeElement !== document.body) {
        return
      }
      if (isEditableTarget(e.target)) return

      const mod = e.metaKey || e.ctrlKey
      if (!mod) return

      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        undo()
        return
      }
      if (e.key === 'z' && e.shiftKey) {
        e.preventDefault()
        redo()
        return
      }
      if (e.key === 'y' && !e.shiftKey) {
        e.preventDefault()
        redo()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [undo, redo])

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
      setNodes((nds) => {
        const nextNodes = [...nds, newNode]
        commitGraph({ nodes: nextNodes, edges })
        return nextNodes
      })
    },
    [nodes, edges, setNodes, commitGraph],
  )

  const handleAutoLayout = useCallback(() => {
    setNodes((nds) => {
      const nextNodes = applyDagreLayout(nds, edges)
      commitGraph({ nodes: nextNodes, edges })
      return nextNodes
    })
    setFitViewRequest((n) => n + 1)
  }, [edges, setNodes, commitGraph])

  const handleUpdateNodeData = useCallback(
    (nodeId: string, data: PipelineNodeData, options?: { commit?: 'immediate' | 'debounced' }) => {
      if (options?.commit === 'immediate') {
        flushCommit()
      }
      setNodes((nds) => {
        const nextNodes = nds.map((n) => (n.id === nodeId ? { ...n, data } : n))
        if (options?.commit === 'immediate') {
          commitGraph({ nodes: nextNodes, edges })
        }
        return nextNodes
      })
      if (options?.commit !== 'immediate') {
        debouncedCommit()
      }
    },
    [setNodes, edges, debouncedCommit, flushCommit, commitGraph],
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
      commitGraph({ nodes: nextNodes, edges: nextEdges })
    },
    [nodes, edges, setNodes, setEdges, selectedNode?.id, t, commitGraph],
  )

  const handleUpdateEdgeData = useCallback(
    (edgeId: string, data: PipelineEdgeData) => {
      setEdges((eds) =>
        eds.map((e) => (e.id === edgeId ? { ...e, data } : e)),
      )
      if (selectedEdge?.id === edgeId) {
        setSelectedEdge((prev) => (prev ? { ...prev, data } : null))
      }
      debouncedCommit()
    },
    [setEdges, selectedEdge?.id, debouncedCommit],
  )

  const handleSave = () => {
    flushSync(() => {
      inspectorFlushRef.current?.()
    })
    flushCommit()
    const liveNodes = nodesRef.current
    const liveEdges = edgesRef.current
    const result = validatePipelineGraph(pipelineId, liveNodes, liveEdges)
    if (!result.ok) {
      toast.error(formatValidationErrors(t, result))
      return
    }
    const body = toPipelineCreate(pipelineId, liveNodes, liveEdges)
    if (isEdit && routeId) {
      update.mutate(
        { id: routeId, body },
        {
          onSuccess: () => {
            saveLayout(pipelineId, liveNodes)
            navigate('/pipelines')
          },
        },
      )
    } else {
      add.mutate(body, {
        onSuccess: () => {
          saveLayout(pipelineId, liveNodes)
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
    <div ref={editorRef} className="flex flex-col h-[calc(100vh-0px)] -m-0" tabIndex={-1}>
      <header className="flex-shrink-0 flex items-center gap-4 px-4 py-3 border-b border-border bg-bg-surface">
        <Link to="/pipelines" className="text-[12px] text-ink-muted hover:text-accent">
          ← {t('pipelines.editor.back')}
        </Link>
        <div className="flex items-center gap-2 min-w-0 max-w-md flex-1">
          <label
            htmlFor="pipeline-id"
            className="shrink-0 text-[11px] font-medium uppercase tracking-wide text-ink-muted whitespace-nowrap"
          >
            {t('pipelines.field.id')}
          </label>
          <Input
            id="pipeline-id"
            placeholder="detection-pipeline"
            value={pipelineId}
            onChange={(e) => setPipelineId(e.target.value)}
            disabled={isEdit}
            className={`flex-1 min-w-0 py-1.5 ${isEdit ? 'opacity-60 cursor-not-allowed' : ''}`}
          />
        </div>
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
          onAutoLayout={handleAutoLayout}
          onGraphCommit={(snap) => (snap ? commitGraph(snap) : scheduleCommit())}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={onNodeDragStop}
          onUndo={undo}
          onRedo={redo}
          canUndo={canUndo}
          canRedo={canRedo}
          onSelectionChange={(node, edge) => {
            setSelectedNode(node)
            setSelectedEdge(edge)
          }}
        />
        <PipelineInspector
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          graphRevision={graphRevision}
          inspectorFlushRef={inspectorFlushRef}
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
