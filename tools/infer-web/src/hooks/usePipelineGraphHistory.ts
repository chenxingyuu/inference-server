import { useCallback, useRef, useState } from 'react'
import {
  useEdgesState,
  useNodesState,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import {
  cloneGraphSnapshot,
  createHistoryState,
  graphsEqual,
  pushSnapshot,
  redoHistory,
  undoHistory,
  type GraphSnapshot,
} from '../lib/pipelineGraphHistory'
import type { PipelineEdgeData, PipelineNodeData } from '../lib/pipelineGraph'

export function usePipelineGraphHistory(maxSize = 50) {
  const [nodes, setNodes, onNodesChangeBase] = useNodesState<Node<PipelineNodeData>>([])
  const [edges, setEdges, onEdgesChangeBase] = useEdgesState<Edge<PipelineEdgeData>>([])

  const [history, setHistory] = useState(() =>
    createHistoryState(cloneGraphSnapshot([], []), maxSize),
  )
  const [graphRevision, setGraphRevision] = useState(0)

  const isApplyingHistoryRef = useRef(false)
  const dragBaselineRef = useRef<GraphSnapshot | null>(null)
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  nodesRef.current = nodes
  edgesRef.current = edges

  const canUndo = history.past.length > 0
  const canRedo = history.future.length > 0

  const getLiveSnapshot = useCallback(
    (): GraphSnapshot => cloneGraphSnapshot(nodesRef.current, edgesRef.current),
    [],
  )

  const commitGraph = useCallback((snapshot?: GraphSnapshot) => {
    if (isApplyingHistoryRef.current) return
    const snap = snapshot ?? getLiveSnapshot()
    setHistory((h) => pushSnapshot(h, snap))
  }, [getLiveSnapshot])

  const scheduleCommit = useCallback(() => {
    requestAnimationFrame(() => {
      commitGraph()
    })
  }, [commitGraph])

  const resetHistory = useCallback(
    (nextNodes: Node<PipelineNodeData>[], nextEdges: Edge<PipelineEdgeData>[]) => {
      const snap = cloneGraphSnapshot(nextNodes, nextEdges)
      setHistory(createHistoryState(snap, maxSize))
      setGraphRevision((r) => r + 1)
    },
    [maxSize],
  )

  const setGraph = useCallback(
    (
      nextNodes: Node<PipelineNodeData>[],
      nextEdges: Edge<PipelineEdgeData>[],
      options?: { syncHistoryPresent?: boolean },
    ) => {
      setNodes(nextNodes)
      setEdges(nextEdges)
      if (options?.syncHistoryPresent) {
        const snap = cloneGraphSnapshot(nextNodes, nextEdges)
        setHistory(createHistoryState(snap, maxSize))
      }
    },
    [setNodes, setEdges, maxSize],
  )

  const applyHistorySnapshot = useCallback(
    (snapshot: GraphSnapshot) => {
      isApplyingHistoryRef.current = true
      setNodes(snapshot.nodes)
      setEdges(snapshot.edges)
      requestAnimationFrame(() => {
        isApplyingHistoryRef.current = false
      })
    },
    [setNodes, setEdges],
  )

  const undo = useCallback(() => {
    setHistory((h) => {
      const result = undoHistory(h)
      if (!result) return h
      applyHistorySnapshot(result.snapshot)
      setGraphRevision((r) => r + 1)
      return result.state
    })
  }, [applyHistorySnapshot])

  const redo = useCallback(() => {
    setHistory((h) => {
      const result = redoHistory(h)
      if (!result) return h
      applyHistorySnapshot(result.snapshot)
      setGraphRevision((r) => r + 1)
      return result.state
    })
  }, [applyHistorySnapshot])

  const onNodeDragStart = useCallback(() => {
    dragBaselineRef.current = getLiveSnapshot()
  }, [getLiveSnapshot])

  const onNodeDragStop = useCallback(() => {
    const baseline = dragBaselineRef.current
    dragBaselineRef.current = null
    if (!baseline || isApplyingHistoryRef.current) return
    const current = getLiveSnapshot()
    if (!graphsEqual(baseline, current)) {
      commitGraph()
    }
  }, [getLiveSnapshot, commitGraph])

  const onNodesChange = useCallback(
    (changes: NodeChange<Node<PipelineNodeData>>[]) => {
      onNodesChangeBase(changes)
    },
    [onNodesChangeBase],
  )

  const onEdgesChange = useCallback(
    (changes: EdgeChange<Edge<PipelineEdgeData>>[]) => {
      onEdgesChangeBase(changes)
      if (
        !isApplyingHistoryRef.current &&
        changes.some((c) => c.type === 'remove')
      ) {
        scheduleCommit()
      }
    },
    [onEdgesChangeBase, scheduleCommit],
  )

  return {
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
    isApplyingHistoryRef,
    graphRevision,
  }
}
