import type { Edge, Node } from '@xyflow/react'
import type { PipelineEdgeData, PipelineNodeData } from './pipelineGraph'

export interface GraphSnapshot {
  nodes: Node<PipelineNodeData>[]
  edges: Edge<PipelineEdgeData>[]
}

export interface GraphHistoryState {
  past: GraphSnapshot[]
  present: GraphSnapshot
  future: GraphSnapshot[]
  maxSize: number
}

const DEFAULT_MAX_SIZE = 50

function stripNode(n: Node<PipelineNodeData>) {
  return {
    id: n.id,
    type: n.type,
    position: n.position,
    data: n.data,
  }
}

function stripEdge(e: Edge<PipelineEdgeData>) {
  return {
    id: e.id,
    source: e.source,
    target: e.target,
    type: e.type,
    data: e.data,
  }
}

export function cloneGraphSnapshot(
  nodes: Node<PipelineNodeData>[],
  edges: Edge<PipelineEdgeData>[],
): GraphSnapshot {
  return structuredClone({ nodes, edges })
}

export function graphsEqual(a: GraphSnapshot, b: GraphSnapshot): boolean {
  const sa = JSON.stringify({
    nodes: a.nodes.map(stripNode),
    edges: a.edges.map(stripEdge),
  })
  const sb = JSON.stringify({
    nodes: b.nodes.map(stripNode),
    edges: b.edges.map(stripEdge),
  })
  return sa === sb
}

export function createHistoryState(
  present: GraphSnapshot,
  maxSize = DEFAULT_MAX_SIZE,
): GraphHistoryState {
  return {
    past: [],
    present: cloneGraphSnapshot(present.nodes, present.edges),
    future: [],
    maxSize,
  }
}

export function pushSnapshot(
  state: GraphHistoryState,
  snapshot: GraphSnapshot,
): GraphHistoryState {
  const next = cloneGraphSnapshot(snapshot.nodes, snapshot.edges)
  if (graphsEqual(state.present, next)) return state

  const past = [...state.past, cloneGraphSnapshot(state.present.nodes, state.present.edges)]
  const trimmed =
    past.length > state.maxSize ? past.slice(past.length - state.maxSize) : past

  return {
    ...state,
    past: trimmed,
    present: next,
    future: [],
  }
}

export function undoHistory(
  state: GraphHistoryState,
): { state: GraphHistoryState; snapshot: GraphSnapshot } | null {
  if (state.past.length === 0) return null

  const previous = state.past[state.past.length - 1]
  const past = state.past.slice(0, -1)
  const future = [
    cloneGraphSnapshot(state.present.nodes, state.present.edges),
    ...state.future,
  ]

  return {
    state: {
      ...state,
      past,
      present: cloneGraphSnapshot(previous.nodes, previous.edges),
      future,
    },
    snapshot: cloneGraphSnapshot(previous.nodes, previous.edges),
  }
}

export function redoHistory(
  state: GraphHistoryState,
): { state: GraphHistoryState; snapshot: GraphSnapshot } | null {
  if (state.future.length === 0) return null

  const [next, ...future] = state.future
  const past = [
    ...state.past,
    cloneGraphSnapshot(state.present.nodes, state.present.edges),
  ]
  const trimmed =
    past.length > state.maxSize ? past.slice(past.length - state.maxSize) : past

  return {
    state: {
      ...state,
      past: trimmed,
      present: cloneGraphSnapshot(next.nodes, next.edges),
      future,
    },
    snapshot: cloneGraphSnapshot(next.nodes, next.edges),
  }
}
