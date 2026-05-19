import dagre from '@dagrejs/dagre'
import type { Edge, Node } from '@xyflow/react'
import type { PipelineCreate, PipelineInfo, StageConfig, EdgeConfig, DropPolicy } from '../types'
import { getNodeTypeDef } from './nodeTypes'

export const PIPELINE_NODE_TYPE = 'pipelineNode'
export const PIPELINE_EDGE_TYPE = 'pipelineEdge'

export const NODE_WIDTH = 200
export const NODE_HEIGHT = 72

export type PipelineNodeData = {
  stageId: string
  stageType: string
  with?: Record<string, string>
}

export type PipelineEdgeData = {
  from: string
  to: string
  capacity: number
  drop_policy: DropPolicy
}

const LAYOUT_PREFIX = 'pipeline-layout:v1:'

export function layoutStorageKey(pipelineId: string): string {
  return `${LAYOUT_PREFIX}${pipelineId}`
}

export function loadLayout(pipelineId: string): Record<string, { x: number; y: number }> | null {
  try {
    const raw = localStorage.getItem(layoutStorageKey(pipelineId))
    if (!raw) return null
    return JSON.parse(raw) as Record<string, { x: number; y: number }>
  } catch {
    return null
  }
}

export function saveLayout(pipelineId: string, nodes: Node<PipelineNodeData>[]): void {
  const positions: Record<string, { x: number; y: number }> = {}
  for (const n of nodes) {
    positions[n.id] = { x: n.position.x, y: n.position.y }
  }
  localStorage.setItem(layoutStorageKey(pipelineId), JSON.stringify(positions))
}

export function defaultStageId(stageType: string, existingIds: Set<string>): string {
  const base = stageType.replaceAll('.', '_')
  if (!existingIds.has(base)) return base
  let i = 2
  while (existingIds.has(`${base}_${i}`)) i++
  return `${base}_${i}`
}

export function createStageNode(
  stageType: string,
  position: { x: number; y: number },
  existingIds: Set<string>,
): Node<PipelineNodeData> {
  const def = getNodeTypeDef(stageType)
  const stageId = defaultStageId(stageType, existingIds)
  const withRecord: Record<string, string> = {}
  def?.withTemplate.forEach(({ k, v }) => { if (k) withRecord[k] = v })
  return {
    id: stageId,
    type: PIPELINE_NODE_TYPE,
    position,
    data: {
      stageId,
      stageType,
      with: Object.keys(withRecord).length ? withRecord : undefined,
    },
  }
}

export function fromPipelineInfo(
  pipeline: PipelineInfo,
  layout?: Record<string, { x: number; y: number }> | null,
): { nodes: Node<PipelineNodeData>[]; edges: Edge<PipelineEdgeData>[] } {
  const nodes: Node<PipelineNodeData>[] = pipeline.nodes.map((n, i) => ({
    id: n.id,
    type: PIPELINE_NODE_TYPE,
    position: layout?.[n.id] ?? { x: 80 + (i % 4) * 220, y: 80 + Math.floor(i / 4) * 100 },
    data: {
      stageId: n.id,
      stageType: n.type,
      with: n.with,
    },
  }))

  const edges: Edge<PipelineEdgeData>[] = (pipeline.edges ?? []).map((e, i) => ({
    id: edgeId(e.from, e.to, i),
    source: e.from,
    target: e.to,
    type: PIPELINE_EDGE_TYPE as string,
    data: {
      from: e.from,
      to: e.to,
      capacity: e.capacity ?? 32,
      drop_policy: (e.drop_policy ?? 'drop_oldest') as DropPolicy,
    },
  }))

  return { nodes, edges }
}

function edgeId(from: string, to: string, index: number): string {
  return `e:${from}->${to}:${index}`
}

export function toPipelineCreate(
  pipelineId: string,
  rfNodes: Node<PipelineNodeData>[],
  rfEdges: Edge<PipelineEdgeData>[],
): PipelineCreate {
  const nodes: StageConfig[] = rfNodes.map((n) => ({
    id: n.data.stageId,
    type: n.data.stageType,
    with: n.data.with,
  }))
  const edges: EdgeConfig[] = rfEdges.map((e) => ({
    from: e.data?.from ?? e.source,
    to: e.data?.to ?? e.target,
    capacity: e.data?.capacity ?? 32,
    drop_policy: e.data?.drop_policy ?? 'drop_oldest',
  }))
  return { id: pipelineId, nodes, edges }
}

export type ValidationResult =
  | { ok: true }
  | { ok: false; errors: string[] }

export function validatePipelineGraph(
  pipelineId: string,
  rfNodes: Node<PipelineNodeData>[],
  rfEdges: Edge<PipelineEdgeData>[],
): ValidationResult {
  const errors: string[] = []

  if (!pipelineId.trim()) {
    errors.push('pipeline_id_empty')
  }
  if (rfNodes.length === 0) {
    errors.push('pipeline_nodes_empty')
  }

  const nodeIds = new Set<string>()
  for (const n of rfNodes) {
    const id = n.data.stageId?.trim() ?? ''
    const type = n.data.stageType?.trim() ?? ''
    if (!id) errors.push('node_id_empty')
    else if (nodeIds.has(id)) errors.push(`duplicate_node_id:${id}`)
    else nodeIds.add(id)
    if (!type) errors.push(`node_type_empty:${id || n.id}`)

    if (type === 'sink.stream') {
      const url = n.data.with?.output_url?.trim()
      if (!url) errors.push(`sink_stream_output_url:${id}`)
    }
  }

  const edgePairs: { from: string; to: string }[] = []
  for (const e of rfEdges) {
    const from = e.data?.from ?? e.source
    const to = e.data?.to ?? e.target
    const cap = e.data?.capacity ?? 32
    if (!nodeIds.has(from)) errors.push(`edge_from_missing:${from}`)
    if (!nodeIds.has(to)) errors.push(`edge_to_missing:${to}`)
    if (cap < 1) errors.push(`edge_capacity_invalid:${from}->${to}`)
    if (from && to) edgePairs.push({ from, to })
  }

  if (errors.length > 0) {
    return { ok: false, errors: [...new Set(errors)] }
  }

  if (hasCycle(nodeIds, edgePairs)) {
    errors.push('pipeline_cycle')
    return { ok: false, errors }
  }

  return { ok: true }
}

export function hasCycle(
  nodeIds: Set<string>,
  edges: { from: string; to: string }[],
): boolean {
  const indegree = new Map<string, number>()
  const graph = new Map<string, string[]>()
  for (const id of nodeIds) {
    indegree.set(id, 0)
    graph.set(id, [])
  }
  for (const { from, to } of edges) {
    graph.get(from)!.push(to)
    indegree.set(to, (indegree.get(to) ?? 0) + 1)
  }
  const queue: string[] = []
  for (const [id, deg] of indegree) {
    if (deg === 0) queue.push(id)
  }
  let visited = 0
  while (queue.length > 0) {
    const cur = queue.shift()!
    visited++
    for (const nxt of graph.get(cur) ?? []) {
      const nextDeg = (indegree.get(nxt) ?? 0) - 1
      indegree.set(nxt, nextDeg)
      if (nextDeg === 0) queue.push(nxt)
    }
  }
  return visited !== nodeIds.size
}

export function wouldCreateCycle(
  rfEdges: Edge<PipelineEdgeData>[],
  from: string,
  to: string,
): boolean {
  if (from === to) return true
  const nodeIds = new Set<string>()
  const pairs: { from: string; to: string }[] = []
  for (const e of rfEdges) {
    const f = e.data?.from ?? e.source
    const t = e.data?.to ?? e.target
    nodeIds.add(f)
    nodeIds.add(t)
    pairs.push({ from: f, to: t })
  }
  nodeIds.add(from)
  nodeIds.add(to)
  pairs.push({ from, to })
  return hasCycle(nodeIds, pairs)
}

export function applyDagreLayout(
  nodes: Node<PipelineNodeData>[],
  edges: Edge<PipelineEdgeData>[],
): Node<PipelineNodeData>[] {
  if (nodes.length === 0) return nodes

  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 100, marginx: 40, marginy: 40 })

  for (const n of nodes) {
    g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const e of edges) {
    g.setEdge(e.source, e.target)
  }
  dagre.layout(g)

  return nodes.map((n) => {
    const pos = g.node(n.id)
    return {
      ...n,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    }
  })
}

export function renameNodeId(
  nodes: Node<PipelineNodeData>[],
  edges: Edge<PipelineEdgeData>[],
  oldId: string,
  newId: string,
): { nodes: Node<PipelineNodeData>[]; edges: Edge<PipelineEdgeData>[] } {
  const trimmed = newId.trim()
  if (!trimmed || trimmed === oldId) {
    return { nodes, edges }
  }

  const nextNodes = nodes.map((n) => {
    if (n.id !== oldId) return n
    return {
      ...n,
      id: trimmed,
      data: { ...n.data, stageId: trimmed },
    }
  })

  const nextEdges = edges.map((e) => {
    const from = (e.data?.from ?? e.source) === oldId ? trimmed : (e.data?.from ?? e.source)
    const to = (e.data?.to ?? e.target) === oldId ? trimmed : (e.data?.to ?? e.target)
    const idx = edges.indexOf(e)
    return {
      ...e,
      id: edgeId(from, to, idx),
      source: e.source === oldId ? trimmed : e.source,
      target: e.target === oldId ? trimmed : e.target,
      data: {
        from,
        to,
        capacity: e.data?.capacity ?? 32,
        drop_policy: e.data?.drop_policy ?? 'drop_oldest',
      },
    }
  })

  return { nodes: nextNodes, edges: nextEdges }
}

export function removeNodeAndEdges(
  nodes: Node<PipelineNodeData>[],
  edges: Edge<PipelineEdgeData>[],
  nodeId: string,
): { nodes: Node<PipelineNodeData>[]; edges: Edge<PipelineEdgeData>[] } {
  return {
    nodes: nodes.filter((n) => n.id !== nodeId),
    edges: edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
  }
}

export function pairsToWith(ps: { k: string; v: string }[]): Record<string, string> | undefined {
  const record: Record<string, string> = {}
  ps.forEach(({ k, v }) => { if (k) record[k] = v })
  return Object.keys(record).length ? record : undefined
}
