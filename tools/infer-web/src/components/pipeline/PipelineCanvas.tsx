import { useCallback, useEffect, useRef } from 'react'
import {
  ReactFlow,
  Background,
  MiniMap,
  addEdge,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
} from '@xyflow/react'
import toast from 'react-hot-toast'
import type { GraphSnapshot } from '../../lib/pipelineGraphHistory'
import type { PipelineEdgeData, PipelineNodeData } from '../../lib/pipelineGraph'
import { wouldCreateCycle, createStageNode, PIPELINE_EDGE_TYPE } from '../../lib/pipelineGraph'
import { getHandleConfig } from '../../lib/pipelineHandles'
import { getDragNodeType } from './NodePalette'
import { pipelineEdgeTypes, pipelineNodeTypes } from './pipelineFlowTypes'
import { PipelineFlowControls } from './PipelineFlowControls'
import { useT } from '../../lib/i18n'

interface PipelineCanvasProps {
  nodes: Node<PipelineNodeData>[]
  edges: Edge<PipelineEdgeData>[]
  onNodesChange: OnNodesChange<Node<PipelineNodeData>>
  onEdgesChange: OnEdgesChange<Edge<PipelineEdgeData>>
  setNodes: React.Dispatch<React.SetStateAction<Node<PipelineNodeData>[]>>
  setEdges: React.Dispatch<React.SetStateAction<Edge<PipelineEdgeData>[]>>
  onSelectionChange: (node: Node<PipelineNodeData> | null, edge: Edge<PipelineEdgeData> | null) => void
  fitViewRequest?: number
  onAutoLayout: () => void
  onGraphCommit: (snapshot?: GraphSnapshot) => void
  onNodeDragStart: () => void
  onNodeDragStop: () => void
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
}

export function PipelineCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  setNodes,
  setEdges,
  onSelectionChange,
  fitViewRequest = 0,
  onAutoLayout,
  onGraphCommit,
  onNodeDragStart,
  onNodeDragStop,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
}: PipelineCanvasProps) {
  const { t } = useT()
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const { screenToFlowPosition, fitView } = useReactFlow()

  useEffect(() => {
    if (!fitViewRequest || nodes.length === 0) return
    const timer = window.setTimeout(() => {
      void fitView({ padding: 0.2, maxZoom: 1, duration: 200 })
    }, 50)
    return () => window.clearTimeout(timer)
  }, [fitViewRequest, nodes.length, fitView])

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return
      if (connection.source === connection.target) {
        toast.error(t('pipelines.editor.error_self_loop'))
        return
      }
      if (wouldCreateCycle(edges, connection.source, connection.target)) {
        toast.error(t('pipelines.editor.error_cycle'))
        return
      }
      const newEdge: Edge<PipelineEdgeData> = {
        id: `e:${connection.source}->${connection.target}:${edges.length}`,
        source: connection.source,
        target: connection.target,
        type: PIPELINE_EDGE_TYPE,
        data: {
          from: connection.source,
          to: connection.target,
          capacity: 32,
          drop_policy: 'drop_oldest',
        },
      }
      setEdges((eds) => {
        const nextEdges = addEdge(newEdge, eds)
        onGraphCommit({ nodes, edges: nextEdges })
        return nextEdges
      })
    },
    [edges, nodes, setEdges, onGraphCommit, t],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const nodeType = getDragNodeType(e.dataTransfer)
      if (!nodeType) return
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const existingIds = new Set(nodes.map((n) => n.id))
      const newNode = createStageNode(nodeType, position, existingIds)
      setNodes((nds) => {
        const nextNodes = [...nds, newNode]
        onGraphCommit({ nodes: nextNodes, edges })
        return nextNodes
      })
    },
    [nodes, edges, screenToFlowPosition, setNodes, onGraphCommit],
  )

  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      const source = 'source' in connection ? connection.source : null
      const target = 'target' in connection ? connection.target : null
      if (!source || !target || source === target) return false
      const sourceNode = nodes.find((n) => n.id === source)
      const targetNode = nodes.find((n) => n.id === target)
      if (!sourceNode || !targetNode) return false
      const srcHandles = getHandleConfig(sourceNode.data.stageType)
      const tgtHandles = getHandleConfig(targetNode.data.stageType)
      if (srcHandles.outputs === 'none' || tgtHandles.inputs === 'none') return false
      return !wouldCreateCycle(edges, source, target)
    },
    [nodes, edges],
  )

  return (
    <div ref={reactFlowWrapper} className="flex-1 min-h-0 h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        isValidConnection={isValidConnection}
        nodeTypes={pipelineNodeTypes}
        edgeTypes={pipelineEdgeTypes}
        deleteKeyCode={['Backspace', 'Delete']}
        onNodesDelete={(deleted) => {
          const ids = new Set(deleted.map((n) => n.id))
          setEdges((eds) => eds.filter((e) => !ids.has(e.source) && !ids.has(e.target)))
          onGraphCommit()
        }}
        onSelectionChange={({ nodes: selNodes, edges: selEdges }) => {
          onSelectionChange(
            selNodes.length === 1 ? selNodes[0] : null,
            selEdges.length === 1 ? selEdges[0] : null,
          )
        }}
        className="pipeline-editor-flow bg-bg-base"
      >
        <Background gap={16} size={1} color="#1e2d3d" />
        <PipelineFlowControls
          onAutoLayout={onAutoLayout}
          layoutDisabled={nodes.length === 0}
          onUndo={onUndo}
          onRedo={onRedo}
          canUndo={canUndo}
          canRedo={canRedo}
        />
        <MiniMap
          className="pipeline-flow-minimap"
          maskColor="rgba(11, 15, 20, 0.72)"
          nodeColor={(n) => {
            const d = n.data as PipelineNodeData
            return d?.stageType?.startsWith('source') ? '#34d399' : '#22d3ee'
          }}
        />
      </ReactFlow>
    </div>
  )
}
