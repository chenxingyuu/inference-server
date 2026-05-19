import type { NodeTypes, EdgeTypes } from '@xyflow/react'
import { PipelineNode } from './PipelineNode'
import { PipelineEdge } from './PipelineEdge'
import { PIPELINE_NODE_TYPE, PIPELINE_EDGE_TYPE } from '../../lib/pipelineGraph'

export const pipelineNodeTypes: NodeTypes = {
  [PIPELINE_NODE_TYPE]: PipelineNode,
}

export const pipelineEdgeTypes: EdgeTypes = {
  [PIPELINE_EDGE_TYPE]: PipelineEdge,
}
