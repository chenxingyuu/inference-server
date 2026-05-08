export type SourceState = 'CONNECTING' | 'STREAMING' | 'RECONNECTING' | 'DEGRADED' | 'STOPPED'
export type TaskState = 'running' | 'stopped'
export type BackendType = 'cpu' | 'trt' | 'ascend'
export type ModelVersion = 'v5' | 'v8' | 'v11' | 'v26'
export type ModelType = 'detector' | 'classifier'
export type DropPolicy = 'block' | 'drop_oldest' | 'drop_newest'
export type SamplingMode = 'frame_count' | 'time_based'

export interface SourceInfo {
  id: string
  url: string
  state: SourceState
  reconnect_count: number
}

export interface SourceCreate {
  id: string
  url: string
  reconnect_delay_ms?: number
  max_reconnect_delay_ms?: number
  degraded_threshold?: number
  max_reconnect_attempts?: number
  open_timeout_ms?: number
  read_timeout_ms?: number
}

export interface PipelineNodeInfo {
  id: string
  type: string
  with?: Record<string, string>
}

export interface PipelineEdgeInfo {
  from: string
  to: string
  capacity: number
  drop_policy: DropPolicy
}

export interface PipelineInfo {
  id: string
  nodes: PipelineNodeInfo[]
  edges: PipelineEdgeInfo[]
}

export interface StageConfig {
  id: string
  type: string
  with?: Record<string, string>
}

export interface EdgeConfig {
  from: string
  to: string
  capacity?: number
  drop_policy?: DropPolicy
}

export interface PipelineCreate {
  id: string
  nodes: StageConfig[]
  edges?: EdgeConfig[]
}

export interface ModelInfo {
  id: string
  backend: string
  version: string
  input_shape: string
  batch_size: number
  instance_count: number
}

export interface ModelLoad {
  id: string
  backend?: BackendType
  version?: ModelVersion
  model_type?: ModelType
  onnx_path?: string
  engine_path?: string
  batch_size?: number
  input_shape?: { c: number; h: number; w: number }
  conf_thresh?: number
  nms_thresh?: number
  device_id?: number
  num_classes?: number
  instance_count?: number
}

export interface TaskInfo {
  id: string
  state: TaskState
  source_id?: string
  pipeline_id?: string
  sample_fps?: number
  sampling_mode?: SamplingMode
  use_hwdec?: boolean
  use_ascend_dvpp?: boolean
  ascend_device_id?: number
}

export interface TaskCreate {
  id: string
  source_id: string
  pipeline_id: string
  sample_fps?: number
  sampling_mode?: SamplingMode
  use_hwdec?: boolean
  use_ascend_dvpp?: boolean
  ascend_device_id?: number
}

export interface OkResponse {
  status: string
  id: string
}
