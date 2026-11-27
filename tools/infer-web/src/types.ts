export type SourceState = 'CONNECTING' | 'STREAMING' | 'RECONNECTING' | 'DEGRADED' | 'STOPPED' | 'FAILED'
export type TaskState = 'stopped' | 'starting' | 'running' | 'failed'
export type BackendType = 'cpu' | 'trt' | 'ascend'
export type ModelVersion = 'v5' | 'v8' | 'v11' | 'v26'
export type ModelType = 'detector' | 'classifier'
export type DropPolicy = 'block' | 'drop_oldest' | 'drop_newest'
export type SamplingMode = 'frame_count' | 'time_based'

export interface SourceAvailability {
  status: 'online' | 'offline' | 'unchecked' | 'unsupported'
  basis: 'probe' | 'none'
  observed_at_ms?: number
  reason?: string
}

export interface SourceProbeResult {
  passed: boolean
  checked_at_ms: number
  elapsed_ms: number
  reason?: string
}

export interface SourceInfo {
  id: string
  url: string
  state: SourceState
  reconnect_count: number
  availability?: SourceAvailability
  last_probe?: SourceProbeResult | null
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
  model_type?: string
  num_classes?: number
  conf_thresh?: number
  nms_thresh?: number
  device_id?: number
  preferred_batch_sizes?: number[]
  max_queue_delay_us?: number
  class_names?: string[]
}

export interface ModelLoad {
  id: string
  backend?: BackendType
  version?: ModelVersion
  model_type?: ModelType
  onnx_path?: string
  engine_path?: string
  om_paths?: Record<string, string>
  batch_size?: number
  input_shape?: { c: number; h: number; w: number }
  conf_thresh?: number
  nms_thresh?: number
  device_id?: number
  num_classes?: number
  instance_count?: number
}

export interface RepositoryModel {
  id: string
  backend: string
  version: string
  model_type: string
  batch_size: number
  num_classes: number
  conf_thresh: number
  nms_thresh: number
  instance_count: number
  loaded: boolean
}

export interface ModelDetail {
  model: ModelInfo
  registered: boolean
  repository: { version: string; weight_paths: string[] } | null
  repository_error: boolean
  pipelines: { id: string; nodes: string[]; cascade_nodes: string[]; tasks: Pick<TaskInfo, 'id' | 'state'>[] }[]
  cascade_parents: string[]
  cascade_children: { model_id: string; attribute_key: string; trigger_classes: number[]; crop_expand: number }[]
}

export interface ModelTestResult {
  model_type: ModelType
  backend: BackendType | 'mps' | 'tensorrt'
  conf_thresh: number
  nms_thresh: number | null
  width: number
  height: number
  preview_jpeg: string
  results: {
    class_id: number
    class_name: string
    confidence: number
    bbox?: { x1: number; y1: number; x2: number; y2: number }
  }[]
}

export interface TaskInfo {
  id: string
  state: TaskState
  failure?: { code: string; message: string }
  cleanup_pending?: boolean
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

export interface ArchiveAnalysis {
  token: string
  suggested_id?: string
  backend: string
  yolo_version: string
  model_type: string
  batch_size: number
  conf_thresh: number
  nms_thresh: number
  num_classes: number
  instance_count: number
  input_size?: [number, number]
  device_id: number
  device_ids?: number[]
  has_config: boolean
  files_found: string[]
}

export interface OkResponse {
  status: string
  id: string
}
