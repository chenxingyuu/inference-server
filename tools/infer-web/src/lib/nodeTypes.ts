export interface ParamDef {
  k: string
  v: string
  description?: string
}

export interface NodeTypeDef {
  type: string
  category: string
  withTemplate: ParamDef[]
}

export const NODE_TYPE_DEFS: NodeTypeDef[] = [
  { type: 'source.rtsp',       category: 'source',      withTemplate: [] },
  { type: 'source.file',       category: 'source',      withTemplate: [
    { k: 'loop', v: 'false', description: 'Loop the file indefinitely (true/false)' },
  ]},
  { type: 'decode.ffmpeg',     category: 'decode',      withTemplate: [] },
  { type: 'infer.sahiScheduler', category: 'infer',     withTemplate: [
    { k: 'tile_width',          v: '960',  description: 'Width of each SAHI tile in pixels' },
    { k: 'tile_height',         v: '1144', description: 'Height of each SAHI tile in pixels' },
    { k: 'overlap_ratio',       v: '0.25', description: 'Horizontal overlap ratio between adjacent tiles (0.0–1.0)' },
    { k: 'y_overlap_ratio',     v: '0',    description: 'Vertical overlap ratio between adjacent tiles (0.0–1.0)' },
    { k: 'full_interval',       v: '5',    description: 'Run a full-frame inference pass every N frames' },
    { k: 'roi_expand_ratio',    v: '1.3',  description: 'Expand ROI bounds by this ratio before cropping (e.g. 1.3 = +30%)' },
    { k: 'max_tiles_per_frame', v: '48',   description: 'Maximum tiles to process per frame; excess tiles are skipped' },
  ]},
  { type: 'infer.engine',      category: 'infer',       withTemplate: [
    { k: 'model_id', v: '', description: 'ID of the model registered in the model repository' },
  ]},
  { type: 'post.sahiMerge',    category: 'postprocess', withTemplate: [
    { k: 'merge_iou',        v: '0.55', description: 'IoU threshold for merging overlapping tile detections (0.0–1.0)' },
    { k: 'merge_ios',        v: '0.8',  description: 'Intersection-over-smaller threshold for merging detections (0.0–1.0)' },
    { k: 'stale_timeout_ms', v: '2000', description: 'Discard tile results older than this value (ms)' },
  ]},
  { type: 'postprocess.yolo',  category: 'postprocess', withTemplate: [] },
  { type: 'track.bytetrack',   category: 'track',       withTemplate: [
    { k: 'high_det_thresh',     v: '0.5', description: 'Confidence threshold for high-quality detections used in primary matching' },
    { k: 'low_det_thresh',      v: '0.1', description: 'Confidence threshold for low-quality detections used in secondary matching' },
    { k: 'match_iou_thresh',    v: '0.3', description: 'Minimum IoU to associate a detection with an existing track' },
    { k: 'min_hits_to_confirm', v: '2',   description: 'Consecutive detections required to confirm a new track' },
    { k: 'max_lost_frames',     v: '30',  description: 'Frames a track can disappear before being dropped' },
  ]},
  { type: 'join.byFrameId',    category: 'join',        withTemplate: [] },
  { type: 'archive.raw',       category: 'archive',     withTemplate: [] },
  { type: 'sink.publish',      category: 'sink',        withTemplate: [
    { k: 'to', v: '', description: 'Name of the publisher to route results to (e.g. kafka, redis)' },
  ]},
  { type: 'sink.ffplay',       category: 'sink',        withTemplate: [
    { k: 'fps',             v: '5',           description: 'Display frame rate cap' },
    { k: 'draw_conf_thresh', v: '0.25',       description: 'Only draw bounding boxes above this confidence (0.0–1.0)' },
    { k: 'drop_policy',     v: 'drop_oldest', description: 'Queue overflow policy: block, drop_oldest, or drop_newest' },
  ]},
  { type: 'sink.stream',       category: 'sink',        withTemplate: [
    { k: 'output_url',                    v: '',            description: 'Output stream URL (e.g. rtsp://host/live/cam)' },
    { k: 'protocol',                      v: 'rtsp',        description: 'Streaming protocol: rtsp or rtmp' },
    { k: 'fps',                           v: '25',          description: 'Output frame rate' },
    { k: 'gop',                           v: '0',           description: 'Group-of-pictures size; 0 = auto (typically 2× fps)' },
    { k: 'bitrate_kbps',                  v: '4000',        description: 'Target video bitrate in Kbps' },
    { k: 'queue_capacity',                v: '256',         description: 'Internal frame queue depth before drop policy applies' },
    { k: 'drop_policy',                   v: 'drop_oldest', description: 'Queue overflow policy: block, drop_oldest, or drop_newest' },
    { k: 'reconnect_initial_ms',          v: '1000',        description: 'Initial reconnect delay in milliseconds' },
    { k: 'reconnect_max_ms',              v: '10000',       description: 'Maximum reconnect delay with exponential backoff (ms)' },
    { k: 'encoder',                       v: 'ffmpeg_x264', description: 'Video encoder to use (e.g. ffmpeg_x264, ffmpeg_h264_nvenc)' },
    { k: 'ascend_device_id',              v: '-1',          description: 'Ascend device ID for hardware encode; -1 = disabled' },
    { k: 'draw_conf_thresh',              v: '0.25',        description: 'Only draw bounding boxes above this confidence (0.0–1.0)' },
    { k: 'line_thickness',                v: '2',           description: 'Bounding box outline thickness in pixels' },
    { k: 'draw_timestamp',                v: 'false',       description: 'Overlay timestamp on each frame (true/false)' },
    { k: 'draw_timestamp_with_stream_id', v: 'false',       description: 'Include stream ID in the timestamp overlay (true/false)' },
    { k: 'timestamp_x',                   v: '10',          description: 'Timestamp overlay X position in pixels' },
    { k: 'timestamp_y',                   v: '28',          description: 'Timestamp overlay Y position in pixels' },
    { k: 'timestamp_font_scale',          v: '0.6',         description: 'Font scale factor for the timestamp text' },
    { k: 'timestamp_thickness',           v: '1',           description: 'Line thickness for the timestamp text' },
  ]},
]

export const NODE_CATEGORIES = ['source', 'decode', 'infer', 'postprocess', 'track', 'join', 'archive', 'sink'] as const

export const getNodeTypeDef = (type: string) =>
  NODE_TYPE_DEFS.find((d) => d.type === type)
