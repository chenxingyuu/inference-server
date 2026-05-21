export interface ParamDef {
  k: string
  v: string
  description?: { en: string; zh: string }
}

export interface NodeTypeDef {
  type: string
  category: string
  withTemplate: ParamDef[]
}

export const NODE_TYPE_DEFS: NodeTypeDef[] = [
  { type: 'source.rtsp',       category: 'source',      withTemplate: [] },
  { type: 'source.file',       category: 'source',      withTemplate: [
    { k: 'loop', v: 'false', description: {
      en: 'Loop the file indefinitely (true/false)',
      zh: '是否循环播放文件（true/false）',
    }},
  ]},
  { type: 'decode.ffmpeg',     category: 'decode',      withTemplate: [] },
  { type: 'infer.sahiScheduler', category: 'infer',     withTemplate: [
    { k: 'tile_width',          v: '960',  description: { en: 'Width of each SAHI tile in pixels',                                    zh: '每个 SAHI 分块的宽度（像素）' } },
    { k: 'tile_height',         v: '1144', description: { en: 'Height of each SAHI tile in pixels',                                   zh: '每个 SAHI 分块的高度（像素）' } },
    { k: 'overlap_ratio',       v: '0.25', description: { en: 'Horizontal overlap ratio between adjacent tiles (0.0–1.0)',             zh: '相邻分块之间的水平重叠比例（0.0–1.0）' } },
    { k: 'y_overlap_ratio',     v: '0',    description: { en: 'Vertical overlap ratio between adjacent tiles (0.0–1.0)',               zh: '相邻分块之间的垂直重叠比例（0.0–1.0）' } },
    { k: 'full_interval',       v: '5',    description: { en: 'Run a full-frame inference pass every N frames',                        zh: '每隔 N 帧执行一次全图推理' } },
    { k: 'roi_expand_ratio',    v: '1.3',  description: { en: 'Expand ROI bounds by this ratio before cropping (e.g. 1.3 = +30%)',    zh: '裁剪前按此比例扩展 ROI 范围（如 1.3 = 扩大 30%）' } },
    { k: 'max_tiles_per_frame', v: '48',   description: { en: 'Maximum tiles to process per frame; excess tiles are skipped',         zh: '每帧最多处理的分块数，超出部分跳过' } },
  ]},
  { type: 'infer.engine',      category: 'infer',       withTemplate: [
    { k: 'model_id', v: '', description: { en: 'ID of the model registered in the model repository', zh: '在模型仓库中注册的模型 ID' } },
  ]},
  { type: 'post.sahiMerge',    category: 'postprocess', withTemplate: [
    { k: 'merge_iou',        v: '0.55', description: { en: 'IoU threshold for merging overlapping tile detections (0.0–1.0)',       zh: '合并来自不同分块的重叠检测框的 IoU 阈值（0.0–1.0）' } },
    { k: 'merge_ios',        v: '0.8',  description: { en: 'Intersection-over-smaller threshold for merging detections (0.0–1.0)', zh: '合并检测框的 IoS（较小框交集占比）阈值（0.0–1.0）' } },
    { k: 'stale_timeout_ms', v: '2000', description: { en: 'Discard tile results older than this value (ms)',                       zh: '丢弃超过此时间的分块结果（毫秒）' } },
  ]},
  { type: 'postprocess.yolo',  category: 'postprocess', withTemplate: [] },
  { type: 'track.bytetrack',   category: 'track',       withTemplate: [
    { k: 'high_det_thresh',     v: '0.5', description: { en: 'Confidence threshold for high-quality detections (primary matching)',    zh: '高质量检测框的置信度阈值，用于主匹配' } },
    { k: 'low_det_thresh',      v: '0.1', description: { en: 'Confidence threshold for low-quality detections (secondary matching)',   zh: '低质量检测框的置信度阈值，用于二次匹配' } },
    { k: 'match_iou_thresh',    v: '0.3', description: { en: 'Minimum IoU to associate a detection with an existing track',           zh: '检测框与已有轨迹关联的最小 IoU' } },
    { k: 'min_hits_to_confirm', v: '2',   description: { en: 'Consecutive detections required to confirm a new track',                zh: '确认新轨迹所需的连续检测帧数' } },
    { k: 'max_lost_frames',     v: '30',  description: { en: 'Frames a track can disappear before being dropped',                     zh: '轨迹消失超过此帧数后被删除' } },
  ]},
  { type: 'join.byFrameId',    category: 'join',        withTemplate: [] },
  { type: 'archive.raw',       category: 'archive',     withTemplate: [] },
  { type: 'sink.publish',      category: 'sink',        withTemplate: [
    { k: 'to', v: '', description: { en: 'Name of the publisher to route results to (e.g. kafka, redis)', zh: '结果路由到的发布者名称（如 kafka、redis）' } },
  ]},
  { type: 'sink.ffplay',       category: 'sink',        withTemplate: [
    { k: 'fps',              v: '5',           description: { en: 'Display frame rate cap',                                                                    zh: '显示帧率上限' } },
    { k: 'draw_conf_thresh', v: '0.25',        description: { en: 'Only draw bounding boxes above this confidence (0.0–1.0)',                                  zh: '仅绘制置信度高于此阈值的检测框（0.0–1.0）' } },
    { k: 'drop_policy',      v: 'drop_oldest', description: { en: 'Queue overflow policy: block, drop_oldest, or drop_newest',                                 zh: '队列溢出策略：block（阻塞）、drop_oldest（丢弃最旧）、drop_newest（丢弃最新）' } },
  ]},
  { type: 'sink.stream',       category: 'sink',        withTemplate: [
    { k: 'output_url',                    v: '',            description: { en: 'Output stream URL (e.g. rtsp://host/live/cam)',                                 zh: '输出流地址（如 rtsp://host/live/cam）' } },
    { k: 'protocol',                      v: 'rtsp',        description: { en: 'Streaming protocol: rtsp or rtmp',                                             zh: '推流协议：rtsp 或 rtmp' } },
    { k: 'fps',                           v: '25',          description: { en: 'Output frame rate',                                                            zh: '输出帧率' } },
    { k: 'gop',                           v: '0',           description: { en: 'Group-of-pictures size; 0 = auto (typically 2× fps)',                          zh: '关键帧间隔；0 = 自动（通常为 fps 的 2 倍）' } },
    { k: 'bitrate_kbps',                  v: '4000',        description: { en: 'Target video bitrate in Kbps',                                                 zh: '目标视频码率（Kbps）' } },
    { k: 'queue_capacity',                v: '256',         description: { en: 'Internal frame queue depth before drop policy applies',                         zh: '触发丢帧策略前的帧队列深度' } },
    { k: 'drop_policy',                   v: 'drop_oldest', description: { en: 'Queue overflow policy: block, drop_oldest, or drop_newest',                    zh: '队列溢出策略：block（阻塞）、drop_oldest（丢弃最旧）、drop_newest（丢弃最新）' } },
    { k: 'reconnect_initial_ms',          v: '1000',        description: { en: 'Initial reconnect delay in milliseconds',                                      zh: '初始重连延迟（毫秒）' } },
    { k: 'reconnect_max_ms',              v: '10000',       description: { en: 'Maximum reconnect delay with exponential backoff (ms)',                         zh: '指数退避的最大重连延迟（毫秒）' } },
    { k: 'encoder',                       v: 'ffmpeg_x264', description: { en: 'Video encoder to use (e.g. ffmpeg_x264, ffmpeg_h264_nvenc)',                   zh: '使用的视频编码器（如 ffmpeg_x264、ffmpeg_h264_nvenc）' } },
    { k: 'ascend_device_id',              v: '-1',          description: { en: 'Ascend device ID for hardware encode; -1 = disabled',                          zh: '硬件编码使用的 Ascend 设备 ID；-1 = 禁用' } },
    { k: 'draw_conf_thresh',              v: '0.25',        description: { en: 'Only draw bounding boxes above this confidence (0.0–1.0)',                     zh: '仅绘制置信度高于此阈值的检测框（0.0–1.0）' } },
    { k: 'line_thickness',                v: '2',           description: { en: 'Bounding box outline thickness in pixels',                                      zh: '检测框边线粗细（像素）' } },
    { k: 'draw_timestamp',                v: 'false',       description: { en: 'Overlay timestamp on each frame (true/false)',                                  zh: '在每帧上叠加时间戳（true/false）' } },
    { k: 'draw_timestamp_with_stream_id', v: 'false',       description: { en: 'Include stream ID in the timestamp overlay (true/false)',                      zh: '在时间戳中包含流 ID（true/false）' } },
    { k: 'timestamp_x',                   v: '10',          description: { en: 'Timestamp overlay X position in pixels',                                       zh: '时间戳叠加的 X 坐标（像素）' } },
    { k: 'timestamp_y',                   v: '28',          description: { en: 'Timestamp overlay Y position in pixels',                                       zh: '时间戳叠加的 Y 坐标（像素）' } },
    { k: 'timestamp_font_scale',          v: '0.6',         description: { en: 'Font scale factor for the timestamp text',                                      zh: '时间戳文字的字体缩放比例' } },
    { k: 'timestamp_thickness',           v: '1',           description: { en: 'Line thickness for the timestamp text',                                         zh: '时间戳文字的线条粗细' } },
  ]},
]

export const NODE_CATEGORIES = ['source', 'decode', 'infer', 'postprocess', 'track', 'join', 'archive', 'sink'] as const

export const getNodeTypeDef = (type: string) =>
  NODE_TYPE_DEFS.find((d) => d.type === type)
