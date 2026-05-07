export interface NodeTypeDef {
  type: string
  category: string
  withTemplate: { k: string; v: string }[]
}

export const NODE_TYPE_DEFS: NodeTypeDef[] = [
  { type: 'source.rtsp',       category: 'source',      withTemplate: [] },
  { type: 'source.file',       category: 'source',      withTemplate: [{ k: 'loop', v: 'false' }] },
  { type: 'decode.ffmpeg',     category: 'decode',      withTemplate: [] },
  { type: 'infer.engine',      category: 'infer',       withTemplate: [{ k: 'model_id', v: '' }] },
  { type: 'postprocess.yolo',  category: 'postprocess', withTemplate: [] },
  { type: 'track.bytetrack',   category: 'track',       withTemplate: [
    { k: 'high_det_thresh',     v: '0.5' },
    { k: 'low_det_thresh',      v: '0.1' },
    { k: 'match_iou_thresh',    v: '0.3' },
    { k: 'min_hits_to_confirm', v: '2'   },
    { k: 'max_lost_frames',     v: '30'  },
  ]},
  { type: 'join.byFrameId',    category: 'join',        withTemplate: [] },
  { type: 'archive.raw',       category: 'archive',     withTemplate: [] },
  { type: 'sink.publish',      category: 'sink',        withTemplate: [] },
  { type: 'sink.ffplay',       category: 'sink',        withTemplate: [
    { k: 'fps',             v: '5'           },
    { k: 'draw_conf_thresh', v: '0.25'       },
    { k: 'drop_policy',     v: 'drop_oldest' },
  ]},
  { type: 'sink.stream',       category: 'sink',        withTemplate: [
    { k: 'output_url', v: ''     },
    { k: 'protocol',   v: 'rtsp' },
    { k: 'fps',        v: '25'   },
  ]},
]

export const NODE_CATEGORIES = ['source', 'decode', 'infer', 'postprocess', 'track', 'join', 'archive', 'sink'] as const

export const getNodeTypeDef = (type: string) =>
  NODE_TYPE_DEFS.find((d) => d.type === type)
