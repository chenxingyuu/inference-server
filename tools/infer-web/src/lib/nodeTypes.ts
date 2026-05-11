export interface NodeTypeDef {
  type: string
  category: string
  withTemplate: { k: string; v: string }[]
}

export const NODE_TYPE_DEFS: NodeTypeDef[] = [
  { type: 'source.rtsp',       category: 'source',      withTemplate: [] },
  { type: 'source.file',       category: 'source',      withTemplate: [{ k: 'loop', v: 'false' }] },
  { type: 'decode.ffmpeg',     category: 'decode',      withTemplate: [] },
  { type: 'infer.sahiScheduler', category: 'infer',     withTemplate: [
    { k: 'tile_width',         v: '960'  },
    { k: 'tile_height',        v: '1144' },
    { k: 'overlap_ratio',      v: '0.25' },
    { k: 'y_overlap_ratio',    v: '0'    },
    { k: 'full_interval',      v: '5'    },
    { k: 'roi_expand_ratio',   v: '1.3'  },
    { k: 'max_tiles_per_frame', v: '48'  },
  ]},
  { type: 'infer.engine',      category: 'infer',       withTemplate: [{ k: 'model_id', v: '' }] },
  { type: 'post.sahiMerge',    category: 'postprocess', withTemplate: [
    { k: 'merge_iou',         v: '0.55' },
    { k: 'stale_timeout_ms',  v: '2000' },
  ]},
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
    { k: 'output_url',                  v: ''            },
    { k: 'protocol',                    v: 'rtsp'        },
    { k: 'fps',                         v: '25'          },
    { k: 'gop',                         v: '0'           },
    { k: 'bitrate_kbps',                v: '4000'        },
    { k: 'queue_capacity',              v: '256'         },
    { k: 'drop_policy',                 v: 'drop_oldest' },
    { k: 'reconnect_initial_ms',        v: '1000'        },
    { k: 'reconnect_max_ms',            v: '10000'       },
    { k: 'encoder',                     v: 'ffmpeg_x264' },
    { k: 'ascend_device_id',            v: '-1'          },
    { k: 'draw_conf_thresh',            v: '0.25'        },
    { k: 'line_thickness',              v: '2'           },
    { k: 'draw_timestamp',              v: 'false'       },
    { k: 'draw_timestamp_with_stream_id', v: 'false'     },
    { k: 'timestamp_x',                 v: '10'          },
    { k: 'timestamp_y',                 v: '28'          },
    { k: 'timestamp_font_scale',        v: '0.6'         },
    { k: 'timestamp_thickness',         v: '1'           },
  ]},
]

export const NODE_CATEGORIES = ['source', 'decode', 'infer', 'postprocess', 'track', 'join', 'archive', 'sink'] as const

export const getNodeTypeDef = (type: string) =>
  NODE_TYPE_DEFS.find((d) => d.type === type)
