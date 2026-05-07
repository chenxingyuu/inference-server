import { createContext, useContext, useState, type ReactNode } from 'react'

export type Lang = 'en' | 'zh'

// ─── Dictionaries ──────────────────────────────────────────────────────────

const en = {
  'nav.dashboard': 'Dashboard',
  'nav.sources':   'Sources',
  'nav.models':    'Models',
  'nav.pipelines': 'Pipelines',
  'nav.tasks':     'Tasks',

  'sidebar.server':  'Server',
  'sidebar.api_key': 'API Key',
  'sidebar.not_set': 'not set',

  'health.ok':          'Engine healthy',
  'health.engine_down': 'C++ engine not connected — Go server is reachable, start the inference engine',
  'health.unreachable': 'Cannot reach server — check the Server URL in the sidebar ({url})',
  'health.checking':    'Checking…',

  'common.id':      'ID',
  'common.state':   'State',
  'common.actions': 'Actions',
  'common.cancel':  'Cancel',
  'common.save':    'Save',
  'common.remove':  'Remove',

  'dashboard.title':           'Dashboard',
  'dashboard.subtitle':        'System overview — auto-refreshes every 5–30s',
  'dashboard.source_states':   'Source States',
  'dashboard.task_states':     'Task States',
  'dashboard.more':            '+{n} more — view all in Sources',
  'dashboard.stat.sources':    'Sources',
  'dashboard.stat.models':     'Models',
  'dashboard.stat.pipelines':  'Pipelines',
  'dashboard.stat.tasks':      'Tasks',
  'dashboard.sources.sub':     '{streaming} streaming · {degraded} degraded',
  'dashboard.sources.empty':   'none configured',
  'dashboard.models.sub':      '{backends}',
  'dashboard.models.empty':    'none loaded',
  'dashboard.pipelines.sub':   '{nodes} nodes total',
  'dashboard.pipelines.empty': 'none defined',
  'dashboard.tasks.sub':       '{running} running',
  'dashboard.tasks.empty':     'none created',

  'sources.title':                 'Sources',
  'sources.subtitle':              'Video stream sources (RTSP / file)',
  'sources.add':                   '+ Add Source',
  'sources.modal_title':           'Add Source',
  'sources.col.reconnects':        'Reconnects',
  'sources.col.url':               'URL',
  'sources.field.id':              'Source ID',
  'sources.field.id_hint':         'Unique identifier',
  'sources.field.url':             'URL',
  'sources.field.url_hint':        'rtsp:// or file://…',
  'sources.advanced':              'Advanced options',
  'sources.field.reconnect_delay': 'Reconnect delay (ms)',
  'sources.field.max_delay':       'Max reconnect delay (ms)',
  'sources.field.max_attempts':    'Max reconnect attempts',
  'sources.field.open_timeout':    'Open timeout (ms)',
  'sources.field.read_timeout':    'Read timeout (ms)',
  'sources.empty':                 'No sources configured. Add one to start streaming.',
  'sources.adding':                'Adding…',

  'models.title':                  'Models',
  'models.subtitle':               'Loaded inference models',
  'models.load':                   '+ Load Model',
  'models.modal_title':            'Load Model',
  'models.col.backend':            'Backend',
  'models.col.version':            'Version',
  'models.col.input_shape':        'Input Shape',
  'models.col.batch':              'Batch',
  'models.col.instances':          'Instances',
  'models.field.id':               'Model ID',
  'models.field.backend':          'Backend',
  'models.field.version':          'Version',
  'models.field.model_type':       'Model Type',
  'models.field.instances':        'Instances',
  'models.field.onnx_path':        'ONNX Path',
  'models.field.onnx_path_hint':   'Required for cpu backend',
  'models.field.engine_path':      'Engine Path',
  'models.field.engine_path_hint': 'Required for trt / ascend backend',
  'models.field.batch_size':       'Batch Size',
  'models.field.num_classes':      'Num Classes',
  'models.field.conf_thresh':      'Conf Thresh',
  'models.field.nms_thresh':       'NMS Thresh',
  'models.empty':                  'No models loaded. Load a model to enable inference.',
  'models.loading':                'Loading…',

  'pipelines.title':           'Pipelines',
  'pipelines.subtitle':        'Processing node graphs',
  'pipelines.add':             '+ Add Pipeline',
  'pipelines.modal_title':     'Add Pipeline',
  'pipelines.col.nodes':       'Nodes',
  'pipelines.col.edges':       'Edges',
  'pipelines.field.id':        'Pipeline ID',
  'pipelines.section.nodes':   'Nodes',
  'pipelines.section.edges':   'Edges (optional)',
  'pipelines.add_node':        '+ Add node',
  'pipelines.add_edge':        '+ Add edge',
  'pipelines.add_param':       '+ param',
  'pipelines.col.node_id':     'Node ID',
  'pipelines.col.type':        'Type',
  'pipelines.col.type_custom': 'Custom',
  'pipelines.col.key':         'Key',
  'pipelines.col.value':       'Value',
  'pipelines.col.from':        'From',
  'pipelines.col.to':          'To',
  'pipelines.col.capacity':    'Capacity',
  'pipelines.col.drop_policy': 'Drop Policy',
  'pipelines.empty':           'No pipelines defined. Add one to connect sources to inference.',
  'pipelines.adding':          'Adding…',
  'pipelines.copy':            'Copy',
  'pipelines.edit':            'Edit',
  'pipelines.modal_edit_title': 'Edit Pipeline',
  'pipelines.saving':          'Saving…',
  'pipelines.save':            'Save',

  'tasks.title':               'Tasks',
  'tasks.subtitle_stats':      '{total} total · {running} running',
  'tasks.subtitle_empty':      'Bind a source to a pipeline and start inference',
  'tasks.add':                 '+ New Task',
  'tasks.modal_title':         'New Task',
  'tasks.field.id':            'Task ID',
  'tasks.field.source':        'Source',
  'tasks.field.source_hint':   'Must be added in Sources',
  'tasks.field.pipeline':      'Pipeline',
  'tasks.field.pipeline_hint': 'Must be added in Pipelines',
  'tasks.field.sample_fps':    'Sample FPS',
  'tasks.field.sampling':      'Sampling Mode',
  'tasks.field.hwdec':         'Enable hardware decode (hwdec)',
  'tasks.select_source':       '— select source —',
  'tasks.select_pipeline':     '— select pipeline —',
  'tasks.empty':               'No tasks. Create one to start inference.',
  'tasks.creating':            'Creating…',
} as const

const zh: Record<keyof typeof en, string> = {
  'nav.dashboard': '控制台',
  'nav.sources':   '数据源',
  'nav.models':    '模型',
  'nav.pipelines': '流水线',
  'nav.tasks':     '任务',

  'sidebar.server':  '服务器',
  'sidebar.api_key': 'API 密钥',
  'sidebar.not_set': '未设置',

  'health.ok':          '引擎运行正常',
  'health.engine_down': 'C++ 引擎未连接 — Go server 可达，请启动推理引擎',
  'health.unreachable': '无法连接服务器 — 请检查侧边栏中的服务器地址（{url}）',
  'health.checking':    '检测中…',

  'common.id':      'ID',
  'common.state':   '状态',
  'common.actions': '操作',
  'common.cancel':  '取消',
  'common.save':    '保存',
  'common.remove':  '删除',

  'dashboard.title':           '控制台',
  'dashboard.subtitle':        '系统概览 — 每 5–30 秒自动刷新',
  'dashboard.source_states':   '数据源状态',
  'dashboard.task_states':     '任务状态',
  'dashboard.more':            '+{n} 条 — 前往数据源页面查看全部',
  'dashboard.stat.sources':    '数据源',
  'dashboard.stat.models':     '模型',
  'dashboard.stat.pipelines':  '流水线',
  'dashboard.stat.tasks':      '任务',
  'dashboard.sources.sub':     '{streaming} 流式 · {degraded} 降级',
  'dashboard.sources.empty':   '尚未配置',
  'dashboard.models.sub':      '{backends}',
  'dashboard.models.empty':    '未加载',
  'dashboard.pipelines.sub':   '共 {nodes} 个节点',
  'dashboard.pipelines.empty': '尚未定义',
  'dashboard.tasks.sub':       '{running} 个运行中',
  'dashboard.tasks.empty':     '尚未创建',

  'sources.title':                 '数据源',
  'sources.subtitle':              '视频流数据源（RTSP / 文件）',
  'sources.add':                   '+ 添加数据源',
  'sources.modal_title':           '添加数据源',
  'sources.col.reconnects':        '重连次数',
  'sources.col.url':               '地址',
  'sources.field.id':              '数据源 ID',
  'sources.field.id_hint':         '唯一标识符',
  'sources.field.url':             '地址',
  'sources.field.url_hint':        'rtsp:// 或 file://…',
  'sources.advanced':              '高级选项',
  'sources.field.reconnect_delay': '重连间隔（ms）',
  'sources.field.max_delay':       '最大重连间隔（ms）',
  'sources.field.max_attempts':    '最大重连次数',
  'sources.field.open_timeout':    '连接超时（ms）',
  'sources.field.read_timeout':    '读取超时（ms）',
  'sources.empty':                 '暂无数据源，添加一个以开始流式传输。',
  'sources.adding':                '添加中…',

  'models.title':                  '模型',
  'models.subtitle':               '已加载的推理模型',
  'models.load':                   '+ 加载模型',
  'models.modal_title':            '加载模型',
  'models.col.backend':            '后端',
  'models.col.version':            '版本',
  'models.col.input_shape':        '输入形状',
  'models.col.batch':              '批次',
  'models.col.instances':          '实例数',
  'models.field.id':               '模型 ID',
  'models.field.backend':          '后端',
  'models.field.version':          '版本',
  'models.field.model_type':       '模型类型',
  'models.field.instances':        '实例数',
  'models.field.onnx_path':        'ONNX 路径',
  'models.field.onnx_path_hint':   'cpu 后端必填',
  'models.field.engine_path':      '引擎路径',
  'models.field.engine_path_hint': 'trt / ascend 后端必填',
  'models.field.batch_size':       '批次大小',
  'models.field.num_classes':      '类别数',
  'models.field.conf_thresh':      '置信度阈值',
  'models.field.nms_thresh':       'NMS 阈值',
  'models.empty':                  '暂无模型，加载一个以启用推理。',
  'models.loading':                '加载中…',

  'pipelines.title':           '流水线',
  'pipelines.subtitle':        '处理节点图',
  'pipelines.add':             '+ 添加流水线',
  'pipelines.modal_title':     '添加流水线',
  'pipelines.col.nodes':       '节点数',
  'pipelines.col.edges':       '边数',
  'pipelines.field.id':        '流水线 ID',
  'pipelines.section.nodes':   '节点',
  'pipelines.section.edges':   '边（可选）',
  'pipelines.add_node':        '+ 添加节点',
  'pipelines.add_edge':        '+ 添加边',
  'pipelines.add_param':       '+ 参数',
  'pipelines.col.node_id':     '节点 ID',
  'pipelines.col.type':        '类型',
  'pipelines.col.type_custom': '自定义',
  'pipelines.col.key':         '键',
  'pipelines.col.value':       '值',
  'pipelines.col.from':        '起点',
  'pipelines.col.to':          '终点',
  'pipelines.col.capacity':    '容量',
  'pipelines.col.drop_policy': '丢弃策略',
  'pipelines.empty':           '暂无流水线，添加一个以连接数据源与推理。',
  'pipelines.adding':          '添加中…',
  'pipelines.copy':            '复制',
  'pipelines.edit':            '编辑',
  'pipelines.modal_edit_title': '编辑流水线',
  'pipelines.saving':          '保存中…',
  'pipelines.save':            '保存',

  'tasks.title':               '任务',
  'tasks.subtitle_stats':      '共 {total} 个 · {running} 个运行中',
  'tasks.subtitle_empty':      '绑定数据源与流水线以启动推理',
  'tasks.add':                 '+ 新建任务',
  'tasks.modal_title':         '新建任务',
  'tasks.field.id':            '任务 ID',
  'tasks.field.source':        '数据源',
  'tasks.field.source_hint':   '需先在数据源页面添加',
  'tasks.field.pipeline':      '流水线',
  'tasks.field.pipeline_hint': '需先在流水线页面添加',
  'tasks.field.sample_fps':    '采样帧率',
  'tasks.field.sampling':      '采样模式',
  'tasks.field.hwdec':         '启用硬件解码（hwdec）',
  'tasks.select_source':       '— 选择数据源 —',
  'tasks.select_pipeline':     '— 选择流水线 —',
  'tasks.empty':               '暂无任务，创建一个以启动推理。',
  'tasks.creating':            '创建中…',
}

// ─── Context & Hook ────────────────────────────────────────────────────────

type Key = keyof typeof en

interface I18nCtx {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: Key, vars?: Record<string, string | number>) => string
}

const Ctx = createContext<I18nCtx | null>(null)

const dicts: Record<Lang, Record<string, string>> = { en, zh }

function detectLang(): Lang {
  const stored = localStorage.getItem('infer_lang')
  if (stored === 'en' || stored === 'zh') return stored as Lang
  return navigator.language.startsWith('zh') ? 'zh' : 'en'
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectLang)

  const setLang = (l: Lang) => {
    localStorage.setItem('infer_lang', l)
    setLangState(l)
  }

  const t = (key: Key, vars?: Record<string, string | number>): string => {
    let str: string = dicts[lang][key] ?? dicts.en[key] ?? key
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        str = str.replaceAll(`{${k}}`, String(v))
      }
    }
    return str
  }

  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>
}

export function useT() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useT must be used within I18nProvider')
  return ctx
}
