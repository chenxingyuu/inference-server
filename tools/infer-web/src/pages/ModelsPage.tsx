import { Fragment, useMemo, useRef, useState, type ReactNode } from 'react'
import { useModels, useUnloadModel, useRepositoryModels, useLoadFromRepository } from '../hooks/queries'
import { Modal } from '../components/ui/Modal'
import { Field, Input, Select } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import { api } from '../lib/api'
import type { BackendType, ModelVersion, ArchiveAnalysis } from '../types'

function Tag({ children }: { children: string }) {
  return (
    <span className="inline-block font-mono text-[10px] px-1.5 py-0.5 bg-bg-elevated border border-border rounded text-ink-secondary">
      {children}
    </span>
  )
}

function StatusBadge({ loaded }: { loaded: boolean }) {
  const { t } = useT()
  return loaded ? (
    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
      {t('models.status.loaded')}
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-ink-muted">
      <span className="w-1.5 h-1.5 rounded-full bg-ink-muted/40 inline-block" />
      {t('models.status.available')}
    </span>
  )
}

function DetailItem({ label, value }: { label: string; value: ReactNode }) {
  const displayValue = value === undefined || value === null || value === '' ? '—' : value

  return (
    <div>
      <div className="label mb-1">{label}</div>
      <div className="font-mono text-[12px] text-ink-secondary break-all">{displayValue}</div>
    </div>
  )
}

function formatList(values?: Array<string | number>) {
  return values && values.length > 0 ? values.join(', ') : undefined
}

interface UploadForm {
  id: string
  backend: BackendType
  yolo_version: ModelVersion
  model_type: 'detector' | 'classifier'
  version: string
  batch_size: string
  conf_thresh: string
  nms_thresh: string
  num_classes: string
  instance_count: string
}

const UPLOAD_DEFAULTS: UploadForm = {
  id: '',
  backend: 'ascend',
  yolo_version: 'v8',
  model_type: 'detector',
  version: '1',
  batch_size: '16',
  conf_thresh: '0.25',
  nms_thresh: '0.45',
  num_classes: '80',
  instance_count: '1',
}

const FILE_INPUT_CLS = "block w-full text-sm text-ink-secondary file:mr-3 file:py-1 file:px-2 file:rounded file:border file:border-border file:text-xs file:bg-bg-elevated file:text-ink-secondary hover:file:bg-bg-base cursor-pointer"

function UploadModal({ open, onClose, onSuccess }: { open: boolean; onClose: () => void; onSuccess: () => void }) {
  const { t } = useT()
  const [mode, setMode] = useState<'manual' | 'archive'>('manual')
  const [form, setForm] = useState<UploadForm>(UPLOAD_DEFAULTS)
  const [token, setToken] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [uploaded, setUploaded] = useState(0)
  const [total, setTotal] = useState(0)
  const abortRef = useRef<AbortController | null>(null)

  // Manual mode file refs
  const batch1Ref = useRef<HTMLInputElement>(null)
  const batch4Ref = useRef<HTMLInputElement>(null)
  const batch8Ref = useRef<HTMLInputElement>(null)
  const batch16Ref = useRef<HTMLInputElement>(null)
  const engineRef = useRef<HTMLInputElement>(null)
  const onnxRef = useRef<HTMLInputElement>(null)

  // Archive mode file ref
  const archiveRef = useRef<HTMLInputElement>(null)

  const set = <K extends keyof UploadForm>(k: K, v: UploadForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const reset = () => {
    setForm(UPLOAD_DEFAULTS)
    setToken(null)
    setProgress(null)
    setUploaded(0)
    setTotal(0)
    ;[batch1Ref, batch4Ref, batch8Ref, batch16Ref, engineRef, onnxRef, archiveRef].forEach((r) => {
      if (r.current) r.current.value = ''
    })
  }

  const handleClose = () => {
    abortRef.current?.abort()
    reset()
    onClose()
  }

  const fmtBytes = (b: number) => b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1024 / 1024).toFixed(1)} MB`

  const analyze = async () => {
    const file = archiveRef.current?.files?.[0]
    if (!file) return
    const fd = new FormData()
    fd.append('archive', file)
    const ac = new AbortController()
    abortRef.current = ac
    try {
      setProgress(0)
      const result = await api.models.analyze(
        fd,
        (pct, loaded, size) => { setProgress(pct); setUploaded(loaded); setTotal(size) },
        ac.signal,
      )
      setProgress(null)
      applyAnalysis(result)
    } catch (e: unknown) {
      setProgress(null)
      alert(e instanceof Error ? e.message : 'Analysis failed')
    }
  }

  const applyAnalysis = (result: ArchiveAnalysis) => {
    setForm({
      id: result.suggested_id ?? '',
      backend: result.backend as BackendType,
      yolo_version: result.yolo_version as ModelVersion,
      model_type: result.model_type as 'detector' | 'classifier',
      version: '1',
      batch_size: String(result.batch_size),
      conf_thresh: String(result.conf_thresh),
      nms_thresh: String(result.nms_thresh),
      num_classes: String(result.num_classes),
      instance_count: String(result.instance_count),
    })
    setToken(result.token)
    setMode('manual')
  }

  const upload = async () => {
    if (!form.id) return
    const fd = new FormData()
    fd.append('id', form.id)
    fd.append('backend', form.backend)
    fd.append('yolo_version', form.yolo_version)
    fd.append('model_type', form.model_type)
    fd.append('version', form.version)
    fd.append('batch_size', form.batch_size)
    fd.append('conf_thresh', form.conf_thresh)
    fd.append('nms_thresh', form.nms_thresh)
    fd.append('num_classes', form.num_classes)
    fd.append('instance_count', form.instance_count)

    if (token) {
      fd.append('token', token)
    } else if (form.backend === 'ascend') {
      const b1 = batch1Ref.current?.files?.[0]
      if (!b1) return
      fd.append('batch_1', b1)
      if (batch4Ref.current?.files?.[0]) fd.append('batch_4', batch4Ref.current.files[0])
      if (batch8Ref.current?.files?.[0]) fd.append('batch_8', batch8Ref.current.files[0])
      if (batch16Ref.current?.files?.[0]) fd.append('batch_16', batch16Ref.current.files[0])
    } else if (form.backend === 'trt') {
      const f = engineRef.current?.files?.[0]
      if (!f) return
      fd.append('engine', f)
    } else {
      const f = onnxRef.current?.files?.[0]
      if (!f) return
      fd.append('onnx', f)
    }

    const ac = new AbortController()
    abortRef.current = ac
    try {
      setProgress(0)
      await api.models.upload(
        fd,
        (pct, loaded, size) => { setProgress(pct); setUploaded(loaded); setTotal(size) },
        ac.signal,
      )
      reset()
      onSuccess()
      onClose()
    } catch (e: unknown) {
      setProgress(null)
      alert(e instanceof Error ? e.message : 'Upload failed')
    }
  }

  const analyzing = mode === 'archive' && progress !== null
  const uploading = mode === 'manual' && progress !== null

  const canAnalyze = !!archiveRef.current?.files?.[0] && !analyzing
  const canSubmit = (() => {
    if (!form.id || uploading) return false
    if (token) return true
    if (form.backend === 'ascend') return !!batch1Ref.current?.files?.[0]
    if (form.backend === 'trt') return !!engineRef.current?.files?.[0]
    return !!onnxRef.current?.files?.[0]
  })()

  return (
    <Modal open={open} onClose={handleClose} title={t('models.upload_modal_title')} width="max-w-xl">
      <div className="space-y-4">
        {/* Mode tabs */}
        <div className="flex gap-1 p-0.5 bg-bg-elevated rounded-lg w-fit">
          {(['manual', 'archive'] as const).map((m) => (
            <button
              key={m}
              onClick={() => { if (!analyzing && !uploading) setMode(m) }}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${mode === m ? 'bg-bg-base shadow-sm text-ink-primary font-medium' : 'text-ink-muted hover:text-ink-secondary'}`}
            >
              {m === 'manual' ? t('models.mode_manual') : t('models.mode_archive')}
            </button>
          ))}
        </div>

        {/* Archive tab */}
        {mode === 'archive' && (
          <div className="space-y-3">
            <Field label={t('models.archive_select')}>
              <input
                ref={archiveRef}
                type="file"
                accept=".zip,.tar.gz,.tgz"
                disabled={analyzing}
                onChange={() => setToken(null)}
                className={FILE_INPUT_CLS}
              />
            </Field>
            <p className="text-[11px] text-ink-muted leading-relaxed">{t('models.archive_hint')}</p>
            {analyzing && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-ink-secondary">
                  <span>{t('models.archive_analyzing')}</span>
                  <span>{total > 0 ? `${fmtBytes(uploaded)} / ${fmtBytes(total)}` : `${progress}%`}</span>
                </div>
                <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full transition-all duration-200" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}
            <div className="flex justify-end gap-2 pt-1">
              <button onClick={handleClose} className="btn-ghost">{t('common.cancel')}</button>
              <button onClick={analyze} disabled={!canAnalyze} className="btn-primary">
                {analyzing ? t('models.archive_analyzing') : t('models.mode_archive')}
              </button>
            </div>
          </div>
        )}

        {/* Manual tab */}
        {mode === 'manual' && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <Field label={t('common.id')}>
                <div className="relative">
                  <Input
                    placeholder="yolo-v8-det"
                    value={form.id}
                    onChange={(e) => set('id', e.target.value)}
                    disabled={uploading}
                  />
                  {token && (
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-medium text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                      {t('models.archive_from_archive')}
                    </span>
                  )}
                </div>
              </Field>
              <Field label={t('models.field.version_num')}>
                <Input
                  placeholder="1"
                  value={form.version}
                  onChange={(e) => set('version', e.target.value)}
                  disabled={uploading}
                />
              </Field>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <Field label={t('models.field.backend')}>
                <Select value={form.backend} onChange={(e) => set('backend', e.target.value as BackendType)} disabled={uploading}>
                  <option value="ascend">ascend</option>
                  <option value="trt">trt</option>
                  <option value="cpu">cpu</option>
                </Select>
              </Field>
              <Field label={t('models.field.yolo_version')}>
                <Select value={form.yolo_version} onChange={(e) => set('yolo_version', e.target.value as ModelVersion)} disabled={uploading}>
                  <option value="v5">v5</option>
                  <option value="v8">v8</option>
                  <option value="v11">v11</option>
                  <option value="v26">v26</option>
                </Select>
              </Field>
              <Field label={t('models.field.model_type')}>
                <Select value={form.model_type} onChange={(e) => set('model_type', e.target.value as 'detector' | 'classifier')} disabled={uploading}>
                  <option value="detector">detector</option>
                  <option value="classifier">classifier</option>
                </Select>
              </Field>
            </div>

            <div className="grid grid-cols-4 gap-3">
              <Field label={t('models.field.batch_size')}>
                <Input type="number" min={1} value={form.batch_size} onChange={(e) => set('batch_size', e.target.value)} disabled={uploading} />
              </Field>
              <Field label={t('models.field.num_classes')}>
                <Input type="number" min={1} value={form.num_classes} onChange={(e) => set('num_classes', e.target.value)} disabled={uploading} />
              </Field>
              <Field label={t('models.field.conf_thresh')}>
                <Input type="number" step={0.01} min={0} max={1} value={form.conf_thresh} onChange={(e) => set('conf_thresh', e.target.value)} disabled={uploading} />
              </Field>
              <Field label={t('models.field.nms_thresh')}>
                <Input type="number" step={0.01} min={0} max={1} value={form.nms_thresh} onChange={(e) => set('nms_thresh', e.target.value)} disabled={uploading} />
              </Field>
            </div>

            {/* File inputs — only shown in manual mode without a token */}
            {!token && (
              <div className="border border-border rounded-lg p-3 space-y-3 bg-bg-elevated/40">
                {form.backend === 'ascend' && (
                  <>
                    <Field label={t('models.field.batch_1')}>
                      <input ref={batch1Ref} type="file" accept=".om" disabled={uploading} className={FILE_INPUT_CLS} />
                    </Field>
                    <Field label={t('models.field.batch_4')}>
                      <input ref={batch4Ref} type="file" accept=".om" disabled={uploading} className={FILE_INPUT_CLS} />
                    </Field>
                    <Field label={t('models.field.batch_8')}>
                      <input ref={batch8Ref} type="file" accept=".om" disabled={uploading} className={FILE_INPUT_CLS} />
                    </Field>
                    <Field label={t('models.field.batch_16')}>
                      <input ref={batch16Ref} type="file" accept=".om" disabled={uploading} className={FILE_INPUT_CLS} />
                    </Field>
                  </>
                )}
                {form.backend === 'trt' && (
                  <Field label={t('models.field.engine_file')}>
                    <input ref={engineRef} type="file" accept=".engine" disabled={uploading} className={FILE_INPUT_CLS} />
                  </Field>
                )}
                {form.backend === 'cpu' && (
                  <Field label={t('models.field.onnx_file')}>
                    <input ref={onnxRef} type="file" accept=".onnx" disabled={uploading} className={FILE_INPUT_CLS} />
                  </Field>
                )}
              </div>
            )}

            {uploading && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-ink-secondary">
                  <span>{t('models.uploading')}</span>
                  <span>{total > 0 ? `${fmtBytes(uploaded)} / ${fmtBytes(total)}` : `${progress}%`}</span>
                </div>
                <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full transition-all duration-200" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <button onClick={handleClose} className="btn-ghost">{t('common.cancel')}</button>
              <button onClick={upload} disabled={!canSubmit} className="btn-primary">
                {uploading ? t('models.uploading') : t('models.upload')}
              </button>
            </div>
          </>
        )}
      </div>
    </Modal>
  )
}

export function ModelsPage() {
  const { data: models = [], isLoading, refetch: refetchModels } = useModels()
  const { data: repoModels = [], isLoading: repoLoading, refetch: refetchRepo } = useRepositoryModels()
  const unload = useUnloadModel()
  const loadFromRepo = useLoadFromRepository()
  const { t } = useT()

  const [uploadOpen, setUploadOpen] = useState(false)
  const [expandedLoaded, setExpandedLoaded] = useState<string | null>(null)
  const [expandedRepo, setExpandedRepo] = useState<string | null>(null)
  const repoById = useMemo(() => new Map(repoModels.map((m) => [m.id, m])), [repoModels])

  return (
    <div>
      <PageHeader
        title={t('models.title')}
        subtitle={t('models.subtitle')}
        action={
          <button onClick={() => setUploadOpen(true)} className="btn-primary">
            {t('models.upload')}
          </button>
        }
      />

      {/* Loaded Models */}
      <div className="section">
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('models.col.backend')}</th>
                <th>{t('models.col.version')}</th>
                <th>{t('models.col.summary')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={5} />}
              {!isLoading && models.length === 0 && (
                <EmptyState message={t('models.empty')} />
              )}
              {models.map((m) => {
                const repo = repoById.get(m.id)
                const modelType = m.model_type ?? repo?.model_type
                const numClasses = m.num_classes ?? repo?.num_classes
                const confThresh = m.conf_thresh ?? repo?.conf_thresh
                const nmsThresh = m.nms_thresh ?? repo?.nms_thresh
                const batchSize = m.batch_size ?? repo?.batch_size
                const instanceCount = m.instance_count ?? repo?.instance_count

                return (
                  <Fragment key={m.id}>
                    <tr
                      className="cursor-pointer"
                      onClick={() => setExpandedLoaded(expandedLoaded === m.id ? null : m.id)}
                    >
                      <td className="font-mono text-[12px] text-ink-primary font-medium">
                        <span className="mr-2 text-ink-muted">{expandedLoaded === m.id ? '▼' : '▶'}</span>
                        {m.id}
                      </td>
                      <td><Tag>{m.backend || repo?.backend || '—'}</Tag></td>
                      <td><Tag>{m.version || repo?.version || '—'}</Tag></td>
                      <td className="font-mono text-[12px] text-ink-secondary">
                        {modelType ? `${modelType} · ` : ''}
                        {m.input_shape || '—'}
                        <span className="text-ink-muted">
                          {' · '}
                          {t('models.col.batch')} {batchSize}
                          {' · '}
                          {t('models.col.instances')} {instanceCount}
                        </span>
                      </td>
                      <td className="text-right pr-2" onClick={(e) => e.stopPropagation()}>
                        <DeleteButton
                          loading={unload.isPending}
                          onClick={() => unload.mutate(m.id)}
                        />
                      </td>
                    </tr>
                    {expandedLoaded === m.id && (
                      <tr>
                        <td colSpan={5} className="bg-bg-overlay px-6 py-3">
                          <div className="label mb-2">{t('models.section.details')}</div>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            <DetailItem label={t('models.field.model_type')} value={modelType} />
                            <DetailItem label={t('models.col.input_shape')} value={m.input_shape} />
                            <DetailItem label={t('models.field.batch_size')} value={batchSize} />
                            <DetailItem label={t('models.field.instances')} value={instanceCount} />
                            <DetailItem label={t('models.field.num_classes')} value={numClasses} />
                            <DetailItem label={t('models.field.conf_thresh')} value={confThresh} />
                            <DetailItem label={t('models.field.nms_thresh')} value={nmsThresh} />
                            <DetailItem label={t('models.field.device_id')} value={m.device_id} />
                            <DetailItem label={t('models.field.preferred_batch_sizes')} value={formatList(m.preferred_batch_sizes)} />
                            <DetailItem label={t('models.field.max_queue_delay_us')} value={m.max_queue_delay_us} />
                            <DetailItem label={t('models.field.class_names')} value={formatList(m.class_names)} />
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Repository */}
      <div className="section">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-ink-primary">{t('models.repo_title')}</h2>
            <p className="text-xs text-ink-muted mt-0.5">{t('models.repo_subtitle')}</p>
          </div>
        </div>
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('models.col.backend')}</th>
                <th>{t('models.col.version')}</th>
                <th>{t('common.state')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {repoLoading && <LoadingRows cols={5} />}
              {!repoLoading && repoModels.length === 0 && (
                <EmptyState message={t('models.repo_empty')} />
              )}
              {repoModels.map((m) => (
                <Fragment key={m.id}>
                  <tr
                    className="cursor-pointer"
                    onClick={() => setExpandedRepo(expandedRepo === m.id ? null : m.id)}
                  >
                    <td className="font-mono text-[12px] text-ink-primary font-medium">
                      <span className="mr-2 text-ink-muted">{expandedRepo === m.id ? '▼' : '▶'}</span>
                      {m.id}
                    </td>
                    <td><Tag>{m.backend}</Tag></td>
                    <td><Tag>{m.version}</Tag></td>
                    <td><StatusBadge loaded={m.loaded} /></td>
                    <td className="text-right pr-2" onClick={(e) => e.stopPropagation()}>
                      {!m.loaded && (
                        <button
                          onClick={() => loadFromRepo.mutate(m.id)}
                          disabled={loadFromRepo.isPending}
                          className="text-xs text-primary hover:underline disabled:opacity-50"
                        >
                          {loadFromRepo.isPending ? t('models.repo_loading') : t('models.repo_load')}
                        </button>
                      )}
                    </td>
                  </tr>
                  {expandedRepo === m.id && (
                    <tr>
                      <td colSpan={5} className="bg-bg-overlay px-6 py-3">
                        <div className="label mb-2">{t('models.section.details')}</div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          <DetailItem label={t('models.field.model_type')} value={m.model_type} />
                          <DetailItem label={t('models.field.batch_size')} value={m.batch_size} />
                          <DetailItem label={t('models.field.instances')} value={m.instance_count} />
                          <DetailItem label={t('models.field.num_classes')} value={m.num_classes} />
                          <DetailItem label={t('models.field.conf_thresh')} value={m.conf_thresh} />
                          <DetailItem label={t('models.field.nms_thresh')} value={m.nms_thresh} />
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onSuccess={() => {
          void refetchModels()
          void refetchRepo()
        }}
      />
    </div>
  )
}
