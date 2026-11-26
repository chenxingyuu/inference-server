import { useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { usePipelines } from '../hooks/queries'
import { api, ApiError } from '../lib/api'
import { useT } from '../lib/i18n'
import type { ArchiveAnalysis } from '../types'
import { Field, Input, Select } from './ui/Field'
import { Modal } from './ui/Modal'

const ascendBatches = [4, 8, 16] as const
const yoloVersions = ['v5', 'v8', 'v11', 'v26']
const backendNames: Record<string, string> = { cpu: 'ONNX Runtime', trt: 'TensorRT', ascend: 'Ascend', mps: 'CoreML' }
const archiveExtensions = /\.(zip|tar\.gz|tgz)$/i
const acceptedExtensions = /\.(onnx|engine|om|zip|tar\.gz|tgz)$/i

function getFields(result: ArchiveAnalysis) {
  return {
    id: result.suggested_id ?? '',
    yolo_version: result.yolo_version,
    model_type: result.model_type,
    input_height: result.input_size ? String(result.input_size[0]) : '',
    input_width: result.input_size ? String(result.input_size[1]) : '',
    num_classes: String(result.num_classes),
    batch_size: String(result.batch_size),
    conf_thresh: String(result.conf_thresh),
    nms_thresh: String(result.nms_thresh),
    instance_count: String(result.instance_count),
    device_id: String(result.device_id),
    device_ids: result.device_ids?.join(', ') ?? '',
  }
}

type Fields = ReturnType<typeof getFields>
type UploadState = { analysis: ArchiveAnalysis; fields: Fields }

function describeError(error: unknown, t: ReturnType<typeof useT>['t']) {
  if (error instanceof ApiError) {
    if (error.status === 413) return t('models.upload_too_large')
    switch (error.code) {
      case 'already_exists': return t('models.upload_conflict')
      case 'publish_busy': return t('models.publish_busy')
      case 'engine_unavailable': return t('models.engine_unavailable')
      case 'publication_auth_required': return t('models.publication_auth_required')
      case 'unauthorized': return t('models.publication_unauthorized')
      case 'upload_expired': return t('models.upload_expired')
    }
  }
  return error instanceof Error && error.message ? error.message : t('models.upload_failed')
}

export function ModelUploadModal({ open, onClose, onPublished }: {
  open: boolean
  onClose: () => void
  onPublished: () => void
}) {
  const { t } = useT()
  const navigate = useNavigate()
  const { data: pipelines = [] } = usePipelines()
  const [pipeline, setPipeline] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [extraWeights, setExtraWeights] = useState<Partial<Record<number, File>>>({})
  const [upload, setUpload] = useState<UploadState | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [publishing, setPublishing] = useState(false)
  const [publishedID, setPublishedID] = useState('')
  const [dragging, setDragging] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)

  const handleClose = () => {
    if (publishing) return
    controller.current?.abort()
    controller.current = null
    setFile(null)
    setExtraWeights({})
    setUpload(null)
    setProgress(null)
    setPublishedID('')
    setPipeline('')
    setDragging(false)
    setCopied(false)
    setError('')
    onClose()
  }

  const chooseFile = (selected?: File) => {
    setError('')
    setUpload(null)
    setExtraWeights({})
    if (!selected || !acceptedExtensions.test(selected.name)) {
      setFile(null)
      if (selected) setError(t('models.file_formats'))
      return
    }
    setFile(selected)
  }

  const analyze = async (event: FormEvent) => {
    event.preventDefault()
    if (!file || controller.current) return
    setError('')
    setProgress(0)
    const body = new FormData()
    body.append(archiveExtensions.test(file.name) ? 'archive' : 'file', file)
    for (const [batch, weight] of Object.entries(extraWeights)) {
      if (weight) body.append(`batch_${batch}`, weight)
    }

    const request = new AbortController()
    controller.current = request
    try {
      const analysis = await api.models.analyze(body, setProgress, request.signal)
      if (controller.current === request) setUpload({ analysis, fields: getFields(analysis) })
    } catch (e) {
      if (controller.current === request) setError(describeError(e, t))
    } finally {
      if (controller.current === request) {
        controller.current = null
        setProgress(null)
      }
    }
  }

  const publish = async (event: FormEvent) => {
    event.preventDefault()
    if (!upload || publishing) return
    setError('')
    setPublishing(true)
    const { analysis, fields } = upload
    const original = getFields(analysis)
    const body = new URLSearchParams({ id: fields.id.trim(), token: analysis.token })
    for (const key of Object.keys(fields) as Array<keyof Fields>) {
      if (key !== 'id' && key !== 'input_height' && key !== 'input_width' && fields[key] !== original[key]) {
        body.set(key, fields[key])
      }
    }
    if (fields.input_height !== original.input_height || fields.input_width !== original.input_width) {
      body.set('input_height', fields.input_height)
      body.set('input_width', fields.input_width)
    }

    try {
      const result = await api.models.publish(body)
      setPublishedID(result.id)
      onPublished()
    } catch (e) {
      setError(describeError(e, t))
      if (e instanceof ApiError && e.status === 410) setUpload(null)
    } finally {
      setPublishing(false)
    }
  }

  const setField = (key: keyof Fields, value: string) => {
    setUpload((current) => current && { ...current, fields: { ...current.fields, [key]: value } })
  }
  const numberField = (key: keyof Fields, label: string, min: number, max?: number) => (
    <Field label={label}>
      <Input aria-label={label} type="number" min={min} max={max} step={max === 1 ? 'any' : 1}
        required value={upload?.fields[key] ?? ''} onChange={(e) => setField(key, e.target.value)} />
    </Field>
  )
  const errorMessage = error && (
    <p role="alert" className="rounded border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
      {error}
    </p>
  )

  return (
    <Modal open={open} onClose={handleClose} title={t('models.upload_modal_title')} width="max-w-xl">
      {publishedID ? (
        <div className="space-y-4">
          <div role="status" className="rounded border border-success/30 bg-success/10 p-4">
            <p className="font-semibold text-success">{t('models.upload_success')}</p>
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
              <code className="break-all text-sm text-ink-primary">{publishedID}</code>
              <button type="button" className="btn-ghost" onClick={async () => {
                try {
                  await navigator.clipboard.writeText(publishedID)
                  setCopied(true)
                } catch (e) {
                  setError(describeError(e, t))
                }
              }}>
                {t(copied ? 'models.copied_id' : 'models.copy_id')}
              </button>
            </div>
          </div>
          <p className="text-sm text-ink-secondary">{t('models.published_hint')}</p>
          {!!pipelines.length && (
            <Select aria-label={t('models.choose_pipeline')} value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
              <option value="">{t('models.choose_pipeline')}</option>
              {pipelines.map(({ id }) => <option key={id} value={id}>{id}</option>)}
            </Select>
          )}
          {errorMessage}
          <div className="flex justify-end gap-2">
            <button type="button" className="btn-primary" onClick={() => {
              const path = pipeline ? `/pipelines/${encodeURIComponent(pipeline)}/edit` : '/pipelines'
              handleClose()
              navigate(path)
            }}>{t('models.open_pipeline')}</button>
          </div>
        </div>
      ) : upload ? (
        <form onSubmit={publish} className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-bg-elevated p-3">
            <span className="break-all font-mono text-xs text-ink-secondary">{file?.name}</span>
            <span className="text-xs font-medium text-ink-primary">
              {backendNames[upload.analysis.backend] ?? upload.analysis.backend}
            </span>
          </div>
          <fieldset disabled={publishing} className="space-y-4">
            <Field label={t('common.id')}>
              <Input aria-label={t('common.id')} required pattern="[A-Za-z0-9_\-]+" value={upload.fields.id}
                onChange={(e) => setField('id', e.target.value)} />
            </Field>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Field label={t('models.field.model_type')}>
                <Select aria-label={t('models.field.model_type')} value={upload.fields.model_type}
                  onChange={(e) => setField('model_type', e.target.value)}>
                  <option value="detector">{t('models.detector')}</option>
                  <option value="classifier">{t('models.classifier')}</option>
                </Select>
              </Field>
              <Field label={t('models.field.yolo_version')}>
                <Select aria-label={t('models.field.yolo_version')} value={upload.fields.yolo_version}
                  onChange={(e) => setField('yolo_version', e.target.value)}>
                  {yoloVersions.map((version) => <option key={version} value={version}>{version}</option>)}
                </Select>
              </Field>
              {numberField('input_height', t('models.input_height'), 1)}
              {numberField('input_width', t('models.input_width'), 1)}
              {numberField('num_classes', t('models.field.num_classes'), 1)}
            </div>
            <details className="rounded border border-border px-3 py-2">
              <summary className="cursor-pointer text-sm text-ink-secondary">{t('models.advanced_settings')}</summary>
              <div className="mt-3 grid grid-cols-1 gap-3 pb-1 sm:grid-cols-2">
                {numberField('batch_size', t('models.field.batch_size'), 1)}
                {numberField('instance_count', t('models.field.instances'), 1)}
                {numberField('conf_thresh', t('models.field.conf_thresh'), 0, 1)}
                {numberField('nms_thresh', t('models.field.nms_thresh'), 0, 1)}
                {numberField('device_id', t('models.field.device_id'), 0)}
                <Field label={t('models.device_ids')}>
                  <Input aria-label={t('models.device_ids')} value={upload.fields.device_ids}
                    onChange={(e) => setField('device_ids', e.target.value)} />
                </Field>
              </div>
            </details>
          </fieldset>
          {errorMessage}
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" className="btn-ghost" disabled={publishing} onClick={() => chooseFile()}>
              {t('models.choose_again')}
            </button>
            <button type="submit" className="btn-primary" disabled={publishing}>
              {t(publishing ? 'models.checking_weights' : error ? 'models.retry_publish' : 'models.confirm_publish')}
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={analyze} className="space-y-4">
          <label className={`flex min-h-32 cursor-pointer flex-col items-center justify-center gap-2 rounded border border-dashed p-4 text-center focus-within:ring-2 focus-within:ring-primary ${dragging ? 'border-primary bg-primary/10' : 'border-border bg-bg-elevated'}`}
            onDragOver={(e) => { e.preventDefault(); if (progress === null) setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              if (progress === null) chooseFile(e.dataTransfer.files[0])
            }}>
            <input type="file" className="sr-only" accept=".onnx,.engine,.om,.zip,.tar.gz,.tgz"
              disabled={progress !== null} aria-label={t('models.choose_file')}
              onChange={(e) => { chooseFile(e.target.files?.[0]); e.target.value = '' }} />
            <span className="max-w-full break-all text-sm font-medium text-ink-primary">
              {file?.name ?? t('models.choose_file')}
            </span>
            <span className="text-xs text-ink-muted">{t('models.file_formats')}</span>
          </label>
          {file && /\.om$/i.test(file.name) && (
            <details key={file.name + file.lastModified} className="rounded border border-border px-3 py-2">
              <summary className="cursor-pointer text-sm text-ink-secondary">{t('models.advanced_settings')}</summary>
              <p className="mt-2 text-xs text-ink-muted">{t('models.ascend_batch_hint')}</p>
              <div className="mt-2 space-y-2">
                {ascendBatches.map((batch) => (
                  <Field key={batch} label={`Batch ${batch} (.om)`}>
                    <input type="file" accept=".om" aria-label={`Batch ${batch} (.om)`} disabled={progress !== null}
                      className="block w-full text-xs text-ink-secondary file:mr-2 file:rounded file:border-0 file:bg-bg-elevated file:px-2 file:py-1"
                      onChange={(e) => {
                        const selected = e.target.files?.[0]
                        setExtraWeights((current) => ({ ...current, [batch]: selected }))
                      }} />
                  </Field>
                ))}
              </div>
            </details>
          )}
          {progress !== null && (
            <div className="space-y-1" role="status">
              <p className="text-xs text-ink-secondary">{t(progress === 100 ? 'models.checking_files' : 'models.uploading')}</p>
              <progress className="w-full accent-primary" max={100} value={progress} aria-label={t('models.uploading')} />
            </div>
          )}
          {errorMessage}
          <div className="flex justify-end gap-2">
            <button type="button" className="btn-ghost" onClick={handleClose}>{t('common.cancel')}</button>
            <button type="submit" className="btn-primary" disabled={!file || progress !== null}>
              {t('models.inspect_files')}
            </button>
          </div>
        </form>
      )}
    </Modal>
  )
}
