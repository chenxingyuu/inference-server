import { useCallback, useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { TaskStatusBadge } from '../components/ui/StatusBadge'
import { PageHeader } from '../components/layout/Layout'
import { useModelDetail } from '../hooks/queries'
import { ApiError, api } from '../lib/api'
import { useT } from '../lib/i18n'
import type { ModelDetail, ModelTestResult } from '../types'

const maxImageBytes = 8 * 1024 * 1024
const minImageZoom = 50
const maxImageZoom = 300
const imageZoomStep = 25

type SelectedImage = { file: File; url: string }
type PreviewZoomAnchor = { x: number; y: number; clientX: number; clientY: number }
type PreviewPointer = { clientX?: number; clientY?: number }
type SafariGestureEvent = Event & PreviewPointer & { scale: number }

function DetailItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="break-words font-mono text-xs text-ink-secondary">{children ?? '—'}</dd>
    </div>
  )
}

function DetailSection({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="card space-y-4 p-4 md:p-5">
      <h2 className="text-sm font-semibold text-ink-primary">{title}</h2>
      {children}
    </section>
  )
}

function ModelInformation({ detail }: { detail: ModelDetail }) {
  const { t } = useT()
  const { model, registered, repository, repository_error } = detail
  const source = repository || !registered ? t('models.source.repository')
    : repository_error ? undefined : t('models.source.outside_repository')

  return (
    <DetailSection title={t('models.detail.information')}>
      <dl className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        <DetailItem label={t('common.state')}>
          {t(registered ? 'models.status.registered' : 'models.status.unregistered')}
        </DetailItem>
        <DetailItem label={t('models.field.source')}>{source}</DetailItem>
        <DetailItem label={t('models.col.backend')}>{model.backend}</DetailItem>
        <DetailItem label={t('models.field.yolo_version')}>{model.version}</DetailItem>
        <DetailItem label={t('models.field.model_type')}>
          {model.model_type && t(model.model_type === 'classifier' ? 'models.classifier' : 'models.detector')}
        </DetailItem>
        <DetailItem label={t('models.col.input_shape')}>{model.input_shape}</DetailItem>
        <DetailItem label={t('models.field.batch_size')}>{model.batch_size}</DetailItem>
        <DetailItem label={t('models.field.instances')}>{model.instance_count}</DetailItem>
        <DetailItem label={t('models.field.device_id')}>{model.device_id}</DetailItem>
        <DetailItem label={t('models.field.num_classes')}>{model.num_classes}</DetailItem>
        <DetailItem label={t('models.field.conf_thresh')}>{model.conf_thresh?.toFixed(3)}</DetailItem>
        {model.model_type !== 'classifier' && (
          <DetailItem label={t('models.field.nms_thresh')}>{model.nms_thresh?.toFixed(3)}</DetailItem>
        )}
        <DetailItem label={t('models.field.preferred_batch_sizes')}>{model.preferred_batch_sizes?.join(', ') || undefined}</DetailItem>
        <DetailItem label={t('models.field.max_queue_delay_us')}>{model.max_queue_delay_us}</DetailItem>
        <DetailItem label={t('models.field.class_names')}>{model.class_names?.join(', ') || undefined}</DetailItem>
      </dl>
    </DetailSection>
  )
}

function ModelReferences({ detail }: { detail: ModelDetail }) {
  const { t } = useT()
  const { pipelines, cascade_parents, cascade_children } = detail

  return (
    <DetailSection title={t('models.detail.references')}>
      {pipelines.length === 0 && cascade_parents.length === 0 && cascade_children.length === 0 && (
        <p className="text-xs text-ink-muted">{t('models.detail.no_references')}</p>
      )}
      {pipelines.map(pipeline => (
        <div className="space-y-1 border-b border-border pb-3" key={pipeline.id}>
          <Link className="font-mono text-xs text-accent hover:underline" to={`/pipelines/${encodeURIComponent(pipeline.id)}/edit`}>
            {pipeline.id}
          </Link>
          {pipeline.nodes.length > 0 && (
            <p className="break-words text-xs text-ink-muted">{t('models.detail.nodes', { ids: pipeline.nodes.join(', ') })}</p>
          )}
          {pipeline.cascade_nodes.length > 0 && (
            <p className="break-words text-xs text-ink-muted">
              {t('models.detail.cascade_nodes', { ids: pipeline.cascade_nodes.join(', ') })}
            </p>
          )}
          <ul className="flex flex-wrap gap-x-3 gap-y-1">
            {pipeline.tasks.map(task => (
              <li className="text-xs text-ink-secondary" key={task.id}>
                {task.id} · <TaskStatusBadge state={task.state} />
              </li>
            ))}
          </ul>
        </div>
      ))}
      {cascade_parents.length > 0 && (
        <div>
          <h3 className="label">{t('models.detail.cascade_parents')}</h3>
          <ul className="flex flex-wrap gap-3">
            {cascade_parents.map(id => (
              <li key={id}><Link className="font-mono text-xs text-accent hover:underline" to={`/models/${encodeURIComponent(id)}`}>{id}</Link></li>
            ))}
          </ul>
        </div>
      )}
      {cascade_children.length > 0 && (
        <div>
          <h3 className="label">{t('models.detail.cascade_children')}</h3>
          <ul className="space-y-1">
            {cascade_children.map((child, index) => (
              <li key={`${child.model_id}-${child.attribute_key}-${index}`} className="text-xs text-ink-secondary">
                <Link className="font-mono text-accent hover:underline" to={`/models/${encodeURIComponent(child.model_id)}`}>{child.model_id}</Link>
                {child.attribute_key && <span className="ml-2">{child.attribute_key}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </DetailSection>
  )
}

function ModelRepository({ detail }: { detail: ModelDetail }) {
  const { t } = useT()
  return (
    <DetailSection title={t('models.source.repository')}>
      {detail.repository ? (
        <dl className="grid gap-4">
          <DetailItem label={t('models.detail.active_version')}>{detail.repository.version}</DetailItem>
          <DetailItem label={t('models.detail.weight_files')}>
            <ul className="space-y-1">
              {detail.repository.weight_paths.map(path => <li key={path}>{path}</li>)}
            </ul>
          </DetailItem>
        </dl>
      ) : (
        <p className={detail.repository_error ? 'text-xs text-danger' : 'text-xs text-ink-muted'}>
          {t(detail.repository_error ? 'models.detail.repository_error' : 'models.source.outside_repository')}
        </p>
      )}
    </DetailSection>
  )
}

function confidencePercent(score: number) {
  return `${(score * 100).toFixed(1)}%`
}

function ModelTestResultImage({ result }: { result: ModelTestResult }) {
  const { t } = useT()
  const fontSize = result.width / 48

  return (
    <svg className="absolute inset-0 block h-full w-full" viewBox={`0 0 ${result.width} ${result.height}`} role="img" aria-label={t('models.test.preview')}>
      <image href={`data:image/jpeg;base64,${result.preview_jpeg}`} width={result.width} height={result.height} preserveAspectRatio="none" />
      {result.model_type === 'detector' && result.results.map(({ bbox, class_id, class_name, confidence }, index) => bbox && (
        <g key={index}>
          <rect x={bbox.x1} y={bbox.y1} width={bbox.x2 - bbox.x1} height={bbox.y2 - bbox.y1}
            className="fill-none stroke-green-400" strokeWidth={2} vectorEffect="non-scaling-stroke" />
          <text x={bbox.x1} y={Math.max(fontSize, bbox.y1 - fontSize / 4)} fontSize={fontSize}
            stroke="#121714" strokeWidth={fontSize / 5} paintOrder="stroke" fill="#dcfce7">
            {class_name || class_id} {confidencePercent(confidence)}
          </text>
        </g>
      ))}
    </svg>
  )
}

function ModelTestResults({ result }: { result: ModelTestResult }) {
  const { t } = useT()

  return (
    <div className="min-w-0 space-y-3 xl:pt-11">
      <dl className="grid grid-cols-2 gap-3">
        <DetailItem label={t('models.col.backend')}>{result.backend}</DetailItem>
        <DetailItem label={t('models.field.conf_thresh')}>{result.conf_thresh.toFixed(3)}</DetailItem>
        {result.nms_thresh !== null && <DetailItem label={t('models.field.nms_thresh')}>{result.nms_thresh.toFixed(3)}</DetailItem>}
      </dl>
      {result.results.length ? (
        <ul className="max-h-80 divide-y divide-border overflow-auto rounded border border-border">
          {result.results.map(({ class_id, class_name, confidence }, index) => (
            <li className="flex justify-between gap-3 px-3 py-2 text-xs text-ink-secondary" key={index}>
              <span className="break-all">{class_name || class_id}</span>
              <span className="font-mono text-ink-primary">{confidencePercent(confidence)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p role="status" className="text-xs text-ink-muted">
          {t(result.model_type === 'detector' ? 'models.test.no_detections' : 'models.test.no_classes')}
        </p>
      )}
    </div>
  )
}

function ModelTestImagePreview({ image, result }: { image: SelectedImage | null; result?: ModelTestResult }) {
  const { t } = useT()
  const [previewFailed, setPreviewFailed] = useState(false)
  const [originalAspect, setOriginalAspect] = useState<number>()
  const [zoom, setZoom] = useState(100)
  const currentZoom = useRef(zoom)
  const viewportRef = useRef<HTMLDivElement>(null)
  const imageSurfaceRef = useRef<HTMLDivElement>(null)
  const zoomAnchor = useRef<PreviewZoomAnchor | null>(null)
  const title = t(result ? 'models.test.preview' : 'models.test.original_preview')
  const canZoom = !!result || (!!image && !previewFailed)
  const imageAspect = result ? result.width / result.height : originalAspect

  useLayoutEffect(() => {
    const viewport = viewportRef.current
    const anchor = zoomAnchor.current
    const bounds = imageSurfaceRef.current?.getBoundingClientRect()
    zoomAnchor.current = null
    if (!viewport || !anchor || !bounds) return

    viewport.scrollLeft += bounds.left + anchor.x * bounds.width - anchor.clientX
    viewport.scrollTop += bounds.top + anchor.y * bounds.height - anchor.clientY
  }, [zoom])

  const changeZoom = useCallback((requestedZoom: number, pointer?: PreviewPointer) => {
    const nextZoom = Math.min(maxImageZoom, Math.max(minImageZoom, requestedZoom))
    if (!Number.isFinite(nextZoom) || nextZoom === currentZoom.current) return

    const viewport = viewportRef.current
    const surface = imageSurfaceRef.current?.getBoundingClientRect()
    if (!viewport || !surface?.width || !surface.height) return

    const bounds = viewport.getBoundingClientRect()
    const clientX = pointer?.clientX ?? bounds.left + viewport.clientLeft + viewport.clientWidth / 2
    const clientY = pointer?.clientY ?? bounds.top + viewport.clientTop + viewport.clientHeight / 2
    zoomAnchor.current = {
      x: (clientX - surface.left) / surface.width,
      y: (clientY - surface.top) / surface.height,
      clientX,
      clientY,
    }
    currentZoom.current = nextZoom
    setZoom(nextZoom)
  }, [])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport || !canZoom) return

    let safariStartZoom: number | null = null

    function handleWheel(event: WheelEvent) {
      if (!event.ctrlKey) return
      event.preventDefault()
      if (safariStartZoom !== null) return

      const units = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : 1
      changeZoom(currentZoom.current * Math.exp(-event.deltaY * units * 0.01), event)
    }

    // Safari sends gesture events for trackpad pinches instead of Ctrl-wheel.
    function handleGestureStart(event: Event) {
      event.preventDefault()
      safariStartZoom = currentZoom.current
    }

    function handleGestureChange(event: Event) {
      if (safariStartZoom === null) return
      event.preventDefault()
      const gesture = event as SafariGestureEvent
      changeZoom(safariStartZoom * gesture.scale, gesture)
    }

    function handleGestureEnd(event: Event) {
      if (safariStartZoom === null) return
      event.preventDefault()
      safariStartZoom = null
    }

    viewport.addEventListener('wheel', handleWheel, { passive: false })
    viewport.addEventListener('gesturestart', handleGestureStart, { passive: false })
    viewport.addEventListener('gesturechange', handleGestureChange, { passive: false })
    viewport.addEventListener('gestureend', handleGestureEnd, { passive: false })
    return () => {
      viewport.removeEventListener('wheel', handleWheel)
      viewport.removeEventListener('gesturestart', handleGestureStart)
      viewport.removeEventListener('gesturechange', handleGestureChange)
      viewport.removeEventListener('gestureend', handleGestureEnd)
    }
  }, [canZoom, changeZoom])

  function handlePreviewKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!canZoom || event.altKey || event.ctrlKey || event.metaKey) return

    switch (event.key) {
      case '+':
      case '=':
        changeZoom(currentZoom.current + imageZoomStep)
        break
      case '-':
        changeZoom(currentZoom.current - imageZoomStep)
        break
      case '0':
        changeZoom(100)
        break
      default:
        return
    }
    event.preventDefault()
  }

  return (
    <div className="min-w-0 space-y-2">
      <div className="flex min-h-9 flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-medium text-ink-secondary">{title}</h3>
        {canZoom && <span className="font-mono text-xs tabular-nums text-ink-secondary">{Math.round(zoom)}%</span>}
      </div>
      <div ref={viewportRef}
        className={`h-80 overflow-auto rounded border border-border bg-bg-base [container-type:size] [overflow-anchor:none] sm:h-96 ${zoom > 100 ? 'overscroll-contain' : ''}`}
        role="region" aria-label={title} aria-describedby={canZoom ? 'model-test-gesture-help' : undefined}
        tabIndex={canZoom ? 0 : undefined} onKeyDown={handlePreviewKeyDown}
        onDoubleClick={canZoom ? () => changeZoom(100) : undefined}>
        {canZoom ? (
          <div className="grid h-max min-h-full w-max min-w-full place-items-center">
            <div ref={imageSurfaceRef} className="relative" style={imageAspect
              ? { width: `min(${zoom}cqw, ${zoom * imageAspect}cqh)`, aspectRatio: imageAspect }
              : { width: '100cqw', height: '100cqh' }}>
              {result ? <ModelTestResultImage result={result} /> : image && (
                <img className="absolute inset-0 h-full w-full object-contain" src={image.url} alt={image.file.name} draggable={false}
                  onLoad={event => {
                    const { naturalWidth, naturalHeight } = event.currentTarget
                    setOriginalAspect(naturalWidth / naturalHeight)
                  }}
                  onError={() => setPreviewFailed(true)} />
              )}
            </div>
          </div>
        ) : (
          <div className="grid h-full place-items-center px-4 text-center text-xs text-ink-secondary">
            <p>{t(previewFailed ? 'models.test.preview_unavailable' : 'models.test.preview_empty')}</p>
          </div>
        )}
      </div>
      {canZoom && <p id="model-test-gesture-help" className="text-xs text-ink-secondary">{t('models.test.gesture_help')}</p>}
    </div>
  )
}

function ModelImageTest({ detail }: { detail: ModelDetail }) {
  const { t } = useT()
  const [image, setImage] = useState<SelectedImage | null>(null)
  const [invalidFile, setInvalidFile] = useState(false)
  const test = useMutation({ mutationFn: (file: File) => api.models.test(detail.model.id, file) })

  useEffect(() => {
    if (!image) return
    return () => URL.revokeObjectURL(image.url)
  }, [image])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (detail.registered && image && !test.isPending) test.mutate(image.file)
  }

  const errorCode = test.error instanceof ApiError ? test.error.code : undefined
  const errorMessage = errorCode === 'model_test_timeout' ? t('models.test.timeout')
    : errorCode === 'model_test_busy' ? t('models.test.busy')
    : errorCode === 'unauthorized' ? t('models.test.unauthorized')
    : errorCode === 'model_test_auth_required' ? t('models.test.auth_required')
    : errorCode === 'model_not_registered' ? t('models.test.register_first')
    : errorCode === 'model_test_unsupported_model' ? t('models.test.unsupported_model')
    : errorCode === 'invalid_image' || errorCode === 'image_too_large' ? t('models.test.image_limit')
    : t('models.test.failed')

  return (
    <DetailSection id="model-image-test" title={t('models.test.title')}>
      {detail.cascade_children.length > 0 && <p className="text-xs text-ink-muted">{t('models.test.cascade_scope')}</p>}
      {!detail.registered ? (
        <p className="text-xs text-ink-muted">{t('models.test.register_first')}</p>
      ) : (
        <form className="flex flex-wrap items-end gap-3" onSubmit={submit}>
          <label className="min-w-0 space-y-1">
            <span className="label">{t('models.test.image')}</span>
            <input type="file" accept="image/jpeg,image/png" disabled={test.isPending}
              className="block max-w-full text-xs text-ink-secondary file:mr-3 file:rounded file:border-0 file:bg-bg-elevated file:px-3 file:py-1.5 file:text-ink-primary"
              onChange={event => {
                const selected = event.target.files?.[0] ?? null
                const invalid = !!selected && (selected.size < 1 || selected.size > maxImageBytes)
                setImage(selected && !invalid ? { file: selected, url: URL.createObjectURL(selected) } : null)
                setInvalidFile(invalid)
                test.reset()
              }} />
          </label>
          <button type="submit" className="btn-primary" disabled={!image || test.isPending}>
            {t(test.isPending ? 'models.test.running' : 'models.test.run')}
          </button>
        </form>
      )}
      {invalidFile && <p role="alert" className="text-xs text-danger">{t('models.test.image_limit')}</p>}
      {test.isError && <p role="alert" className="text-xs text-danger">{errorMessage}</p>}
      {detail.registered && (
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <ModelTestImagePreview key={image?.url} image={image} result={test.data} />
          {test.data && <ModelTestResults result={test.data} />}
        </div>
      )}
    </DetailSection>
  )
}

function ModelDetailTabs({ detail }: { detail: ModelDetail }) {
  const { t } = useT()
  const [searchParams, setSearchParams] = useSearchParams()
  const tabButtons = useRef<Array<HTMLButtonElement | null>>([])
  const tabs = [
    {
      id: 'card',
      label: t('models.detail.tabs.card'),
      content: (
        <>
          <ModelInformation detail={detail} />
          <ModelReferences detail={detail} />
        </>
      ),
    },
    { id: 'files', label: t('models.detail.tabs.files'), content: <ModelRepository detail={detail} /> },
    { id: 'test', label: t('models.detail.tabs.test'), content: <ModelImageTest detail={detail} /> },
  ] as const

  type TabId = typeof tabs[number]['id']

  const requestedTab = searchParams.get('tab')
  const activeTab = tabs.find(tab => tab.id === requestedTab)?.id ?? 'card'

  function selectTab(id: TabId) {
    if (id === activeTab) return

    const nextParams = new URLSearchParams(searchParams)
    if (id === 'card') nextParams.delete('tab')
    else nextParams.set('tab', id)
    setSearchParams(nextParams)
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (event.altKey || event.ctrlKey || event.metaKey) return

    let nextIndex: number
    switch (event.key) {
      case 'ArrowLeft':
        nextIndex = (index + tabs.length - 1) % tabs.length
        break
      case 'ArrowRight':
        nextIndex = (index + 1) % tabs.length
        break
      case 'Home':
        nextIndex = 0
        break
      case 'End':
        nextIndex = tabs.length - 1
        break
      default:
        return
    }
    event.preventDefault()
    selectTab(tabs[nextIndex].id)
    tabButtons.current[nextIndex]?.focus()
  }

  return (
    <>
      <div role="tablist" aria-label={t('models.detail.title')} className="flex gap-3 overflow-x-auto border-b border-border">
        {tabs.map(({ id, label }, index) => {
          const selected = id === activeTab
          return (
            <button
              key={id}
              type="button"
              role="tab"
              id={`model-tab-${id}`}
              aria-controls={`model-panel-${id}`}
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              ref={button => { tabButtons.current[index] = button }}
              className={`shrink-0 border-b-2 px-3 py-2.5 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent ${
                selected ? 'border-accent text-accent' : 'border-transparent text-ink-muted hover:text-ink-primary'
              }`}
              onClick={() => selectTab(id)}
              onKeyDown={event => handleTabKeyDown(event, index)}
            >
              {label}
            </button>
          )
        })}
      </div>
      {tabs.map(({ id, content }) => (
        <div
          key={id}
          id={`model-panel-${id}`}
          role="tabpanel"
          aria-labelledby={`model-tab-${id}`}
          tabIndex={0}
          hidden={id !== activeTab}
          className="space-y-4 rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {content}
        </div>
      ))}
    </>
  )
}

function ModelDetailContent({ id }: { id: string }) {
  const { t } = useT()
  const { data, error, isPending, isError, refetch, isFetching } = useModelDetail(id)
  const notFound = error instanceof ApiError && error.status === 404

  return (
    <div>
      <PageHeader title={id} subtitle={t('models.detail.title')} action={
        <Link to="/models" className="btn-ghost">{t('models.detail.back')}</Link>
      } />
      <div className="section space-y-4">
        {isPending && <p role="status" className="text-sm text-ink-muted">{t('models.loading')}</p>}
        {isError && (
          <div role="alert" className="flex items-center gap-3 text-sm text-danger">
            {t(notFound ? 'models.detail.not_found' : 'models.detail.load_error')}
            {!notFound && (
              <button type="button" onClick={() => void refetch()} disabled={isFetching} className="btn-ghost">
                {t('models.detail.retry')}
              </button>
            )}
          </div>
        )}
        {data && !notFound && <ModelDetailTabs detail={data} />}
      </div>
    </div>
  )
}

export function ModelDetailPage() {
  const { id = '' } = useParams<{ id: string }>()
  return <ModelDetailContent key={id} id={id} />
}
