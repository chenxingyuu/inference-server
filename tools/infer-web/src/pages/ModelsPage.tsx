import { Fragment, useMemo, useState, type ReactNode } from 'react'
import { useModels, useUnloadModel, useRepositoryModels, useLoadFromRepository } from '../hooks/queries'
import { ModelUploadModal } from '../components/ModelUploadModal'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'

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

function formatThreshold(value?: number) {
  return value === undefined ? undefined : Number(value.toFixed(4)).toString()
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
                            <DetailItem label={t('models.field.conf_thresh')} value={formatThreshold(confThresh)} />
                            <DetailItem label={t('models.field.nms_thresh')} value={formatThreshold(nmsThresh)} />
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
                          <DetailItem label={t('models.field.conf_thresh')} value={formatThreshold(m.conf_thresh)} />
                          <DetailItem label={t('models.field.nms_thresh')} value={formatThreshold(m.nms_thresh)} />
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

      <ModelUploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onPublished={() => {
          void refetchModels()
          void refetchRepo()
        }}
      />
    </div>
  )
}
