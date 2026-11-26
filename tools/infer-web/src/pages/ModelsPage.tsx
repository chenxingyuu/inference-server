import { useState } from 'react'
import { Link } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useModels, useUnloadModel, useRepositoryModels, useLoadFromRepository } from '../hooks/queries'
import { ModelUploadModal } from '../components/ModelUploadModal'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import { api } from '../lib/api'
import type { ModelInfo, RepositoryModel } from '../types'

const backendLabels = new Map([
  ['cpu', 'CPU'],
  ['mps', 'MPS'],
  ['tensorrt', 'TensorRT'],
  ['trt', 'TensorRT'],
  ['cuda', 'TensorRT'],
  ['ascend', 'Ascend'],
])

const yoloLabels = new Map([
  ['v5', 'YOLOv5'],
  ['v8', 'YOLOv8'],
  ['v11', 'YOLO11'],
  ['v26', 'YOLO26'],
])

function Tag({ children }: { children: string }) {
  return (
    <span className="inline-block font-mono text-[10px] px-1.5 py-0.5 bg-bg-elevated border border-border rounded text-ink-secondary">
      {children}
    </span>
  )
}

function StatusBadge({ registered }: { registered: boolean | undefined }) {
  const { t } = useT()
  const label = registered === undefined ? 'models.status.unknown'
    : registered ? 'models.status.registered' : 'models.status.unregistered'

  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap text-[11px] font-medium ${registered ? 'text-success' : 'text-ink-muted'}`}>
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
      {t(label)}
    </span>
  )
}

export function ModelsPage() {
  const { data: registeredModels, isLoading: modelsLoading, isError: modelsError, refetch: refetchModels } = useModels()
  const { data: repositoryModels, isLoading: repoLoading, isError: repoError, refetch: refetchRepo } = useRepositoryModels()
  const unregister = useUnloadModel()
  const register = useLoadFromRepository()
  const { t } = useT()

  const [uploadOpen, setUploadOpen] = useState(false)
  const [exportingID, setExportingID] = useState<string | null>(null)
  const modelsByID = new Map<string, { registered?: ModelInfo; repository?: RepositoryModel }>()
  for (const model of registeredModels ?? []) {
    modelsByID.set(model.id, { registered: model })
  }
  for (const model of repositoryModels ?? []) {
    modelsByID.set(model.id, { ...modelsByID.get(model.id), repository: model })
  }
  const rows = Array.from(modelsByID).sort(([firstID], [secondID]) => firstID.localeCompare(secondID))

  const isLoading = modelsLoading || repoLoading
  const hasListError = modelsError || repoError
  const registrationPending = register.isPending || unregister.isPending

  async function exportModel(id: string) {
    setExportingID(id)
    try {
      const { filename, archive } = await api.models.exportFromRepository(id)
      const url = URL.createObjectURL(archive)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.append(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      toast.error(error instanceof Error && error.message ? error.message : t('models.repo_export_failed'))
    } finally {
      setExportingID(null)
    }
  }

  return (
    <div>
      <PageHeader
        title={t('models.title')}
        action={
          <button onClick={() => setUploadOpen(true)} className="btn-primary">
            {t('models.upload')}
          </button>
        }
      />

      <div className="section">
        {hasListError && (
          <p className="mb-3 text-xs text-danger" role="alert">{t('models.list_error')}</p>
        )}
        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('models.field.id')}</th>
                <th>{t('models.field.model_type')}</th>
                <th>{t('models.col.inference_backend')}</th>
                <th>{t('models.col.registration_state')}</th>
                <th className="text-right">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={5} />}
              {!isLoading && !hasListError && rows.length === 0 && (
                <EmptyState message={t('models.empty')} />
              )}
              {!isLoading && rows.map(([id, { registered, repository }]) => {
                const isRegistered = registeredModels === undefined ? undefined : !!registered
                const modelType = registered?.model_type ?? repository?.model_type
                const typeLabel = modelType === 'detector' ? t('models.detector')
                  : modelType === 'classifier' ? t('models.classifier') : modelType || '—'
                const version = registered?.version || repository?.version
                const yoloLabel = modelType === 'detector' && version ? yoloLabels.get(version) : undefined
                const backend = registered?.backend || repository?.backend
                const backendLabel = backend ? backendLabels.get(backend) ?? backend : '—'

                return (
                  <tr key={id}>
                    <td className="font-mono text-[12px] font-medium">
                      <Link to={`/models/${encodeURIComponent(id)}`} className="text-accent hover:underline">
                        {id}
                      </Link>
                    </td>
                    <td>
                      <div className="whitespace-nowrap text-xs text-ink-secondary">{typeLabel}</div>
                      {yoloLabel && <div className="mt-0.5 font-mono text-[11px] text-ink-muted">{yoloLabel}</div>}
                    </td>
                    <td><Tag>{backendLabel}</Tag></td>
                    <td><StatusBadge registered={isRegistered} /></td>
                    <td className="w-px">
                      <div className="flex items-center justify-end gap-3 whitespace-nowrap">
                        {isRegistered && (
                          <DeleteButton
                            label={t('models.unregister')}
                            loading={registrationPending}
                            onClick={() => unregister.mutate(id)}
                          />
                        )}
                        {isRegistered === false && repository && (
                          <button
                            type="button"
                            onClick={() => register.mutate(id)}
                            disabled={registrationPending}
                            className="text-xs text-accent hover:underline disabled:opacity-50"
                          >
                            {register.isPending && register.variables === id ? t('models.repo_loading') : t('models.repo_load')}
                          </button>
                        )}
                        {repository && (
                          <button
                            type="button"
                            onClick={() => void exportModel(id)}
                            disabled={exportingID !== null}
                            aria-busy={exportingID === id}
                            className="btn-ghost px-2 py-1 text-xs enabled:hover:bg-bg-elevated focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent aria-busy:disabled:cursor-wait aria-busy:disabled:opacity-100 disabled:active:scale-100"
                          >
                            {exportingID === id ? (
                              <svg aria-hidden="true" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="h-3.5 w-3.5 shrink-0 motion-safe:animate-spin">
                                <circle cx="8" cy="8" r="5.5" opacity=".25" />
                                <path d="M8 2.5a5.5 5.5 0 0 1 5.5 5.5" />
                              </svg>
                            ) : (
                              <svg aria-hidden="true" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5 shrink-0">
                                <path d="M8 2v8m-3-3 3 3 3-3M2.5 11v2.5h11V11" />
                              </svg>
                            )}
                            <span className="grid text-center">
                              <span className={exportingID === id ? 'invisible col-start-1 row-start-1' : 'col-start-1 row-start-1'}>
                                {t('models.repo_export')}
                              </span>
                              {exportingID === id && (
                                <span className="col-start-1 row-start-1">{t('models.repo_exporting')}</span>
                              )}
                            </span>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
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
