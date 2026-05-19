import { Fragment, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { usePipelines, useRemovePipeline } from '../hooks/queries'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'

export function PipelinesPage() {
  const { data: pipelines = [], isLoading } = usePipelines()
  const remove = useRemovePipeline()
  const { t } = useT()
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState<string | null>(null)

  return (
    <div>
      <PageHeader
        title={t('pipelines.title')}
        subtitle={t('pipelines.subtitle')}
        action={
          <button onClick={() => navigate('/pipelines/new')} className="btn-primary">
            {t('pipelines.add')}
          </button>
        }
      />

      <div className="section">
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('pipelines.col.nodes')}</th>
                <th>{t('pipelines.col.edges')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={4} />}
              {!isLoading && pipelines.length === 0 && (
                <EmptyState message={t('pipelines.empty')} />
              )}
              {pipelines.map((p) => (
                <Fragment key={p.id}>
                  <tr
                    className="cursor-pointer"
                    onClick={() => setExpanded(expanded === p.id ? null : p.id)}
                  >
                    <td className="font-mono text-[12px] text-ink-primary font-medium">
                      <span className="mr-2 text-ink-muted">{expanded === p.id ? '▼' : '▶'}</span>
                      {p.id}
                    </td>
                    <td className="font-mono text-[12px] text-ink-secondary">{p.nodes.length}</td>
                    <td className="font-mono text-[12px] text-ink-secondary">{p.edges.length}</td>
                    <td className="text-right pr-2 flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => navigate(`/pipelines/new?copyFrom=${encodeURIComponent(p.id)}`)}
                        className="btn-icon text-ink-muted hover:text-accent"
                        title={t('pipelines.copy')}
                      >
                        ⎘
                      </button>
                      <button
                        onClick={() => navigate(`/pipelines/${encodeURIComponent(p.id)}/edit`)}
                        className="btn-icon text-ink-muted hover:text-accent"
                        title={t('pipelines.edit')}
                      >
                        ✎
                      </button>
                      <DeleteButton
                        loading={remove.isPending}
                        onClick={() => remove.mutate(p.id)}
                      />
                    </td>
                  </tr>
                  {expanded === p.id && (
                    <tr>
                      <td colSpan={4} className="bg-bg-overlay px-6 py-3">
                        <div className="flex items-center justify-between mb-3">
                          <span className="label">{t('pipelines.section.nodes')}</span>
                          <Link
                            to={`/pipelines/${encodeURIComponent(p.id)}/edit`}
                            className="text-[11px] text-accent hover:underline"
                          >
                            {t('pipelines.editor.open')}
                          </Link>
                        </div>
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <div className="space-y-1">
                              {p.nodes.map((n) => (
                                <div key={n.id} className="text-[12px]">
                                  <div className="flex gap-2 items-center">
                                    <span className="font-mono text-accent w-28 truncate">{n.id}</span>
                                    <span className="text-ink-muted">→</span>
                                    <span className="font-mono text-ink-secondary">{n.type}</span>
                                  </div>
                                  {n.with && Object.keys(n.with).length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-0.5 pl-[calc(7rem+1.25rem)]">
                                      {Object.entries(n.with).map(([k, v]) => (
                                        <span
                                          key={k}
                                          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-bg-overlay border border-border text-[10px] font-mono text-ink-muted"
                                        >
                                          <span className="text-accent">{k}</span>
                                          <span>=</span>
                                          <span>{v}</span>
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                          {p.edges.length > 0 && (
                            <div>
                              <div className="label mb-2">{t('pipelines.section.edges')}</div>
                              <div className="space-y-1">
                                {p.edges.map((e, i) => (
                                  <div key={i} className="flex gap-2 items-center text-[12px]">
                                    <span className="font-mono text-ink-primary">{e.from}</span>
                                    <span className="text-ink-muted">→</span>
                                    <span className="font-mono text-ink-primary">{e.to}</span>
                                    <span className="text-ink-muted ml-2 text-[10px]">
                                      cap:{e.capacity} · {e.drop_policy}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
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
    </div>
  )
}
