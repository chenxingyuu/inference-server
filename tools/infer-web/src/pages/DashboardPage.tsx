import { useHealth, useSources, useModels, usePipelines, useTasks } from '../hooks/queries'
import { serverUrl } from '../lib/api'
import { useT } from '../lib/i18n'
import { PageHeader } from '../components/layout/Layout'
import type { SourceState, TaskState } from '../types'

function StatCard({
  label, value, sub, color = 'text-ink-primary',
}: {
  label: string
  value: number | string
  sub?: string
  color?: string
}) {
  return (
    <div className="card p-5">
      <div className="text-[10.5px] font-semibold uppercase tracking-widest text-ink-muted mb-3">{label}</div>
      <div className={`text-3xl font-mono font-medium ${color}`}>{value}</div>
      {sub && <div className="text-[11px] text-ink-muted mt-1">{sub}</div>}
    </div>
  )
}

const sourceColors: Record<SourceState, string> = {
  STREAMING: 'text-success',
  CONNECTING: 'text-accent',
  RECONNECTING: 'text-warning',
  DEGRADED: 'text-warning',
  STOPPED: 'text-ink-muted',
}

const taskColors: Record<TaskState, string> = {
  running: 'text-success',
  stopped: 'text-ink-muted',
}

export function DashboardPage() {
  const { data: health } = useHealth()
  const { data: sources = [] } = useSources()
  const { data: models = [] } = useModels()
  const { data: pipelines = [] } = usePipelines()
  const { data: tasks = [] } = useTasks()
  const { t } = useT()

  const streaming = sources.filter((s) => s.state === 'STREAMING').length
  const degraded = sources.filter((s) => s.state === 'DEGRADED' || s.state === 'RECONNECTING').length
  const running = tasks.filter((t) => t.state === 'running').length

  const sourceColor =
    sources.length === 0 ? 'text-ink-muted'
    : degraded > 0 ? 'text-warning'
    : streaming === sources.length ? 'text-success'
    : 'text-accent'

  return (
    <div>
      <PageHeader
        title={t('dashboard.title')}
        subtitle={t('dashboard.subtitle')}
      />

      {/* Health banner */}
      <div className="px-6 pt-5">
        {(() => {
          const s = health?.status
          const cfg =
            s === 'ok'          ? { cls: 'bg-success/8 border-success/20 text-success',  dot: 'bg-success led-pulse', msg: t('health.ok') } :
            s === 'engine_down' ? { cls: 'bg-warning/8 border-warning/20 text-warning',  dot: 'bg-warning',          msg: t('health.engine_down') } :
            s === 'unreachable' ? { cls: 'bg-danger/8  border-danger/25  text-danger',   dot: 'bg-danger',           msg: t('health.unreachable', { url: serverUrl.get() }) } :
                                  { cls: 'bg-bg-elevated border-border   text-ink-muted', dot: 'bg-ink-muted',        msg: t('health.checking') }
          return (
            <div className={`flex items-center gap-2.5 px-4 py-2.5 rounded border text-sm font-medium ${cfg.cls}`}>
              <span className={`w-2 h-2 rounded-full flex-none ${cfg.dot}`} />
              {cfg.msg}
            </div>
          )
        })()}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 p-6">
        <StatCard
          label={t('dashboard.stat.sources')}
          value={sources.length}
          sub={sources.length
            ? t('dashboard.sources.sub', { streaming, degraded })
            : t('dashboard.sources.empty')}
          color={sourceColor}
        />
        <StatCard
          label={t('dashboard.stat.models')}
          value={models.length}
          sub={models.length
            ? t('dashboard.models.sub', { backends: models.map((m) => m.backend).join(', ') })
            : t('dashboard.models.empty')}
        />
        <StatCard
          label={t('dashboard.stat.pipelines')}
          value={pipelines.length}
          sub={pipelines.length
            ? t('dashboard.pipelines.sub', { nodes: pipelines.reduce((s, p) => s + p.nodes.length, 0) })
            : t('dashboard.pipelines.empty')}
        />
        <StatCard
          label={t('dashboard.stat.tasks')}
          value={tasks.length}
          sub={tasks.length ? t('dashboard.tasks.sub', { running }) : t('dashboard.tasks.empty')}
          color={running > 0 ? taskColors.running : 'text-ink-primary'}
        />
      </div>

      {/* Source state breakdown */}
      {sources.length > 0 && (
        <div className="px-6 pb-6">
          <div className="card overflow-hidden">
            <div className="px-4 py-3 border-b border-border">
              <span className="text-[10.5px] font-semibold uppercase tracking-widest text-ink-muted">
                {t('dashboard.source_states')}
              </span>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>{t('common.id')}</th>
                  <th>{t('common.state')}</th>
                  <th>{t('sources.col.reconnects')}</th>
                  <th>{t('sources.col.url')}</th>
                </tr>
              </thead>
              <tbody>
                {sources.slice(0, 8).map((s) => (
                  <tr key={s.id}>
                    <td className="font-mono text-[12px] text-ink-primary">{s.id}</td>
                    <td>
                      <span className={`font-mono text-[11px] font-medium ${sourceColors[s.state]}`}>
                        {s.state}
                      </span>
                    </td>
                    <td className="font-mono text-[12px] text-ink-secondary">{s.reconnect_count}</td>
                    <td className="font-mono text-[11px] text-ink-muted truncate max-w-[220px]">{s.url}</td>
                  </tr>
                ))}
                {sources.length > 8 && (
                  <tr>
                    <td colSpan={4} className="text-center text-[11px] text-ink-muted py-2">
                      {t('dashboard.more', { n: sources.length - 8 })}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Task state breakdown */}
      {tasks.length > 0 && (
        <div className="px-6 pb-6">
          <div className="card overflow-hidden">
            <div className="px-4 py-3 border-b border-border">
              <span className="text-[10.5px] font-semibold uppercase tracking-widest text-ink-muted">
                {t('dashboard.task_states')}
              </span>
            </div>
            <table className="data-table">
              <thead>
                <tr><th>{t('common.id')}</th><th>{t('common.state')}</th></tr>
              </thead>
              <tbody>
                {tasks.map((tk) => (
                  <tr key={tk.id}>
                    <td className="font-mono text-[12px] text-ink-primary">{tk.id}</td>
                    <td>
                      <span className={`font-mono text-[11px] font-medium ${taskColors[tk.state]}`}>
                        {tk.state}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
