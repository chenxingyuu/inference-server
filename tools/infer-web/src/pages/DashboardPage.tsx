import { useHealth, useSources, useModels, usePipelines, useTasks } from '../hooks/queries'
import { serverUrl } from '../lib/api'
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
        title="Dashboard"
        subtitle="System overview — auto-refreshes every 5–30s"
      />

      {/* Health banner */}
      <div className="px-6 pt-5">
        {(() => {
          const s = health?.status
          const cfg =
            s === 'ok'          ? { cls: 'bg-success/8 border-success/20 text-success',  dot: 'bg-success led-pulse', msg: 'Engine healthy' } :
            s === 'engine_down' ? { cls: 'bg-warning/8 border-warning/20 text-warning',  dot: 'bg-warning',          msg: 'C++ engine not connected — Go server is reachable, start the inference engine' } :
            s === 'unreachable' ? { cls: 'bg-danger/8  border-danger/25  text-danger',   dot: 'bg-danger',           msg: `Cannot reach server — check the Server URL in the sidebar (${serverUrl.get()})` } :
                                  { cls: 'bg-bg-elevated border-border   text-ink-muted', dot: 'bg-ink-muted',        msg: 'Checking…' }
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
          label="Sources"
          value={sources.length}
          sub={sources.length ? `${streaming} streaming · ${degraded} degraded` : 'none configured'}
          color={sourceColor}
        />
        <StatCard
          label="Models"
          value={models.length}
          sub={models.length ? models.map((m) => m.backend).join(', ') : 'none loaded'}
        />
        <StatCard
          label="Pipelines"
          value={pipelines.length}
          sub={pipelines.length ? `${pipelines.reduce((s, p) => s + p.nodes.length, 0)} nodes total` : 'none defined'}
        />
        <StatCard
          label="Tasks"
          value={tasks.length}
          sub={tasks.length ? `${running} running` : 'none created'}
          color={running > 0 ? taskColors.running : 'text-ink-primary'}
        />
      </div>

      {/* Source state breakdown */}
      {sources.length > 0 && (
        <div className="px-6 pb-6">
          <div className="card overflow-hidden">
            <div className="px-4 py-3 border-b border-border">
              <span className="text-[10.5px] font-semibold uppercase tracking-widest text-ink-muted">
                Source States
              </span>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>State</th>
                  <th>Reconnects</th>
                  <th>URL</th>
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
                      +{sources.length - 8} more — view all in Sources
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
                Task States
              </span>
            </div>
            <table className="data-table">
              <thead>
                <tr><th>ID</th><th>State</th></tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t.id}>
                    <td className="font-mono text-[12px] text-ink-primary">{t.id}</td>
                    <td>
                      <span className={`font-mono text-[11px] font-medium ${taskColors[t.state]}`}>
                        {t.state}
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
