import { useT } from '../../lib/i18n'
import type { SourceAvailability, TaskState } from '../../types'

type Status = 'online' | 'offline' | TaskState

const cfg: Record<Status, { dot: string; text: string; pulse: boolean }> = {
  online:   { dot: 'bg-success',   text: 'text-success',   pulse: false },
  offline:  { dot: 'bg-danger',    text: 'text-danger',    pulse: false },
  starting: { dot: 'bg-accent',    text: 'text-accent',    pulse: true  },
  failed:   { dot: 'bg-danger',    text: 'text-danger',    pulse: false },
  running:  { dot: 'bg-success',   text: 'text-success',   pulse: true  },
  stopped:  { dot: 'bg-ink-muted', text: 'text-ink-muted', pulse: false },
}

function StatusBadge({ status, label }: { status: Status; label: string }) {
  const c = cfg[status] ?? cfg.stopped
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[11px] font-medium ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot} ${c.pulse ? 'led-pulse' : ''}`} />
      {label}
    </span>
  )
}

export function SourceStatusBadge({ status }: { status?: SourceAvailability['status'] }) {
  const { t } = useT()
  if (status !== 'online' && status !== 'offline') return <span className="text-ink-muted">—</span>
  return <StatusBadge status={status} label={t(`sources.status.${status}`)} />
}

export function TaskStatusBadge({ state }: { state: TaskState }) {
  const { t } = useT()
  return <StatusBadge status={state} label={t(`tasks.state.${state}`)} />
}
