import { useT } from '../../lib/i18n'
import type { SourceState, TaskState } from '../../types'

type Status = SourceState | TaskState

const cfg: Record<Status, { dot: string; text: string; pulse: boolean; label: string }> = {
  STREAMING:    { dot: 'bg-success',  text: 'text-success',  pulse: true,  label: 'STREAMING'    },
  CONNECTING:   { dot: 'bg-accent',   text: 'text-accent',   pulse: true,  label: 'CONNECTING'   },
  RECONNECTING: { dot: 'bg-warning',  text: 'text-warning',  pulse: true,  label: 'RECONNECTING' },
  DEGRADED:     { dot: 'bg-warning',  text: 'text-warning',  pulse: false, label: 'DEGRADED'     },
  STOPPED:      { dot: 'bg-ink-muted',text: 'text-ink-muted',pulse: false, label: 'STOPPED'      },
  starting:     { dot: 'bg-accent',   text: 'text-accent',   pulse: true,  label: 'starting'     },
  failed:       { dot: 'bg-danger',   text: 'text-danger',   pulse: false, label: 'failed'       },
  running:      { dot: 'bg-success',  text: 'text-success',  pulse: true,  label: 'running'      },
  stopped:      { dot: 'bg-ink-muted',text: 'text-ink-muted',pulse: false, label: 'stopped'      },
}

export function StatusBadge({ status, label }: { status: Status; label?: string }) {
  const c = cfg[status] ?? cfg.STOPPED
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[11px] font-medium ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot} ${c.pulse ? 'led-pulse' : ''}`} />
      {label ?? c.label}
    </span>
  )
}

export function TaskStatusBadge({ state }: { state: TaskState }) {
  const { t } = useT()
  return <StatusBadge status={state} label={t(`tasks.state.${state}`)} />
}
