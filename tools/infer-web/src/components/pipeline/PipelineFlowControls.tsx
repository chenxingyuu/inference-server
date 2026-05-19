import type { ReactNode } from 'react'
import { Panel, useReactFlow } from '@xyflow/react'
import { useT } from '../../lib/i18n'

function ControlButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      className="btn-icon w-8 h-8 flex items-center justify-center rounded-sm"
      aria-label={label}
      title={label}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

export function PipelineFlowControls() {
  const { t } = useT()
  const { zoomIn, zoomOut, fitView } = useReactFlow()

  return (
    <Panel position="bottom-left" className="!m-3">
      <div className="flex flex-col gap-0.5 p-1 rounded border border-border bg-bg-surface">
        <ControlButton label={t('pipelines.editor.zoom_in')} onClick={() => zoomIn({ duration: 150 })}>
          <PlusIcon />
        </ControlButton>
        <ControlButton label={t('pipelines.editor.zoom_out')} onClick={() => zoomOut({ duration: 150 })}>
          <MinusIcon />
        </ControlButton>
        <ControlButton
          label={t('pipelines.editor.fit_view')}
          onClick={() => void fitView({ padding: 0.2, maxZoom: 1, duration: 200 })}
        >
          <FitIcon />
        </ControlButton>
      </div>
    </Panel>
  )
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <path d="M7 2.5v9M2.5 7h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

function MinusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <path d="M2.5 7h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

function FitIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <path
        d="M3.5 5V3.5H5M9 3.5H10.5V5M10.5 9V10.5H9M5 10.5H3.5V9"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  )
}
