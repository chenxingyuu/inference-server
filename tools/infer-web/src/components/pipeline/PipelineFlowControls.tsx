import type { ReactNode } from 'react'
import { Panel, useReactFlow } from '@xyflow/react'
import { useT } from '../../lib/i18n'

function ControlButton({
  label,
  onClick,
  disabled = false,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      className="btn-icon w-8 h-8 flex items-center justify-center rounded-sm disabled:opacity-35 disabled:pointer-events-none"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

interface PipelineFlowControlsProps {
  onAutoLayout: () => void
  layoutDisabled?: boolean
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
}

export function PipelineFlowControls({
  onAutoLayout,
  layoutDisabled = false,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
}: PipelineFlowControlsProps) {
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
        <div className="border-t border-border/80 pt-0.5 mt-0.5 flex flex-col gap-0.5">
          <ControlButton
            label={t('pipelines.editor.undo')}
            onClick={onUndo}
            disabled={!canUndo}
          >
            <UndoIcon />
          </ControlButton>
          <ControlButton
            label={t('pipelines.editor.redo')}
            onClick={onRedo}
            disabled={!canRedo}
          >
            <RedoIcon />
          </ControlButton>
          <ControlButton
            label={t('pipelines.editor.auto_layout')}
            onClick={onAutoLayout}
            disabled={layoutDisabled}
          >
            <LayoutIcon />
          </ControlButton>
        </div>
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

function UndoIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <path
        d="M4 5.5H9a2.5 2.5 0 1 1 0 5H8M4 5.5L5.75 3.75M4 5.5L5.75 7.25"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  )
}

function RedoIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <path
        d="M10 5.5H5a2.5 2.5 0 1 0 0 5H6M10 5.5L8.25 3.75M10 5.5L8.25 7.25"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  )
}

function LayoutIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="text-ink-secondary">
      <rect x="2" y="4" width="3" height="2.5" rx="0.5" stroke="currentColor" strokeWidth="1.1" fill="none" />
      <rect x="5.5" y="7.5" width="3" height="2.5" rx="0.5" stroke="currentColor" strokeWidth="1.1" fill="none" />
      <rect x="9" y="4" width="3" height="2.5" rx="0.5" stroke="currentColor" strokeWidth="1.1" fill="none" />
      <path d="M5 5.25H5.5M8.5 8.75H9M8.5 5.25l1 1.25M5.5 8.75l1-1.25" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
    </svg>
  )
}
