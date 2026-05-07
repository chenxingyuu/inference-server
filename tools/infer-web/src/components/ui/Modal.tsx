import { useEffect, type ReactNode } from 'react'

interface Props {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  width?: string
}

export function Modal({ open, onClose, title, children, width = 'max-w-lg' }: Props) {
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-bg-base/80 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className={`relative w-full ${width} bg-bg-surface border border-border rounded-lg shadow-2xl`}>
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="text-sm font-semibold text-ink-primary">{title}</h2>
          <button
            onClick={onClose}
            className="btn-icon text-lg leading-none"
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className="p-5 max-h-[calc(100vh-8rem)] overflow-y-auto">{children}</div>
      </div>
    </div>
  )
}
