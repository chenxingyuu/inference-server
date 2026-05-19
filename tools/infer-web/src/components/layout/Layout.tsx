import { useState, useEffect, type ReactNode } from 'react'
import { Sidebar } from './Sidebar'

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-bg-base">{children}</main>
    </div>
  )
}

interface PageHeaderProps {
  title: string
  subtitle?: string
  action?: ReactNode
}

export function PageHeader({ title, subtitle, action }: PageHeaderProps) {
  return (
    <div className="page-header">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return (
    <tr>
      <td colSpan={99} className="text-center py-12 text-ink-muted text-sm">
        {message}
      </td>
    </tr>
  )
}

export function LoadingRows({ cols }: { cols: number }) {
  return (
    <>
      {[0, 1, 2].map((i) => (
        <tr key={i} className="animate-pulse">
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j} className="px-4 py-3 border-b border-border/60">
              <div className="h-3 bg-bg-elevated rounded w-3/4" />
            </td>
          ))}
        </tr>
      ))}
    </>
  )
}

export function DeleteButton({ onClick, loading }: { onClick: () => void; loading?: boolean }) {
  const [confirming, setConfirming] = useState(false)

  useEffect(() => {
    if (!confirming) return
    const t = setTimeout(() => setConfirming(false), 3000)
    return () => clearTimeout(t)
  }, [confirming])

  if (confirming) {
    return (
      <div className="flex items-center gap-1">
        <button
          onClick={() => { setConfirming(false); onClick() }}
          className="btn-icon text-danger hover:text-danger bg-danger/10 border border-danger/25 rounded px-1"
          title="Confirm"
        >
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="w-3 h-3">
            <path d="M2 8l4 4 8-8" />
          </svg>
        </button>
        <button
          onClick={() => setConfirming(false)}
          className="btn-icon text-ink-muted hover:text-ink-secondary"
          title="Cancel"
        >
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="w-3 h-3">
            <path d="M3 3l10 10M13 3L3 13" />
          </svg>
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      disabled={loading}
      className="btn-icon text-danger/60 hover:text-danger"
      title="Remove"
    >
      <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
        <path d="M6 2h4a1 1 0 0 1 1 1H5a1 1 0 0 1 1-1zM2 4h12l-1 9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1L2 4zm4 2v5m4-5v5" stroke="currentColor" strokeWidth="1" fill="none" strokeLinecap="round" />
      </svg>
    </button>
  )
}
