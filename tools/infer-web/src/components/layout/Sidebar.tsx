import { NavLink } from 'react-router-dom'
import { useHealth } from '../../hooks/queries'
import { auth, serverUrl } from '../../lib/api'
import { useT } from '../../lib/i18n'
import { useState } from 'react'

function ConfigRow({
  label,
  value,
  masked,
  placeholder,
  onSave,
  notSetLabel,
  saveLabel,
  cancelLabel,
}: {
  label: string
  value: string
  masked?: boolean
  placeholder: string
  onSave: (v: string) => void
  notSetLabel: string
  saveLabel: string
  cancelLabel: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  const save = () => {
    onSave(draft.trim())
    setEditing(false)
    setDraft('')
    window.location.reload()
  }

  if (editing) {
    return (
      <div className="space-y-1.5">
        <input
          autoFocus
          type={masked ? 'password' : 'text'}
          placeholder={placeholder}
          className="input-field text-[12px]"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') save()
            if (e.key === 'Escape') setEditing(false)
          }}
        />
        <div className="flex gap-1.5">
          <button onClick={save} className="btn-primary flex-1 justify-center text-[11px] py-1">{saveLabel}</button>
          <button onClick={() => setEditing(false)} className="btn-ghost px-2 py-1 text-[11px]">{cancelLabel}</button>
        </div>
      </div>
    )
  }

  return (
    <button
      onClick={() => { setDraft(value); setEditing(true) }}
      className="w-full text-left px-2 py-1.5 rounded hover:bg-bg-overlay transition-colors"
    >
      <div className="text-[10px] text-ink-muted uppercase tracking-wide mb-0.5">{label}</div>
      <div className="font-mono text-[11px] text-ink-secondary truncate">
        {value
          ? masked ? '••••••••' + value.slice(-4) : value
          : <span className="text-warning">{notSetLabel}</span>
        }
      </div>
    </button>
  )
}

export function Sidebar() {
  const { data: health } = useHealth()
  const { lang, setLang, t } = useT()

  const dotColor =
    health?.status === 'ok'          ? 'bg-success led-pulse' :
    health?.status === 'engine_down' ? 'bg-warning' :
    health?.status === 'unreachable' ? 'bg-danger' :
                                       'bg-ink-muted'

  const nav = [
    {
      to: '/',
      label: t('nav.dashboard'),
      icon: (
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
          <rect x="1" y="1" width="6" height="6" rx="1" />
          <rect x="9" y="1" width="6" height="6" rx="1" />
          <rect x="1" y="9" width="6" height="6" rx="1" />
          <rect x="9" y="9" width="6" height="6" rx="1" />
        </svg>
      ),
    },
    {
      to: '/sources',
      label: t('nav.sources'),
      icon: (
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
          <circle cx="8" cy="6" r="3" />
          <path d="M2 13c0-3.3 2.7-5 6-5s6 1.7 6 5" opacity=".5" />
          <circle cx="8" cy="6" r="5.5" fill="none" stroke="currentColor" strokeWidth="1" opacity=".3" />
        </svg>
      ),
    },
    {
      to: '/models',
      label: t('nav.models'),
      icon: (
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
          <rect x="2" y="4" width="12" height="8" rx="1" opacity=".3" />
          <rect x="2" y="4" width="12" height="2.5" rx="1" />
          <circle cx="4.5" cy="10" r="1" />
          <circle cx="7.5" cy="10" r="1" />
          <path d="M10 9.5h3.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      ),
    },
    {
      to: '/pipelines',
      label: t('nav.pipelines'),
      icon: (
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
          <circle cx="3" cy="8" r="2" />
          <circle cx="13" cy="8" r="2" />
          <circle cx="8" cy="4" r="2" />
          <circle cx="8" cy="12" r="2" opacity=".5" />
          <path d="M5 8h6M8 6v2" stroke="currentColor" strokeWidth="1" fill="none" />
        </svg>
      ),
    },
    {
      to: '/tasks',
      label: t('nav.tasks'),
      icon: (
        <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
          <rect x="2" y="2" width="12" height="2" rx="1" />
          <rect x="2" y="6.5" width="9" height="2" rx="1" />
          <rect x="2" y="11" width="11" height="2" rx="1" />
          <circle cx="13.5" cy="7.5" r="2.5" className="fill-success" />
        </svg>
      ),
    },
  ]

  return (
    <aside className="w-52 flex-none flex flex-col bg-bg-surface border-r border-border h-screen sticky top-0">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 py-4 border-b border-border">
        <span className={`w-2 h-2 rounded-full ${dotColor}`} />
        <span className="font-mono font-medium text-sm text-ink-primary">infer-server</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {nav.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 mx-2 rounded text-sm transition-colors duration-100
               ${isActive
                 ? 'bg-accent/10 text-accent border border-accent/20'
                 : 'text-ink-secondary hover:text-ink-primary hover:bg-bg-overlay border border-transparent'
               }`
            }
          >
            {icon}
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Connection config */}
      <div className="border-t border-border p-3 space-y-1">
        <ConfigRow
          label={t('sidebar.server')}
          value={serverUrl.get()}
          placeholder={window.location.origin}
          onSave={serverUrl.set}
          notSetLabel={t('sidebar.not_set')}
          saveLabel={t('common.save')}
          cancelLabel={t('common.cancel')}
        />
        <ConfigRow
          label={t('sidebar.api_key')}
          value={auth.get()}
          masked
          placeholder="Paste API key…"
          onSave={auth.set}
          notSetLabel={t('sidebar.not_set')}
          saveLabel={t('common.save')}
          cancelLabel={t('common.cancel')}
        />

        {/* Language toggle */}
        <div className="flex gap-1 pt-1 px-2">
          <button
            onClick={() => setLang('en')}
            className={`flex-1 text-[11px] py-1 rounded transition-colors ${
              lang === 'en'
                ? 'bg-accent/15 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink-secondary border border-transparent hover:bg-bg-overlay'
            }`}
          >
            EN
          </button>
          <button
            onClick={() => setLang('zh')}
            className={`flex-1 text-[11px] py-1 rounded transition-colors ${
              lang === 'zh'
                ? 'bg-accent/15 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink-secondary border border-transparent hover:bg-bg-overlay'
            }`}
          >
            中
          </button>
        </div>
      </div>
    </aside>
  )
}
