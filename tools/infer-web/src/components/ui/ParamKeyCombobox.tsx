import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ParamDef } from '../../lib/nodeTypes'

interface ParamKeyComboboxProps {
  value: string
  options: ParamDef[]
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

export function ParamKeyCombobox({
  value,
  options,
  onChange,
  placeholder,
  className = '',
}: ParamKeyComboboxProps) {
  const [query, setQuery] = useState(value)
  const [open, setOpen] = useState(false)
  const [dropPos, setDropPos] = useState({ top: 0, left: 0, width: 0 })
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { setQuery(value) }, [value])

  const filtered = options.filter(
    (o) => !query || o.k.toLowerCase().includes(query.toLowerCase()),
  )

  const openDrop = () => {
    if (!inputRef.current) return
    const rect = inputRef.current.getBoundingClientRect()
    setDropPos({
      top: rect.bottom + 2,
      left: rect.left,
      width: Math.max(rect.width, 240),
    })
    setOpen(true)
  }

  const commit = () => {
    setOpen(false)
    if (query !== value) onChange(query)
  }

  const handleSelect = (k: string) => {
    setQuery(k)
    onChange(k)
    setOpen(false)
  }

  return (
    <div className={`relative ${className}`}>
      <input
        ref={inputRef}
        className="input-field text-[11px]"
        value={query}
        placeholder={placeholder}
        onChange={(e) => {
          setQuery(e.target.value)
          if (!open) openDrop()
        }}
        onFocus={openDrop}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setOpen(false)
            inputRef.current?.blur()
          }
        }}
      />
      {open &&
        createPortal(
          <div
            style={{
              position: 'fixed',
              top: dropPos.top,
              left: dropPos.left,
              width: dropPos.width,
              zIndex: 9999,
            }}
            className="bg-bg-surface border border-border rounded shadow-xl max-h-52 overflow-y-auto py-1"
          >
            {filtered.length === 0 ? (
              <p className="px-3 py-2 text-[10px] text-ink-muted">No matching params</p>
            ) : (
              filtered.map((o) => (
                <button
                  key={o.k}
                  type="button"
                  className="w-full text-left px-3 py-1.5 hover:bg-bg-overlay flex flex-col gap-0.5 group"
                  onMouseDown={(e) => {
                    e.preventDefault()
                    handleSelect(o.k)
                  }}
                >
                  <span className="text-[11px] font-medium text-ink-primary group-hover:text-accent transition-colors">
                    {o.k}
                  </span>
                  {o.description && (
                    <span className="text-[10px] text-ink-muted leading-snug">
                      {o.description}
                    </span>
                  )}
                </button>
              ))
            )}
          </div>,
          document.body,
        )}
    </div>
  )
}
