import type { ParamDef } from '../../lib/nodeTypes'

interface ParamValueInputProps {
  paramDef: ParamDef | undefined
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

export function ParamValueInput({
  paramDef,
  value,
  onChange,
  placeholder,
  className = '',
}: ParamValueInputProps) {
  const type = paramDef?.type ?? 'string'

  if (type === 'boolean') {
    const checked = value === 'true'
    return (
      <div className={`flex items-center gap-2 px-2 ${className}`}>
        <button
          type="button"
          role="switch"
          aria-checked={checked}
          onClick={() => onChange(checked ? 'false' : 'true')}
          className={`relative inline-flex h-4 w-7 flex-shrink-0 items-center rounded-full transition-colors duration-150
            ${checked ? 'bg-accent' : 'bg-border hover:bg-ink-muted/40'}`}
        >
          <span
            className={`inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform duration-150
              ${checked ? 'translate-x-3.5' : 'translate-x-0.5'}`}
          />
        </button>
        <span className="text-[11px] text-ink-secondary select-none">{value}</span>
      </div>
    )
  }

  if (type === 'enum' && paramDef?.options) {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`input-field appearance-none text-[11px] ${className}`}
      >
        {paramDef.options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    )
  }

  if (type === 'number') {
    return (
      <input
        type="number"
        value={value}
        min={paramDef?.min}
        max={paramDef?.max}
        step={paramDef?.step}
        onChange={(e) => onChange(e.target.value)}
        className={`input-field text-[11px] ${className}`}
      />
    )
  }

  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={`input-field text-[11px] ${className}`}
    />
  )
}
