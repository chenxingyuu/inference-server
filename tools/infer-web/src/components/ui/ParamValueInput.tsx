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

  if (type === 'boolean' || (type === 'enum' && paramDef?.options)) {
    const opts = type === 'boolean' ? ['true', 'false'] : paramDef!.options!
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`input-field appearance-none text-[11px] ${className}`}
      >
        {opts.map((opt) => (
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
