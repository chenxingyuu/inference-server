import type { InputHTMLAttributes, SelectHTMLAttributes, ReactNode } from 'react'

interface FieldProps {
  label: string
  hint?: string
  children: ReactNode
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div>
      <label className="label">{label}</label>
      {children}
      {hint && <p className="text-[11px] text-ink-muted mt-1">{hint}</p>}
    </div>
  )
}

export function Input({
  className = '',
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`input-field ${className}`} {...props} />
}

export function Select({
  className = '',
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={`input-field appearance-none ${className}`}
      {...props}
    >
      {children}
    </select>
  )
}
