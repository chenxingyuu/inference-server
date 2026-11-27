import { useState } from 'react'
import { useSources, useAddSource, useRemoveSource } from '../hooks/queries'
import { SourceStatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import type { SourceCreate, SourceInfo } from '../types'

const DEFAULTS: SourceCreate = {
  id: '',
  url: '',
  reconnect_delay_ms: 1000,
  max_reconnect_delay_ms: 60000,
  max_reconnect_attempts: 5,
  open_timeout_ms: 5000,
  read_timeout_ms: 10000,
}

function SourceRow({ source, deleting, onDelete }: {
  source: SourceInfo
  deleting: boolean
  onDelete: () => void
}) {
  const { lang } = useT()
  const { availability, last_probe } = source
  const checkedAt = last_probe ? new Date(last_probe.checked_at_ms) : null
  // Avoid exposing camera credentials on the page while retaining the original API value.
  const displayUrl = source.url.replace(/^(rtsp:\/\/)[^/?#]*@/, '$1')

  return (
    <tr>
      <td className="font-mono text-[12px] text-ink-primary font-medium">{source.id}</td>
      <td>
        <SourceStatusBadge status={availability?.status} />
      </td>
      <td className="whitespace-nowrap font-mono text-[11px] text-ink-muted">
        {checkedAt ? <time dateTime={checkedAt.toISOString()}>{checkedAt.toLocaleString(lang)}</time> : '—'}
      </td>
      <td className="font-mono text-[11px] text-ink-muted max-w-xs truncate">{displayUrl}</td>
      <td className="text-right pr-2">
        <DeleteButton loading={deleting} onClick={onDelete} />
      </td>
    </tr>
  )
}

export function SourcesPage() {
  const { data: sources = [], isLoading, isError } = useSources()
  const add = useAddSource()
  const remove = useRemoveSource()
  const { t } = useT()

  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<SourceCreate>(DEFAULTS)

  const set = (k: keyof SourceCreate, v: string | number) =>
    setForm((f) => ({ ...f, [k]: v }))

  const submit = () => {
    if (!form.id || !form.url) return
    add.mutate(form, { onSuccess: () => { setOpen(false); setForm(DEFAULTS) } })
  }

  return (
    <div>
      <PageHeader
        title={t('sources.title')}
        subtitle={t('sources.subtitle')}
        action={
          <button onClick={() => setOpen(true)} className="btn-primary">
            {t('sources.add')}
          </button>
        }
      />

      <div className="section">
        {isError && <p role="alert" className="mb-2 text-sm text-danger">{t('sources.refresh_failed')}</p>}
        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('sources.col.availability')}</th>
                <th className="whitespace-nowrap">{t('sources.col.checked_at')}</th>
                <th>{t('sources.col.url')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={5} />}
              {!isLoading && !isError && sources.length === 0 && (
                <EmptyState cols={5} message={t('sources.empty')} />
              )}
              {sources.map((s) => (
                <SourceRow key={s.id} source={s} deleting={remove.isPending} onDelete={() => remove.mutate(s.id)} />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title={t('sources.modal_title')}>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Field label={t('sources.field.id')} hint={t('sources.field.id_hint')}>
              <Input
                placeholder="cam01"
                value={form.id}
                onChange={(e) => set('id', e.target.value)}
              />
            </Field>
            <Field label={t('sources.field.url')} hint={t('sources.field.url_hint')}>
              <Input
                placeholder="rtsp://192.168.1.10/stream"
                value={form.url}
                onChange={(e) => set('url', e.target.value)}
              />
            </Field>
          </div>

          <details className="group">
            <summary className="label cursor-pointer list-none flex items-center gap-1 select-none">
              <span className="group-open:rotate-90 transition-transform">▶</span>
              {t('sources.advanced')}
            </summary>
            <div className="mt-3 grid grid-cols-2 gap-4">
              <Field label={t('sources.field.reconnect_delay')}>
                <Input type="number" value={form.reconnect_delay_ms} onChange={(e) => set('reconnect_delay_ms', +e.target.value)} />
              </Field>
              <Field label={t('sources.field.max_delay')}>
                <Input type="number" value={form.max_reconnect_delay_ms} onChange={(e) => set('max_reconnect_delay_ms', +e.target.value)} />
              </Field>
              <Field label={t('sources.field.max_attempts')}>
                <Input type="number" value={form.max_reconnect_attempts} onChange={(e) => set('max_reconnect_attempts', +e.target.value)} />
              </Field>
              <Field label={t('sources.field.open_timeout')}>
                <Input type="number" value={form.open_timeout_ms} onChange={(e) => set('open_timeout_ms', +e.target.value)} />
              </Field>
              <Field label={t('sources.field.read_timeout')}>
                <Input type="number" value={form.read_timeout_ms} onChange={(e) => set('read_timeout_ms', +e.target.value)} />
              </Field>
            </div>
          </details>

          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setOpen(false)} className="btn-ghost">{t('common.cancel')}</button>
            <button
              onClick={submit}
              disabled={!form.id || !form.url || add.isPending}
              className="btn-primary"
            >
              {add.isPending ? t('sources.adding') : t('sources.add')}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
