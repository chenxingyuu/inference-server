import { useState } from 'react'
import { useSources, useAddSource, useRemoveSource } from '../hooks/queries'
import { StatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import type { SourceCreate } from '../types'

const DEFAULTS: SourceCreate = {
  id: '',
  url: '',
  reconnect_delay_ms: 1000,
  max_reconnect_delay_ms: 60000,
  max_reconnect_attempts: 5,
  open_timeout_ms: 5000,
  read_timeout_ms: 10000,
}

export function SourcesPage() {
  const { data: sources = [], isLoading } = useSources()
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
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('common.state')}</th>
                <th>{t('sources.col.reconnects')}</th>
                <th>{t('sources.col.url')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={5} />}
              {!isLoading && sources.length === 0 && (
                <EmptyState message={t('sources.empty')} />
              )}
              {sources.map((s) => (
                <tr key={s.id}>
                  <td className="font-mono text-[12px] text-ink-primary font-medium">{s.id}</td>
                  <td><StatusBadge status={s.state} /></td>
                  <td className="font-mono text-[12px] text-ink-secondary">{s.reconnect_count}</td>
                  <td className="font-mono text-[11px] text-ink-muted max-w-xs truncate">{s.url}</td>
                  <td className="text-right pr-2">
                    <DeleteButton
                      loading={remove.isPending}
                      onClick={() => remove.mutate(s.id)}
                    />
                  </td>
                </tr>
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
