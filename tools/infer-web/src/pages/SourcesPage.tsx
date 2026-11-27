import { useState } from 'react'
import { useSources, useAddSource, useProbeSource, useRemoveSource } from '../hooks/queries'
import { StatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { sourceReasonKey, useT } from '../lib/i18n'
import type { SourceAvailability, SourceCreate, SourceInfo, SourceState } from '../types'

const DEFAULTS: SourceCreate = {
  id: '',
  url: '',
  reconnect_delay_ms: 1000,
  max_reconnect_delay_ms: 60000,
  max_reconnect_attempts: 5,
  open_timeout_ms: 5000,
  read_timeout_ms: 10000,
}

const AVAILABILITY_COLORS: Record<SourceAvailability['status'], SourceState> = {
  online: 'STREAMING',
  connecting: 'CONNECTING',
  reconnecting: 'RECONNECTING',
  unavailable: 'FAILED',
  probe_passed: 'STREAMING',
  probe_failed: 'FAILED',
  unchecked: 'STOPPED',
  expired: 'STOPPED',
  unsupported: 'STOPPED',
}

function SourceRow({ source, deleting, onDelete }: {
  source: SourceInfo
  deleting: boolean
  onDelete: () => void
}) {
  const { lang, t } = useT()
  const probe = useProbeSource(source.id)
  const { availability, last_probe } = source
  const recentCheck = last_probe && (availability?.basis === 'runtime' || availability?.status === 'expired')
  // Avoid exposing camera credentials on the page while retaining the original API value.
  const displayUrl = source.url.replace(/^(rtsp:\/\/)[^/?#]*@/, '$1')
  const formatTime = (timestamp: number) => new Date(timestamp).toLocaleString(lang)

  return (
    <tr>
      <td className="font-mono text-[12px] text-ink-primary font-medium">{source.id}</td>
      <td>
        {availability ? (
          <>
            <StatusBadge
              status={AVAILABILITY_COLORS[availability.status]}
              label={t(`sources.status.${availability.status}`)}
              pulse={availability.basis === 'probe' ? false : undefined}
            />
            {availability.observed_at_ms !== undefined && availability.status !== 'expired' && (
              <div className="text-[11px] text-ink-muted">
                {t(availability.status === 'online' || availability.reason === 'no_recent_frame'
                  ? 'sources.last_frame'
                  : availability.basis === 'probe' ? 'sources.checked_at' : 'sources.observed_at',
                { time: formatTime(availability.observed_at_ms) })}
              </div>
            )}
            {availability.reason && availability.status !== 'expired' && (
              <div className="text-[11px] text-ink-secondary">{t(sourceReasonKey(availability.reason))}</div>
            )}
            {recentCheck && (
              <div className="text-[11px] text-ink-muted">
                {t(last_probe.passed ? 'sources.last_passed' : 'sources.last_failed',
                  { time: formatTime(last_probe.checked_at_ms) })}
                {last_probe.reason && ` · ${t(sourceReasonKey(last_probe.reason))}`}
              </div>
            )}
          </>
        ) : <StatusBadge status={source.state} />}
      </td>
      <td className="font-mono text-[12px] text-ink-secondary">{source.reconnect_count}</td>
      <td className="font-mono text-[11px] text-ink-muted max-w-xs truncate">{displayUrl}</td>
      <td className="text-right pr-2">
        <div className="flex items-center justify-end gap-2 whitespace-nowrap">
          {availability && availability.status !== 'unsupported' && (
            <button type="button" onClick={() => probe.mutate()} disabled={probe.isPending} className="btn-ghost">
              {t(probe.isPending ? 'sources.probing' : 'sources.probe')}
            </button>
          )}
          <DeleteButton loading={deleting} onClick={onDelete} />
        </div>
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
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('sources.col.availability')}</th>
                <th>{t('sources.col.reconnects')}</th>
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
