import { useState } from 'react'
import { useSources, useAddSource, useRemoveSource, useUpdateSource } from '../hooks/queries'
import { SourceStatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton, EditButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import { ApiError } from '../lib/api'
import type { SourceCreate, SourceInfo, SourceUpdate } from '../types'

const DEFAULTS: SourceCreate = {
  id: '',
  url: '',
  reconnect_delay_ms: 1000,
  max_reconnect_delay_ms: 60000,
  max_reconnect_attempts: 5,
  open_timeout_ms: 5000,
  read_timeout_ms: 10000,
}

const CONNECTION_FIELDS = [
  { name: 'reconnect_delay_ms', label: 'sources.field.reconnect_delay', min: 0 },
  { name: 'max_reconnect_delay_ms', label: 'sources.field.max_delay', min: 0 },
  { name: 'max_reconnect_attempts', label: 'sources.field.max_attempts', min: 0 },
  { name: 'open_timeout_ms', label: 'sources.field.open_timeout', min: 1 },
  { name: 'read_timeout_ms', label: 'sources.field.read_timeout', min: 1 },
] as const

function SourceRow({ source, deleting, onDelete, onEdit }: {
  source: SourceInfo
  deleting: boolean
  onDelete: () => void
  onEdit: () => void
}) {
  const { lang, t } = useT()
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
      <td>
        <div className="flex items-center justify-end gap-1 pr-2 whitespace-nowrap">
          <EditButton onClick={onEdit} label={t('sources.edit')} />
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
  const update = useUpdateSource()
  const { t } = useT()

  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState<SourceInfo | null>(null)
  const [form, setForm] = useState<SourceCreate>(DEFAULTS)

  const isSaving = add.isPending || update.isPending
  const editUnavailable = editing !== null && !editing.config
  const set = (k: keyof SourceCreate, v: string | number) =>
    setForm((f) => ({ ...f, [k]: v }))

  const settingsValid = CONNECTION_FIELDS.every(({ name, min }) => {
    const value = form[name]
    return value === undefined || (editing && value === editing.config?.[name]) ||
      (Number.isInteger(value) && value >= min && value <= 2_147_483_647)
  })
  const canSave = Boolean(!editUnavailable && form.id.trim() && form.url.trim() && settingsValid)
  const apiError = update.error instanceof ApiError ? update.error : null
  const editError = update.error && t(
    apiError?.status === 409 ? 'sources.edit_in_use' :
    apiError?.status === 400 ? 'sources.edit_invalid' :
    apiError?.code === 'not_found' ? 'sources.edit_missing' :
    apiError?.status === 404 || apiError?.status === 405 ? 'sources.edit_unavailable' : 'sources.edit_failed',
  )

  const openForm = (source: SourceInfo | null) => {
    update.reset()
    setEditing(source)
    setForm(source ? { id: source.id, url: source.url, ...source.config } : DEFAULTS)
    setOpen(true)
  }

  const submit = () => {
    if (!canSave || isSaving) return
    if (editing) {
      const body: SourceUpdate = { url: form.url.trim() }
      for (const { name } of CONNECTION_FIELDS) {
        const value = form[name]
        if (value !== editing.config?.[name]) body[name] = value
      }
      update.mutate({ id: editing.id, body }, { onSuccess: () => setOpen(false) })
    } else {
      add.mutate(form, { onSuccess: () => setOpen(false) })
    }
  }

  return (
    <div>
      <PageHeader
        title={t('sources.title')}
        subtitle={t('sources.subtitle')}
        action={
          <button onClick={() => openForm(null)} className="btn-primary">
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
                <th className="text-right whitespace-nowrap">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={5} />}
              {!isLoading && !isError && sources.length === 0 && (
                <EmptyState cols={5} message={t('sources.empty')} />
              )}
              {sources.map((s) => (
                <SourceRow
                  key={s.id}
                  source={s}
                  deleting={remove.isPending}
                  onDelete={() => remove.mutate(s.id)}
                  onEdit={() => openForm(s)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Modal
        open={open}
        onClose={() => { if (!isSaving) setOpen(false) }}
        title={t(editing ? 'sources.edit_title' : 'sources.modal_title')}
      >
        <div className="space-y-4">
          {editUnavailable && <p role="alert" className="text-sm text-ink-secondary">{t('sources.edit_unavailable')}</p>}
          <fieldset disabled={editUnavailable} hidden={editUnavailable} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Field label={t('sources.field.id')} hint={editing ? undefined : t('sources.field.id_hint')}>
                <Input
                  placeholder="cam01"
                  aria-label={t('sources.field.id')}
                  value={form.id}
                  readOnly={editing !== null}
                  disabled={isSaving}
                  onChange={(e) => set('id', e.target.value)}
                />
              </Field>
              <Field label={t('sources.field.url')} hint={t('sources.field.url_hint')}>
                <Input
                  placeholder="rtsp://192.168.1.10/stream"
                  aria-label={t('sources.field.url')}
                  value={form.url}
                  disabled={isSaving}
                  onChange={(e) => set('url', e.target.value)}
                />
              </Field>
            </div>

            <details key={editing?.id ?? 'new-source'} className="group">
              <summary className="label cursor-pointer list-none flex items-center gap-1 select-none">
                <span className="group-open:rotate-90 transition-transform">▶</span>
                {t('sources.advanced')}
              </summary>
              <div className="mt-3 grid grid-cols-2 gap-4">
                {CONNECTION_FIELDS.map(({ name, label, min }) => (
                  <Field key={name} label={t(label)}>
                    <Input
                      type="number"
                      aria-label={t(label)}
                      min={min}
                      max={2_147_483_647}
                      value={form[name] ?? ''}
                      disabled={isSaving}
                      onChange={(e) => {
                        const value = e.target.value === '' ? undefined : e.target.valueAsNumber
                        setForm((previous) => ({ ...previous, [name]: value }))
                      }}
                    />
                  </Field>
                ))}
              </div>
            </details>
          </fieldset>

          {editing && editError && <p role="alert" className="text-sm text-danger">{editError}</p>}
          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setOpen(false)} disabled={isSaving} className="btn-ghost">{t('common.cancel')}</button>
            <button
              onClick={submit}
              disabled={!canSave || isSaving}
              className="btn-primary"
            >
              {editing
                ? t(update.isPending ? 'sources.saving' : 'common.save')
                : t(add.isPending ? 'sources.adding' : 'sources.add')}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
