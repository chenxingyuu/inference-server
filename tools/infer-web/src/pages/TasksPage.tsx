import { Fragment, useState, type FormEvent } from 'react'
import {
  useTasks, useAddTask, useUpdateTask, useRemoveTask, useStartTask, useStopTask,
  useSources, usePipelines,
} from '../hooks/queries'
import { TaskStatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input, Select } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton, EditButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import type { TaskCreate, TaskInfo, SamplingMode } from '../types'

type TaskForm = Omit<TaskCreate, 'sample_fps' | 'ascend_device_id'> & {
  sample_fps: string
  ascend_device_id: string
}

const TASK_INTEGER_MAX = 2_147_483_647
const TASK_COLUMN_COUNT = 5

const DEFAULTS: TaskForm = {
  id: '',
  source_id: '',
  pipeline_id: '',
  sample_fps: '5',
  sampling_mode: 'frame_count',
  use_hwdec: false,
  use_ascend_dvpp: false,
  ascend_device_id: '0',
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
      <path d="M4 3l9 5-9 5V3z" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
      <rect x="3" y="3" width="10" height="10" rx="1" />
    </svg>
  )
}

export function TasksPage() {
  const { data: tasks = [], isLoading } = useTasks()
  const { data: sources = [] } = useSources()
  const { data: pipelines = [] } = usePipelines()
  const { t } = useT()

  const add = useAddTask()
  const update = useUpdateTask()
  const remove = useRemoveTask()
  const start = useStartTask()
  const stop = useStopTask()

  const [open, setOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [form, setForm] = useState<TaskForm>(DEFAULTS)
  const [expanded, setExpanded] = useState<string | null>(null)

  const isEditing = editingId !== null
  const isSaving = add.isPending || update.isPending
  const saveError = isEditing ? update.error : add.error
  const sampleFps = Number(form.sample_fps)
  const ascendDeviceId = Number(form.ascend_device_id)
  const sampleFpsValid = Number.isInteger(sampleFps) && sampleFps >= 1 && sampleFps <= TASK_INTEGER_MAX
  const ascendDeviceValid = form.ascend_device_id !== '' && Number.isInteger(ascendDeviceId) &&
    ascendDeviceId >= 0 && ascendDeviceId <= TASK_INTEGER_MAX
  const canSave = !isSaving && !!form.id.trim() && !!form.source_id && !!form.pipeline_id &&
    sampleFpsValid && (!form.use_ascend_dvpp || ascendDeviceValid)

  const set = <K extends keyof TaskForm>(k: K, v: TaskForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const openAdd = () => {
    add.reset()
    setEditingId(null)
    setForm(DEFAULTS)
    setOpen(true)
  }

  const openEdit = (tk: TaskInfo) => {
    update.reset()
    setEditingId(tk.id)
    setForm({
      id: tk.id,
      source_id: tk.source_id ?? '',
      pipeline_id: tk.pipeline_id ?? '',
      sample_fps: tk.sample_fps?.toString() ?? DEFAULTS.sample_fps,
      sampling_mode: tk.sampling_mode ?? 'frame_count',
      use_hwdec: tk.use_hwdec ?? false,
      use_ascend_dvpp: tk.use_ascend_dvpp ?? false,
      ascend_device_id: tk.ascend_device_id?.toString() ?? DEFAULTS.ascend_device_id,
    })
    setOpen(true)
  }

  const closeModal = () => {
    setOpen(false)
    setEditingId(null)
    setForm(DEFAULTS)
  }

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!canSave) return
    const body: TaskCreate = {
      ...form,
      sample_fps: sampleFps,
      ascend_device_id: ascendDeviceValid ? ascendDeviceId : 0,
    }
    if (editingId !== null) {
      update.mutate({ id: editingId, body }, { onSuccess: closeModal })
    } else {
      add.mutate(body, { onSuccess: closeModal })
    }
  }

  const runningCount = tasks.filter((tk) => tk.state === 'running').length
  const sourceStates = new Map(sources.map((source) => [source.id, source.state]))

  return (
    <div>
      <PageHeader
        title={t('tasks.title')}
        subtitle={
          tasks.length
            ? t('tasks.subtitle_stats', { total: tasks.length, running: runningCount })
            : t('tasks.subtitle_empty')
        }
        action={
          <button onClick={openAdd} className="btn-primary">
            {t('tasks.add')}
          </button>
        }
      />

      <div className="section">
        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('tasks.field.id')}</th>
                <th>{t('tasks.col.state')}</th>
                <th>{t('tasks.col.source')}</th>
                <th>{t('tasks.col.pipeline')}</th>
                <th className="text-right pr-4">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={TASK_COLUMN_COUNT} />}
              {!isLoading && tasks.length === 0 && (
                <EmptyState message={t('tasks.empty')} cols={TASK_COLUMN_COUNT} />
              )}
              {tasks.map((tk) => {
                const inactive = tk.state === 'stopped' || tk.state === 'failed'
                const isExpanded = expanded === tk.id
                const sourceState = tk.source_id ? sourceStates.get(tk.source_id) : undefined
                const showSourceWarning = tk.state === 'running' &&
                  (sourceState === 'RECONNECTING' || sourceState === 'DEGRADED')
                const toggleDetails = () => setExpanded((current) => current === tk.id ? null : tk.id)

                return (
                  <Fragment key={tk.id}>
                    <tr className="cursor-pointer" onClick={toggleDetails}>
                      <td>
                        <button
                          type="button"
                          className="inline-flex items-center gap-2 text-left font-mono text-[12px] font-medium text-ink-primary"
                          aria-expanded={isExpanded}
                          onClick={(event) => {
                            event.stopPropagation()
                            toggleDetails()
                          }}
                        >
                          <span className="text-ink-muted" aria-hidden="true">{isExpanded ? '▼' : '▶'}</span>
                          {tk.id}
                        </button>
                      </td>
                      <td>
                        <TaskStatusBadge state={tk.state} />
                        {tk.cleanup_pending && <p className="mt-1 text-[11px] text-ink-muted">{t('tasks.cleaning_up')}</p>}
                      </td>
                      <td>
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                          <span className="font-mono text-[12px] text-ink-secondary">{tk.source_id || '—'}</span>
                          {showSourceWarning && (
                            <span className="whitespace-nowrap text-[11px] text-warning">
                              {t(sourceState === 'RECONNECTING' ? 'tasks.source.reconnecting' : 'tasks.source.degraded')}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="font-mono text-[12px] text-ink-secondary">{tk.pipeline_id || '—'}</td>
                      <td onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1 pr-2">
                          {inactive ? (
                            <>
                              <EditButton
                                disabled={tk.cleanup_pending}
                                onClick={() => openEdit(tk)}
                                label={t('tasks.edit')}
                              />
                              <button
                                onClick={() => start.mutate(tk.id)}
                                disabled={start.isPending || tk.cleanup_pending}
                                title={t(tk.state === 'failed' ? 'tasks.retry' : 'tasks.start')}
                                aria-label={t(tk.state === 'failed' ? 'tasks.retry' : 'tasks.start')}
                                className="btn-icon text-success/60 hover:text-success"
                              >
                                <PlayIcon />
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => stop.mutate(tk.id)}
                              disabled={stop.isPending || tk.cleanup_pending}
                              title={t('tasks.stop')}
                              aria-label={t('tasks.stop')}
                              className="btn-icon text-warning/60 hover:text-warning"
                            >
                              <StopIcon />
                            </button>
                          )}
                          <DeleteButton
                            loading={remove.isPending}
                            onClick={() => remove.mutate(tk.id)}
                          />
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={TASK_COLUMN_COUNT} className="bg-bg-overlay px-6 py-3">
                          {tk.failure && (
                            <p className="mb-2 text-xs text-danger">
                              {t('tasks.failure_reason')} {tk.failure.message}
                            </p>
                          )}
                          <div className="label mb-2">{t('tasks.section.settings')}</div>
                          <dl className="grid w-fit grid-cols-[auto_auto] gap-x-4 gap-y-1 text-[12px]">
                            <dt className="text-ink-muted">{t('tasks.field.sample_fps')}</dt>
                            <dd className="font-mono text-ink-secondary">{tk.sample_fps ?? '—'}</dd>
                            <dt className="text-ink-muted">{t('tasks.field.sampling')}</dt>
                            <dd className="font-mono text-ink-secondary">{tk.sampling_mode ?? '—'}</dd>
                          </dl>
                          {(tk.use_hwdec || tk.use_ascend_dvpp) && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {tk.use_hwdec && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-bg-overlay border border-border text-[10px] font-mono text-ink-muted">
                                  hwdec
                                </span>
                              )}
                              {tk.use_ascend_dvpp && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-bg-overlay border border-border text-[10px] font-mono text-ink-muted">
                                  dvpp · device {tk.ascend_device_id ?? 0}
                                </span>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <Modal
        open={open}
        onClose={() => { if (!isSaving) closeModal() }}
        title={isEditing ? t('tasks.modal_edit_title') : t('tasks.modal_title')}
      >
        <form className="space-y-4" onSubmit={submit} noValidate>
          <Field label={t('tasks.field.id')}>
            <Input
              aria-label={t('tasks.field.id')}
              placeholder="task-cam01-detect"
              value={form.id}
              onChange={(e) => set('id', e.target.value)}
              disabled={isEditing}
              className={isEditing ? 'opacity-60 cursor-not-allowed' : ''}
            />
          </Field>

          <div className="grid grid-cols-2 gap-4">
            <Field label={t('tasks.field.source')} hint={t('tasks.field.source_hint')}>
              <Select
                aria-label={t('tasks.field.source')}
                value={form.source_id}
                onChange={(e) => set('source_id', e.target.value)}
              >
                <option value="">{t('tasks.select_source')}</option>
                {sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id} ({s.state})
                  </option>
                ))}
              </Select>
            </Field>
            <Field label={t('tasks.field.pipeline')} hint={t('tasks.field.pipeline_hint')}>
              <Select
                aria-label={t('tasks.field.pipeline')}
                value={form.pipeline_id}
                onChange={(e) => set('pipeline_id', e.target.value)}
              >
                <option value="">{t('tasks.select_pipeline')}</option>
                {pipelines.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id}
                  </option>
                ))}
              </Select>
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Field label={t('tasks.field.sample_fps')}>
              <Input
                aria-label={t('tasks.field.sample_fps')}
                aria-invalid={!sampleFpsValid}
                aria-describedby={sampleFpsValid ? undefined : 'task-sample-fps-error'}
                type="number"
                min={1}
                max={TASK_INTEGER_MAX}
                step={1}
                value={form.sample_fps}
                onChange={(e) => set('sample_fps', e.target.value)}
              />
              {!sampleFpsValid && (
                <p id="task-sample-fps-error" className="mt-1 text-[11px] text-danger">
                  {t('tasks.field.sample_fps_invalid', { max: TASK_INTEGER_MAX })}
                </p>
              )}
            </Field>
            <Field label={t('tasks.field.sampling')}>
              <Select
                aria-label={t('tasks.field.sampling')}
                value={form.sampling_mode}
                onChange={(e) => set('sampling_mode', e.target.value as SamplingMode)}
              >
                <option value="frame_count">frame_count</option>
                <option value="time_based">time_based</option>
              </Select>
            </Field>
          </div>

          <div className="flex flex-col gap-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.use_hwdec}
                onChange={(e) => set('use_hwdec', e.target.checked)}
                className="w-3.5 h-3.5 accent-accent"
              />
              <span className="text-[12px] text-ink-secondary">{t('tasks.field.hwdec')}</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.use_ascend_dvpp}
                onChange={(e) => set('use_ascend_dvpp', e.target.checked)}
                className="w-3.5 h-3.5 accent-accent"
              />
              <span className="text-[12px] text-ink-secondary">{t('tasks.field.ascend_dvpp')}</span>
            </label>
          </div>

          {form.use_ascend_dvpp && (
            <Field label={t('tasks.field.ascend_device')}>
              <Input
                aria-label={t('tasks.field.ascend_device')}
                aria-invalid={!ascendDeviceValid}
                aria-describedby={ascendDeviceValid ? undefined : 'task-ascend-device-error'}
                type="number"
                min={0}
                max={TASK_INTEGER_MAX}
                step={1}
                value={form.ascend_device_id}
                onChange={(e) => set('ascend_device_id', e.target.value)}
                className="w-32"
              />
              {!ascendDeviceValid && (
                <p id="task-ascend-device-error" className="mt-1 text-[11px] text-danger">
                  {t('tasks.field.ascend_device_invalid', { max: TASK_INTEGER_MAX })}
                </p>
              )}
            </Field>
          )}

          {saveError && <p role="alert" className="text-xs text-danger">{saveError.message}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <button type="button" onClick={closeModal} disabled={isSaving} className="btn-ghost">
              {t('common.cancel')}
            </button>
            <button
              type="submit"
              disabled={!canSave}
              className="btn-primary"
            >
              {isEditing
                ? (update.isPending ? t('tasks.saving') : t('tasks.save'))
                : (add.isPending ? t('tasks.creating') : t('tasks.add'))}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  )
}
