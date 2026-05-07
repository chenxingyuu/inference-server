import { useState } from 'react'
import {
  useTasks, useAddTask, useRemoveTask, useStartTask, useStopTask,
  useSources, usePipelines,
} from '../hooks/queries'
import { StatusBadge } from '../components/ui/StatusBadge'
import { Modal } from '../components/ui/Modal'
import { Field, Input, Select } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import type { TaskCreate, SamplingMode } from '../types'

const DEFAULTS: TaskCreate = {
  id: '',
  source_id: '',
  pipeline_id: '',
  sample_fps: 25,
  sampling_mode: 'frame_count',
  use_hwdec: false,
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
  const remove = useRemoveTask()
  const start = useStartTask()
  const stop = useStopTask()

  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<TaskCreate>(DEFAULTS)

  const set = <K extends keyof TaskCreate>(k: K, v: TaskCreate[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const submit = () => {
    if (!form.id || !form.source_id || !form.pipeline_id) return
    add.mutate(form, { onSuccess: () => { setOpen(false); setForm(DEFAULTS) } })
  }

  const runningCount = tasks.filter((tk) => tk.state === 'running').length

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
          <button onClick={() => setOpen(true)} className="btn-primary">
            {t('tasks.add')}
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
                <th className="text-right pr-4">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={3} />}
              {!isLoading && tasks.length === 0 && (
                <EmptyState message={t('tasks.empty')} />
              )}
              {tasks.map((tk) => (
                <tr key={tk.id}>
                  <td className="font-mono text-[12px] text-ink-primary font-medium">{tk.id}</td>
                  <td><StatusBadge status={tk.state} /></td>
                  <td>
                    <div className="flex items-center justify-end gap-1 pr-2">
                      {tk.state === 'stopped' ? (
                        <button
                          onClick={() => start.mutate(tk.id)}
                          disabled={start.isPending}
                          title="Start"
                          className="btn-icon text-success/60 hover:text-success"
                        >
                          <PlayIcon />
                        </button>
                      ) : (
                        <button
                          onClick={() => stop.mutate(tk.id)}
                          disabled={stop.isPending}
                          title="Stop"
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
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title={t('tasks.modal_title')}>
        <div className="space-y-4">
          <Field label={t('tasks.field.id')}>
            <Input
              placeholder="task-cam01-detect"
              value={form.id}
              onChange={(e) => set('id', e.target.value)}
            />
          </Field>

          <div className="grid grid-cols-2 gap-4">
            <Field label={t('tasks.field.source')} hint={t('tasks.field.source_hint')}>
              <Select
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
                type="number"
                min={1}
                max={120}
                value={form.sample_fps}
                onChange={(e) => set('sample_fps', +e.target.value)}
              />
            </Field>
            <Field label={t('tasks.field.sampling')}>
              <Select
                value={form.sampling_mode}
                onChange={(e) => set('sampling_mode', e.target.value as SamplingMode)}
              >
                <option value="frame_count">frame_count</option>
                <option value="time_based">time_based</option>
              </Select>
            </Field>
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={form.use_hwdec}
              onChange={(e) => set('use_hwdec', e.target.checked)}
              className="w-3.5 h-3.5 accent-accent"
            />
            <span className="text-[12px] text-ink-secondary">{t('tasks.field.hwdec')}</span>
          </label>

          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setOpen(false)} className="btn-ghost">{t('common.cancel')}</button>
            <button
              onClick={submit}
              disabled={!form.id || !form.source_id || !form.pipeline_id || add.isPending}
              className="btn-primary"
            >
              {add.isPending ? t('tasks.creating') : t('tasks.add')}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
