import { useState } from 'react'
import { useModels, useLoadModel, useUnloadModel } from '../hooks/queries'
import { Modal } from '../components/ui/Modal'
import { Field, Input, Select } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import { useT } from '../lib/i18n'
import type { ModelLoad, BackendType, ModelVersion, ModelType } from '../types'

const DEFAULTS: ModelLoad = {
  id: '',
  backend: 'cpu',
  version: 'v8',
  model_type: 'detector',
  onnx_path: '',
  batch_size: 1,
  conf_thresh: 0.25,
  nms_thresh: 0.45,
  num_classes: 80,
  instance_count: 1,
}

function Tag({ children }: { children: string }) {
  return (
    <span className="inline-block font-mono text-[10px] px-1.5 py-0.5 bg-bg-elevated border border-border rounded text-ink-secondary">
      {children}
    </span>
  )
}

export function ModelsPage() {
  const { data: models = [], isLoading } = useModels()
  const load = useLoadModel()
  const unload = useUnloadModel()
  const { t } = useT()

  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<ModelLoad>(DEFAULTS)

  const set = <K extends keyof ModelLoad>(k: K, v: ModelLoad[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const submit = () => {
    if (!form.id) return
    load.mutate(form, { onSuccess: () => { setOpen(false); setForm(DEFAULTS) } })
  }

  return (
    <div>
      <PageHeader
        title={t('models.title')}
        subtitle={t('models.subtitle')}
        action={
          <button onClick={() => setOpen(true)} className="btn-primary">
            {t('models.load')}
          </button>
        }
      />

      <div className="section">
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('common.id')}</th>
                <th>{t('models.col.backend')}</th>
                <th>{t('models.col.version')}</th>
                <th>{t('models.col.input_shape')}</th>
                <th>{t('models.col.batch')}</th>
                <th>{t('models.col.instances')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={7} />}
              {!isLoading && models.length === 0 && (
                <EmptyState message={t('models.empty')} />
              )}
              {models.map((m) => (
                <tr key={m.id}>
                  <td className="font-mono text-[12px] text-ink-primary font-medium">{m.id}</td>
                  <td><Tag>{m.backend || '—'}</Tag></td>
                  <td><Tag>{m.version || '—'}</Tag></td>
                  <td className="font-mono text-[12px] text-ink-secondary">{m.input_shape || '—'}</td>
                  <td className="font-mono text-[12px] text-ink-secondary">{m.batch_size}</td>
                  <td className="font-mono text-[12px] text-ink-secondary">{m.instance_count}</td>
                  <td className="text-right pr-2">
                    <DeleteButton
                      loading={unload.isPending}
                      onClick={() => unload.mutate(m.id)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title={t('models.modal_title')} width="max-w-2xl">
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <Field label={t('models.field.id')}>
              <Input placeholder="yolo-v8-det" value={form.id} onChange={(e) => set('id', e.target.value)} />
            </Field>
            <Field label={t('models.field.backend')}>
              <Select value={form.backend} onChange={(e) => set('backend', e.target.value as BackendType)}>
                <option value="cpu">cpu</option>
                <option value="trt">trt</option>
                <option value="ascend">ascend</option>
              </Select>
            </Field>
            <Field label={t('models.field.version')}>
              <Select value={form.version} onChange={(e) => set('version', e.target.value as ModelVersion)}>
                <option value="v5">v5</option>
                <option value="v8">v8</option>
                <option value="v11">v11</option>
                <option value="v26">v26</option>
              </Select>
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Field label={t('models.field.model_type')}>
              <Select value={form.model_type} onChange={(e) => set('model_type', e.target.value as ModelType)}>
                <option value="detector">detector</option>
                <option value="classifier">classifier</option>
              </Select>
            </Field>
            <Field label={t('models.field.instances')}>
              <Input type="number" min={1} value={form.instance_count}
                onChange={(e) => set('instance_count', +e.target.value)} />
            </Field>
          </div>

          <Field label={t('models.field.onnx_path')} hint={t('models.field.onnx_path_hint')}>
            <Input placeholder="/models/yolov8n.onnx" value={form.onnx_path ?? ''}
              onChange={(e) => set('onnx_path', e.target.value)} />
          </Field>

          <Field label={t('models.field.engine_path')} hint={t('models.field.engine_path_hint')}>
            <Input placeholder="/models/yolov8n.engine" value={form.engine_path ?? ''}
              onChange={(e) => set('engine_path', e.target.value)} />
          </Field>

          <div className="grid grid-cols-4 gap-4">
            <Field label={t('models.field.batch_size')}>
              <Input type="number" min={1} value={form.batch_size}
                onChange={(e) => set('batch_size', +e.target.value)} />
            </Field>
            <Field label={t('models.field.num_classes')}>
              <Input type="number" min={1} value={form.num_classes}
                onChange={(e) => set('num_classes', +e.target.value)} />
            </Field>
            <Field label={t('models.field.conf_thresh')}>
              <Input type="number" step={0.01} min={0} max={1} value={form.conf_thresh}
                onChange={(e) => set('conf_thresh', +e.target.value)} />
            </Field>
            <Field label={t('models.field.nms_thresh')}>
              <Input type="number" step={0.01} min={0} max={1} value={form.nms_thresh}
                onChange={(e) => set('nms_thresh', +e.target.value)} />
            </Field>
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setOpen(false)} className="btn-ghost">{t('common.cancel')}</button>
            <button
              onClick={submit}
              disabled={!form.id || load.isPending}
              className="btn-primary"
            >
              {load.isPending ? t('models.loading') : t('models.load')}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
