import { useState } from 'react'
import { usePipelines, useAddPipeline, useRemovePipeline } from '../hooks/queries'
import { Modal } from '../components/ui/Modal'
import { Field, Input, Select } from '../components/ui/Field'
import { PageHeader, EmptyState, LoadingRows, DeleteButton } from '../components/layout/Layout'
import type { PipelineCreate, StageConfig, EdgeConfig, DropPolicy } from '../types'

const blankNode = (): StageConfig => ({ id: '', type: '' })
const blankEdge = (): EdgeConfig => ({ from: '', to: '', capacity: 32, drop_policy: 'drop_oldest' })

function NodeRow({
  node, idx, onChange, onRemove,
}: {
  node: StageConfig
  idx: number
  onChange: (n: StageConfig) => void
  onRemove: () => void
}) {
  return (
    <div className="flex gap-2 items-start">
      <div className="flex-none w-5 h-7 flex items-center justify-center text-[10px] font-mono text-ink-muted">
        {idx + 1}
      </div>
      <Input
        placeholder="node-id"
        value={node.id}
        onChange={(e) => onChange({ ...node, id: e.target.value })}
        className="flex-1"
      />
      <Input
        placeholder="type (e.g. yolo-infer)"
        value={node.type}
        onChange={(e) => onChange({ ...node, type: e.target.value })}
        className="flex-1"
      />
      <button onClick={onRemove} className="btn-icon text-danger/50 hover:text-danger mt-0.5">×</button>
    </div>
  )
}

function EdgeRow({
  edge, onChange, onRemove,
}: {
  edge: EdgeConfig
  onChange: (e: EdgeConfig) => void
  onRemove: () => void
}) {
  return (
    <div className="flex gap-2 items-start">
      <Input
        placeholder="from"
        value={edge.from}
        onChange={(e) => onChange({ ...edge, from: e.target.value })}
        className="flex-1"
      />
      <span className="text-ink-muted mt-2 flex-none">→</span>
      <Input
        placeholder="to"
        value={edge.to}
        onChange={(e) => onChange({ ...edge, to: e.target.value })}
        className="flex-1"
      />
      <Input
        type="number"
        placeholder="cap"
        value={edge.capacity ?? 32}
        onChange={(e) => onChange({ ...edge, capacity: +e.target.value })}
        className="w-20 flex-none"
      />
      <Select
        value={edge.drop_policy ?? 'drop_oldest'}
        onChange={(e) => onChange({ ...edge, drop_policy: e.target.value as DropPolicy })}
        className="flex-none w-32"
      >
        <option value="block">block</option>
        <option value="drop_oldest">drop_oldest</option>
        <option value="drop_newest">drop_newest</option>
      </Select>
      <button onClick={onRemove} className="btn-icon text-danger/50 hover:text-danger mt-0.5">×</button>
    </div>
  )
}

export function PipelinesPage() {
  const { data: pipelines = [], isLoading } = usePipelines()
  const add = useAddPipeline()
  const remove = useRemovePipeline()

  const [open, setOpen] = useState(false)
  const [pipeId, setPipeId] = useState('')
  const [nodes, setNodes] = useState<StageConfig[]>([blankNode()])
  const [edges, setEdges] = useState<EdgeConfig[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)

  const updateNode = (i: number, n: StageConfig) =>
    setNodes((ns) => ns.map((x, j) => (j === i ? n : x)))
  const updateEdge = (i: number, e: EdgeConfig) =>
    setEdges((es) => es.map((x, j) => (j === i ? e : x)))

  const submit = () => {
    if (!pipeId || nodes.some((n) => !n.id || !n.type)) return
    const body: PipelineCreate = {
      id: pipeId,
      nodes,
      edges: edges.filter((e) => e.from && e.to),
    }
    add.mutate(body, {
      onSuccess: () => {
        setOpen(false)
        setPipeId('')
        setNodes([blankNode()])
        setEdges([])
      },
    })
  }

  return (
    <div>
      <PageHeader
        title="Pipelines"
        subtitle="Processing node graphs"
        action={
          <button onClick={() => setOpen(true)} className="btn-primary">
            + Add Pipeline
          </button>
        }
      />

      <div className="section">
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Nodes</th>
                <th>Edges</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {isLoading && <LoadingRows cols={4} />}
              {!isLoading && pipelines.length === 0 && (
                <EmptyState message="No pipelines defined. Add one to connect sources to inference." />
              )}
              {pipelines.map((p) => (
                <>
                  <tr
                    key={p.id}
                    className="cursor-pointer"
                    onClick={() => setExpanded(expanded === p.id ? null : p.id)}
                  >
                    <td className="font-mono text-[12px] text-ink-primary font-medium">
                      <span className="mr-2 text-ink-muted">{expanded === p.id ? '▼' : '▶'}</span>
                      {p.id}
                    </td>
                    <td className="font-mono text-[12px] text-ink-secondary">{p.nodes.length}</td>
                    <td className="font-mono text-[12px] text-ink-secondary">{p.edges.length}</td>
                    <td className="text-right pr-2" onClick={(e) => e.stopPropagation()}>
                      <DeleteButton
                        loading={remove.isPending}
                        onClick={() => remove.mutate(p.id)}
                      />
                    </td>
                  </tr>
                  {expanded === p.id && (
                    <tr key={`${p.id}-detail`}>
                      <td colSpan={4} className="bg-bg-overlay px-6 py-3">
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <div className="label mb-2">Nodes</div>
                            <div className="space-y-1">
                              {p.nodes.map((n) => (
                                <div key={n.id} className="flex gap-2 items-center text-[12px]">
                                  <span className="font-mono text-accent w-28 truncate">{n.id}</span>
                                  <span className="text-ink-muted">→</span>
                                  <span className="font-mono text-ink-secondary">{n.type}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          {p.edges.length > 0 && (
                            <div>
                              <div className="label mb-2">Edges</div>
                              <div className="space-y-1">
                                {p.edges.map((e, i) => (
                                  <div key={i} className="flex gap-2 items-center text-[12px]">
                                    <span className="font-mono text-ink-primary">{e.from}</span>
                                    <span className="text-ink-muted">→</span>
                                    <span className="font-mono text-ink-primary">{e.to}</span>
                                    <span className="text-ink-muted ml-2 text-[10px]">
                                      cap:{e.capacity} · {e.drop_policy}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title="Add Pipeline" width="max-w-2xl">
        <div className="space-y-5">
          <Field label="Pipeline ID">
            <Input
              placeholder="detection-pipeline"
              value={pipeId}
              onChange={(e) => setPipeId(e.target.value)}
            />
          </Field>

          {/* Nodes */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="label">Nodes</span>
              <button
                onClick={() => setNodes((ns) => [...ns, blankNode()])}
                className="text-[11px] text-accent hover:underline"
              >
                + Add node
              </button>
            </div>
            <div className="space-y-2">
              <div className="flex gap-2 text-[10px] text-ink-muted mb-1 pl-7">
                <span className="flex-1">Node ID</span>
                <span className="flex-1">Type</span>
                <span className="w-4" />
              </div>
              {nodes.map((n, i) => (
                <NodeRow
                  key={i}
                  idx={i}
                  node={n}
                  onChange={(v) => updateNode(i, v)}
                  onRemove={() => setNodes((ns) => ns.filter((_, j) => j !== i))}
                />
              ))}
            </div>
          </div>

          {/* Edges */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="label">Edges (optional)</span>
              <button
                onClick={() => setEdges((es) => [...es, blankEdge()])}
                className="text-[11px] text-accent hover:underline"
              >
                + Add edge
              </button>
            </div>
            {edges.length > 0 && (
              <div className="space-y-2">
                <div className="flex gap-2 text-[10px] text-ink-muted mb-1">
                  <span className="flex-1">From</span>
                  <span className="w-4" />
                  <span className="flex-1">To</span>
                  <span className="w-20">Capacity</span>
                  <span className="w-32">Drop Policy</span>
                  <span className="w-4" />
                </div>
                {edges.map((e, i) => (
                  <EdgeRow
                    key={i}
                    edge={e}
                    onChange={(v) => updateEdge(i, v)}
                    onRemove={() => setEdges((es) => es.filter((_, j) => j !== i))}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setOpen(false)} className="btn-ghost">Cancel</button>
            <button
              onClick={submit}
              disabled={!pipeId || nodes.some((n) => !n.id || !n.type) || add.isPending}
              className="btn-primary"
            >
              {add.isPending ? 'Adding…' : 'Add Pipeline'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
