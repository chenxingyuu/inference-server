import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { api, ApiError } from '../lib/api'
import { sourceReasonKey, useT } from '../lib/i18n'
import type { SourceCreate, SourceInfo, PipelineCreate, ModelLoad, TaskCreate } from '../types'

// ─── Health ────────────────────────────────────────────────────────────────

export const useHealth = () =>
  useQuery({
    queryKey: ['health'],
    queryFn: api.health.check,
    refetchInterval: 10_000,
    retry: false,
  })

// ─── Sources ───────────────────────────────────────────────────────────────

export const useSources = () =>
  useQuery({
    queryKey: ['sources'],
    queryFn: api.sources.list,
    refetchInterval: 5_000,
  })

export const useAddSource = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: SourceCreate) => api.sources.add(body),
    onSuccess: (_, v) => {
      toast.success(`Source "${v.id}" added`)
      qc.invalidateQueries({ queryKey: ['sources'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useRemoveSource = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.sources.remove(id),
    onSuccess: (_, id) => {
      toast.success(`Source "${id}" removed`)
      qc.invalidateQueries({ queryKey: ['sources'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useProbeSource = (id: string) => {
  const qc = useQueryClient()
  const { t } = useT()
  return useMutation({
    mutationFn: () => api.sources.probe(id),
    onSuccess: async ({ availability, last_probe, reused_runtime }) => {
      await qc.cancelQueries({ queryKey: ['sources'] })
      qc.setQueryData<SourceInfo[]>(['sources'], (rows) => rows?.map((row) =>
        row.id === id ? { ...row, availability, last_probe } : row,
      ))
      if (reused_runtime) {
        toast.success(t('sources.probe_runtime', { id }))
      } else if (last_probe?.passed) {
        toast.success(t('sources.probe_passed', { id, ms: last_probe.elapsed_ms }))
      } else {
        const reason = t(sourceReasonKey(last_probe?.reason))
        toast.error(t('sources.probe_failed', { id, reason }))
      }
      void qc.invalidateQueries({ queryKey: ['sources'] })
    },
    onError: (error: Error) => {
      if (error instanceof ApiError && error.status === 401) {
        toast.error(t('sources.probe_unauthorized'))
      } else {
        const reason = error instanceof ApiError ? error.code : undefined
        toast.error(t(sourceReasonKey(reason, 'sources.reason.source_probe_unavailable')))
      }
    },
  })
}

// ─── Pipelines ─────────────────────────────────────────────────────────────

export const usePipelines = () =>
  useQuery({
    queryKey: ['pipelines'],
    queryFn: api.pipelines.list,
    refetchInterval: 30_000,
  })

export const useAddPipeline = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: PipelineCreate) => api.pipelines.add(body),
    onSuccess: (_, v) => {
      toast.success(`Pipeline "${v.id}" added`)
      qc.invalidateQueries({ queryKey: ['pipelines'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useUpdatePipeline = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: PipelineCreate }) => api.pipelines.update(id, body),
    onSuccess: (_, { id }) => {
      toast.success(`Pipeline "${id}" updated`)
      qc.invalidateQueries({ queryKey: ['pipelines'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useRemovePipeline = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.pipelines.remove(id),
    onSuccess: (_, id) => {
      toast.success(`Pipeline "${id}" removed`)
      qc.invalidateQueries({ queryKey: ['pipelines'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

// ─── Models ────────────────────────────────────────────────────────────────

export const useModels = () =>
  useQuery({
    queryKey: ['models'],
    queryFn: api.models.list,
    refetchInterval: 30_000,
  })

export const useModelDetail = (id: string) =>
  useQuery({
    queryKey: ['models', 'detail', id],
    queryFn: ({ signal }) => api.models.detail(id, signal),
    refetchInterval: 30_000,
    retry: (count, error) => count < 1 && !(error instanceof ApiError && error.status < 500),
  })

export const useRepositoryModels = () =>
  useQuery({
    queryKey: ['models', 'repository'],
    queryFn: api.models.listRepository,
    refetchInterval: 30_000,
  })

export const useLoadModel = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: ModelLoad) => api.models.load(body),
    onSuccess: (_, v) => {
      toast.success(`Model "${v.id}" loaded`)
      qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useLoadFromRepository = () => {
  const qc = useQueryClient()
  const { t } = useT()
  return useMutation({
    mutationFn: (id: string) => api.models.loadFromRepository(id),
    onSuccess: (_, id) => {
      toast.success(t('models.register_success', { id }))
      return qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useUnloadModel = () => {
  const qc = useQueryClient()
  const { t } = useT()
  return useMutation({
    mutationFn: (id: string) => api.models.unload(id),
    onSuccess: (_, id) => {
      toast.success(t('models.unregister_success', { id }))
      return qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e: Error) => {
      toast.error(e instanceof ApiError && e.code === 'in_use' ? t('models.unregister_in_use') : e.message)
    },
  })
}

// ─── Tasks ─────────────────────────────────────────────────────────────────

export const useTasks = () =>
  useQuery({
    queryKey: ['tasks'],
    queryFn: api.tasks.list,
    refetchInterval: (query) =>
      query.state.data?.some((task) => task.state === 'starting' || task.cleanup_pending) ? 1_000 : 5_000,
  })

export const useAddTask = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: TaskCreate) => api.tasks.add(body),
    onSuccess: (_, v) => {
      toast.success(`Task "${v.id}" created`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export const useUpdateTask = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: TaskCreate }) => api.tasks.update(id, body),
    onSuccess: (_, { id }) => {
      toast.success(`Task "${id}" updated`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export const useRemoveTask = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.tasks.remove(id),
    onSuccess: (_, id) => {
      toast.success(`Task "${id}" removed`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useStartTask = () => {
  const qc = useQueryClient()
  const { t } = useT()
  return useMutation({
    mutationFn: (id: string) => api.tasks.start(id),
    onSuccess: (_, id) => {
      toast.success(t('tasks.start_requested', { id }))
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useStopTask = () => {
  const qc = useQueryClient()
  const { t } = useT()
  return useMutation({
    mutationFn: (id: string) => api.tasks.stop(id),
    onSuccess: (_, id) => {
      toast.success(t('tasks.stop_requested', { id }))
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}
