import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { api } from '../lib/api'
import type { SourceCreate, PipelineCreate, ModelLoad, TaskCreate } from '../types'

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
  return useMutation({
    mutationFn: (id: string) => api.models.loadFromRepository(id),
    onSuccess: (_, id) => {
      toast.success(`Model "${id}" loaded`)
      qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useUnloadModel = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.models.unload(id),
    onSuccess: (_, id) => {
      toast.success(`Model "${id}" unloaded`)
      qc.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

// ─── Tasks ─────────────────────────────────────────────────────────────────

export const useTasks = () =>
  useQuery({
    queryKey: ['tasks'],
    queryFn: api.tasks.list,
    refetchInterval: 5_000,
  })

export const useAddTask = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: TaskCreate) => api.tasks.add(body),
    onSuccess: (_, v) => {
      toast.success(`Task "${v.id}" created`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
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
    onError: (e: Error) => toast.error(e.message),
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
  return useMutation({
    mutationFn: (id: string) => api.tasks.start(id),
    onSuccess: (_, id) => {
      toast.success(`Task "${id}" started`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}

export const useStopTask = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.tasks.stop(id),
    onSuccess: (_, id) => {
      toast.success(`Task "${id}" stopped`)
      qc.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (e: Error) => toast.error(e.message),
  })
}
