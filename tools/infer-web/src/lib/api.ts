import type {
  SourceInfo, SourceCreate, PipelineInfo, PipelineCreate,
  ModelInfo, ModelLoad, RepositoryModel, TaskInfo, TaskCreate, OkResponse, ArchiveAnalysis,
} from '../types'

export const serverUrl = {
  get: (): string => localStorage.getItem('infer_server_url') ?? window.location.origin,
  set: (url: string) => localStorage.setItem('infer_server_url', url.replace(/\/$/, '')),
  clear: () => localStorage.removeItem('infer_server_url'),
}

export const auth = {
  get: (): string => localStorage.getItem('infer_api_key') ?? '',
  set: (key: string) => localStorage.setItem('infer_api_key', key),
  clear: () => localStorage.removeItem('infer_api_key'),
}

function buildApiUrl(path: string): string {
  const base = serverUrl.get()
  const apiPath = path.startsWith('/') ? path : `/${path}`
  const normalizedBase = base.replace(/\/$/, '')
  const origin = window.location.origin.replace(/\/$/, '')
  const usePrefixedApi = normalizedBase === origin
  return usePrefixedApi ? `${normalizedBase}/api${apiPath}` : `${normalizedBase}${apiPath}`
}

class ApiError extends Error {
  constructor(public status: number, message: string, public code?: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const key = auth.get()
  const res = await fetch(buildApiUrl(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(key ? { Authorization: `Bearer ${key}` } : {}),
      ...init?.headers,
    },
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.message ?? res.statusText, body.code)
  }

  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

const json = (body: unknown) => JSON.stringify(body)

export type HealthStatus = 'ok' | 'engine_down' | 'unreachable'

export const api = {
  health: {
    check: async (): Promise<{ status: HealthStatus }> => {
      let res: Response
      try {
        res = await fetch(buildApiUrl('/healthz'))
      } catch {
        return { status: 'unreachable' }
      }
      if (!res.ok) return { status: 'engine_down' }
      return { status: 'ok' }
    },
  },

  sources: {
    list: () => req<SourceInfo[]>('/sources'),
    add: (body: SourceCreate) => req<OkResponse>('/sources', { method: 'POST', body: json(body) }),
    remove: (id: string) => req<OkResponse>(`/sources/${id}`, { method: 'DELETE' }),
  },

  pipelines: {
    list: () => req<PipelineInfo[]>('/pipelines'),
    add: (body: PipelineCreate) => req<OkResponse>('/pipelines', { method: 'POST', body: json(body) }),
    update: (id: string, body: PipelineCreate) => req<OkResponse>(`/pipelines/${id}`, { method: 'PUT', body: json(body) }),
    remove: (id: string) => req<OkResponse>(`/pipelines/${id}`, { method: 'DELETE' }),
  },

  models: {
    list: () => req<ModelInfo[]>('/models'),
    load: (body: ModelLoad) => req<OkResponse>('/models', { method: 'POST', body: json(body) }),
    unload: (id: string) => req<OkResponse>(`/models/${id}`, { method: 'DELETE' }),
    listRepository: () => req<RepositoryModel[]>('/models/repository'),
    loadFromRepository: (id: string) =>
      req<OkResponse>(`/models/repository/${id}/load`, { method: 'POST' }),
    upload: (
      formData: FormData,
      onProgress: (pct: number, loaded: number, total: number) => void,
      signal?: AbortSignal,
    ): Promise<OkResponse> =>
      new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', buildApiUrl('/models/upload'))
        const key = auth.get()
        if (key) xhr.setRequestHeader('Authorization', `Bearer ${key}`)
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100), e.loaded, e.total)
        }
        xhr.onload = () => {
          const body = (() => { try { return JSON.parse(xhr.responseText) } catch { return {} } })()
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(body as OkResponse)
          } else {
            reject(new ApiError(xhr.status, body.message ?? xhr.statusText, body.code))
          }
        }
        xhr.onerror = () => reject(new Error('Network error'))
        if (signal) signal.addEventListener('abort', () => xhr.abort())
        xhr.send(formData)
      }),

    analyze: (
      formData: FormData,
      onProgress: (pct: number, loaded: number, total: number) => void,
      signal?: AbortSignal,
    ): Promise<ArchiveAnalysis> =>
      new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', buildApiUrl('/models/upload/analyze'))
        const key = auth.get()
        if (key) xhr.setRequestHeader('Authorization', `Bearer ${key}`)
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100), e.loaded, e.total)
        }
        xhr.onload = () => {
          const body = (() => { try { return JSON.parse(xhr.responseText) } catch { return {} } })()
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(body as ArchiveAnalysis)
          } else {
            reject(new ApiError(xhr.status, body.message ?? xhr.statusText, body.code))
          }
        }
        xhr.onerror = () => reject(new Error('Network error'))
        if (signal) signal.addEventListener('abort', () => xhr.abort())
        xhr.send(formData)
      }),
  },

  tasks: {
    list: () => req<TaskInfo[]>('/tasks'),
    add: (body: TaskCreate) => req<OkResponse>('/tasks', { method: 'POST', body: json(body) }),
    update: (id: string, body: TaskCreate) => req<OkResponse>(`/tasks/${id}`, { method: 'PUT', body: json(body) }),
    remove: (id: string) => req<OkResponse>(`/tasks/${id}`, { method: 'DELETE' }),
    start: (id: string) => req<OkResponse>(`/tasks/${id}/start`, { method: 'POST' }),
    stop: (id: string) => req<OkResponse>(`/tasks/${id}/stop`, { method: 'POST' }),
  },
}

export { ApiError }
