import { getNodeTypeDef } from './nodeTypes'

export type HandleMode = 'none' | 'single' | 'multi'

export interface HandleConfig {
  inputs: HandleMode
  outputs: HandleMode
}

export function getHandleConfig(stageType: string): HandleConfig {
  if (stageType.startsWith('source.')) {
    return { inputs: 'none', outputs: 'single' }
  }
  if (stageType.startsWith('sink.')) {
    return { inputs: 'single', outputs: 'none' }
  }
  if (stageType === 'join.byFrameId') {
    return { inputs: 'multi', outputs: 'single' }
  }
  return { inputs: 'single', outputs: 'single' }
}

export function getCategoryColor(stageType: string): string {
  const cat = getNodeTypeDef(stageType)?.category ?? 'infer'
  const colors: Record<string, string> = {
    source: '#22c55e',
    decode: '#3b82f6',
    infer: '#a855f7',
    postprocess: '#f59e0b',
    track: '#06b6d4',
    join: '#ec4899',
    archive: '#64748b',
    sink: '#ef4444',
  }
  return colors[cat] ?? '#94a3b8'
}
