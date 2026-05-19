import type { Key } from './i18n'

export function nodeCategoryKey(category: string): Key {
  return `pipelines.node.category.${category}` as Key
}

export function nodeTypeKey(type: string): Key {
  return `pipelines.node.type.${type}` as Key
}

type TFn = (key: Key) => string

function translateOrFallback(t: TFn, key: Key, fallback: string): string {
  const out = t(key)
  return out === key ? fallback : out
}

export function nodeCategoryLabel(t: TFn, category: string): string {
  if (!category) return t('pipelines.editor.inspector_node')
  return translateOrFallback(t, nodeCategoryKey(category), category)
}

export function nodeTypeLabel(t: TFn, type: string): string {
  if (!type || type === '—') return '—'
  return translateOrFallback(t, nodeTypeKey(type), type)
}
