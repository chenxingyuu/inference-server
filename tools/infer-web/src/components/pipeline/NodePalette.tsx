import { NODE_CATEGORIES, NODE_TYPE_DEFS } from '../../lib/nodeTypes'
import { nodeCategoryKey, nodeTypeLabel } from '../../lib/nodeTypeI18n'
import { getCategoryColor } from '../../lib/pipelineHandles'
import { useT } from '../../lib/i18n'

const DRAG_TYPE = 'application/infer-pipeline-node-type'

export function getDragNodeType(dataTransfer: DataTransfer): string | null {
  return dataTransfer.getData(DRAG_TYPE) || null
}

export function setDragNodeType(dataTransfer: DataTransfer, nodeType: string): void {
  dataTransfer.setData(DRAG_TYPE, nodeType)
  dataTransfer.effectAllowed = 'move'
}

interface NodePaletteProps {
  onAddType: (nodeType: string) => void
}

export function NodePalette({ onAddType }: NodePaletteProps) {
  const { t } = useT()

  return (
    <aside className="w-52 flex-shrink-0 border-r border-border bg-bg-surface overflow-y-auto">
      <div className="flex items-center h-9 px-3 border-b border-border shrink-0">
        <span className="text-[11px] font-medium uppercase tracking-wide text-ink-muted leading-none">
          {t('pipelines.editor.palette')}
        </span>
      </div>
      <div className="p-2 pt-2.5 space-y-3">
        {NODE_CATEGORIES.map((cat) => {
          const items = NODE_TYPE_DEFS.filter((d) => d.category === cat)
          if (items.length === 0) return null
          return (
            <div key={cat}>
              <div className="text-[10px] font-semibold uppercase tracking-wide text-ink-muted px-1 mb-1 leading-none">
                {t(nodeCategoryKey(cat))}
              </div>
              <div className="space-y-1">
                {items.map((d) => (
                  <button
                    key={d.type}
                    type="button"
                    draggable
                    onDragStart={(e) => setDragNodeType(e.dataTransfer, d.type)}
                    onClick={() => onAddType(d.type)}
                    className="w-full text-left px-2 py-1.5 rounded border border-border/80 bg-bg-overlay hover:border-accent/50 hover:bg-bg-elevated transition-colors"
                    style={{ borderLeftWidth: 3, borderLeftColor: getCategoryColor(d.type) }}
                    title={d.type}
                  >
                    <span className="text-[11px] text-ink-primary block truncate leading-snug">
                      {nodeTypeLabel(t, d.type)}
                    </span>
                    <span className="font-mono text-[9px] text-ink-muted block truncate leading-snug mt-0.5">
                      {d.type}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
