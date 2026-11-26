import { useT } from '../../lib/i18n'
import { useTheme } from '../../lib/theme'

export function ThemeToggle() {
  const { t } = useT()
  const { theme, toggleTheme } = useTheme()
  const label = t(theme === 'dark' ? 'theme.switch_to_light' : 'theme.switch_to_dark')

  return (
    <button
      type="button"
      className="btn-ghost h-8 w-8 shrink-0 justify-center px-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      onClick={toggleTheme}
      aria-label={label}
      title={label}
    >
      <svg
        viewBox="0 0 24 24"
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {theme === 'dark' ? (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4m0-14.2-1.4 1.4M6.3 17.7l-1.4 1.4" />
          </>
        ) : (
          <path d="M20.5 14.5a8.5 8.5 0 0 1-11-11 8.5 8.5 0 1 0 11 11Z" />
        )}
      </svg>
    </button>
  )
}
