import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { I18nProvider } from './lib/i18n'
import { ThemeProvider } from './lib/theme'
import App from './App'
import './styles/global.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5_000 },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ThemeProvider>
            <App />
          </ThemeProvider>
          <Toaster
            position="bottom-right"
            toastOptions={{
              style: {
                background: 'rgb(var(--color-bg-overlay))',
                color: 'rgb(var(--color-ink-primary))',
                border: '1px solid rgb(var(--color-border))',
                fontFamily: 'Inter, system-ui, sans-serif',
                fontSize: '13px',
                borderRadius: '4px',
              },
              success: {
                iconTheme: { primary: 'rgb(var(--color-success))', secondary: 'rgb(var(--color-bg-overlay))' },
              },
              error: {
                iconTheme: { primary: 'rgb(var(--color-danger))', secondary: 'rgb(var(--color-bg-overlay))' },
              },
            }}
          />
        </BrowserRouter>
      </QueryClientProvider>
    </I18nProvider>
  </React.StrictMode>
)
