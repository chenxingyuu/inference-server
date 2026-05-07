import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import App from './App'
import './styles/global.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5_000 },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#19212e',
              color: '#dde5f0',
              border: '1px solid #1e2d3d',
              fontFamily: 'Inter, system-ui, sans-serif',
              fontSize: '13px',
              borderRadius: '4px',
            },
            success: { iconTheme: { primary: '#34d399', secondary: '#19212e' } },
            error: { iconTheme: { primary: '#f87171', secondary: '#19212e' } },
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
)
