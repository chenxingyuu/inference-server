import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://localhost:8080'
  const proxy = {
    '/api': {
      target,
      changeOrigin: true,
      rewrite: (path: string) => path.replace(/^\/api/, ''),
    },
  }
  return {
    plugins: [react()],
    server: { port: 5173, proxy },
  }
})
