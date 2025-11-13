import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const serverConfig = {
  port: 5173,
  host: '127.0.0.1',
  strictPort: true,
  allowedHosts: true,
}

console.info('[Vite config] server settings:', serverConfig)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: serverConfig,
})
