import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://localhost:5000',
        ws: true,
      },
      '/rtc': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/live': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ready': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
