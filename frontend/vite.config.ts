import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 127.0.0.1, not localhost: on Windows ::1:8000 can be taken by wslrelay instead of Docker.
const apiProxyTarget =
  process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    // Bind-mounted Windows files don't emit inotify events inside the container.
    watch: { usePolling: true },
    proxy: {
      "/api": apiProxyTarget,
    },
  },
});
