import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  server: {
    port: 5173,
    proxy: {
      "/process_text_stream": api,
      "/process_text": api,
      "/graph_viz": api,
      "/graph_explore": api,
      "/ui_config": api,
      "/health": api,
      "/clear_history": api,
      "/stt": api,
      "/login": api,
      "/logout": api,
    },
  },
});
