import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: {
    // The app can stay open while the Docker image is rebuilt. Stable JS
    // chunk names keep an already loaded entrypoint from requesting hashed
    // files that disappeared from the new image.
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: (assetInfo) => assetInfo.name?.endsWith(".css")
          ? "assets/[name][extname]"
          : "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": api,
      "/process_text_stream": api,
      "/process_text": api,
      "/graph_viz": api,
      "/graph_explore": api,
      "/ui_config": api,
      "/health": api,
      "/stt": api,
      "/login": api,
      "/logout": api,
      "^/ui/[^/]+/login": api,
      "^/ui/[^/]+/logout": api,
    },
  },
});
