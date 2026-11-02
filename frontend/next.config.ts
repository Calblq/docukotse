import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server for the Docker image (set in frontend/Dockerfile). Off by default
  // because standalone output can need symlink privileges on Windows.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  agentRules: false, // don't generate AGENTS.md when `next dev` runs
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
