/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: { unoptimized: true }, // Bundled local logo; no installation-directory image cache.
  experimental: {
    isrFlushToDisk: false, // Serve build seeds with bounded memory caching; keep the application read-only.
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
