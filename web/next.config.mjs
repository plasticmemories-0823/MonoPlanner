/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  poweredByHeader: false,
  async headers() {
    return [{ source: "/((?!api/).*)", headers: [
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Referrer-Policy", value: "no-referrer" },
      { key: "Content-Security-Policy", value: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" },
    ] }];
  },
  images: { unoptimized: true }, // Bundled local logo; no installation-directory image cache.
  experimental: {
    mcpServer: false,
    isrFlushToDisk: false, // Serve build seeds with bounded memory caching; keep the application read-only.
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
