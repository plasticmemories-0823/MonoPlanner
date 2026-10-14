/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/((?!api/).*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          {
            key: "Content-Security-Policy",
            value:
              "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
          },
        ],
      },
    ];
  },
  images: { unoptimized: true }, // 使用随包提供的本地图标，不在安装目录建立图片缓存。
  experimental: {
    mcpServer: false,
    isrFlushToDisk: false, // 从构建产物提供内容，使用有界内存缓存，保持应用目录只读。
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
