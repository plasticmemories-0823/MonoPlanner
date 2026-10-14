/** 浏览器不选择后端主机，也不保留服务凭据。 */
export function localApiUrl(): string {
  return typeof window === "undefined"
    ? "/api"
    : `${window.location.origin}/api`;
}
