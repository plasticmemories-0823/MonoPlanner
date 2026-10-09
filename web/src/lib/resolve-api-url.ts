export function resolveApiUrl(queryApiUrl: string, envApiUrl?: string): string {
  return queryApiUrl || envApiUrl || "";
}
