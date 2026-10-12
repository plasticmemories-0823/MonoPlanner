import { Client } from "@langchain/langgraph-sdk";
import { localApiUrl } from "@/lib/local-service";

export function createClient(apiUrl = localApiUrl()) {
  if (apiUrl !== localApiUrl()) throw new Error("Only the local MonoPlanner service is supported");
  return new Client({ apiUrl });
}
