import { configureStore } from '@reduxjs/toolkit'
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'

export const DEFAULT_API_URL = 'http://localhost:8000/api/v1'

export const RAG_STRATEGIES = ['naive', 'advanced', 'corrective', 'self_rag', 'hyde', 'multi_query', 'graph_rag'] as const
export const AGENT_TYPES = ['react', 'plan_execute', 'reflection', 'supervisor', 'rag_agent', 'multi_agent'] as const
export const CHUNK_STRATEGIES = ['recursive', 'semantic', 'parent_child', 'sentence_window'] as const

export type ChunkStrategy = (typeof CHUNK_STRATEGIES)[number]
export type ConnectionOptions = { baseUrl: string; tenantId?: string; apiKey?: string }
export type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }
export type CollectionSummary = { name: string; chunk_count: number; document_count: number; sources?: string[] }
export type DocumentSummary = { document_id: string; source: string; chunk_count: number; metadata: Record<string, unknown> }
export type DocumentChunk = { id: string; content: string; metadata: Record<string, unknown> }
export type RagSource = { id?: string; content?: string; score?: number; [key: string]: unknown }
export type RagQueryResponse = { answer: string; strategy: string; sources: RagSource[]; metadata: Record<string, unknown> }
export type RagCompareResult = { strategy: string; answer: string; sources: RagSource[]; metadata: Record<string, unknown>; latency_ms: number; source_count: number; status: 'success' | 'error'; error?: string | null }
export type RagCompareResponse = { query: string; collection_name: string; results: RagCompareResult[] }
export type AgentResult = { result: string; agent_type: string; steps: Record<string, unknown>[]; tool_calls: Record<string, unknown>[]; iterations: number }
export type QueryLog = { id: number; created_at: string; query: string; strategy: string; collection_name: string; latency_ms: number; status: string; source_count: number; answer_preview: string; error?: string | null }
export type FeedbackItem = { id: number; created_at: string; query_log_id?: number | null; rating: string; comment: string; query?: string | null; strategy?: string | null; collection_name: string }

type IngestResult = { document_id: string; chunks_created: number; collection_name: string; status?: string }
type DeleteDocumentResult = { deleted: string; chunks_deleted: number }
type DeleteCollectionResult = { deleted: string }
type ChatResult = { message: ChatMessage; session_id: string; sources: RagSource[] }
type StreamChatResult = { text: string; session_id: string }

type BaseArg = ConnectionOptions

type DocumentsArg = BaseArg & { collectionName: string }
type ChunksArg = DocumentsArg & { documentId: string; limit?: number }
type ListArg = BaseArg & { limit?: number }
type IngestArg = BaseArg & { file: File; collectionName: string; chunkStrategy: ChunkStrategy | string; chunkSize: number; chunkOverlap: number }
type DeleteArg = DocumentsArg & { documentId: string }
type RagQueryArg = BaseArg & { query: string; strategy: string; collectionName: string; topK: number; includeSources: boolean; rerank: boolean }
type RagCompareArg = BaseArg & { query: string; strategies: string[]; collectionName: string; topK: number }
type AgentArg = BaseArg & { task: string; agentType: string; maxIterations: number; tools?: string[]; sessionId?: string | null; useMemory?: boolean }
type FeedbackArg = BaseArg & { rating: 'up' | 'down'; comment: string; query?: string; strategy?: string; collectionName: string; queryLogId?: number }
type ChatArg = BaseArg & { messages: ChatMessage[]; sessionId?: string | null; ragStrategy?: string | null; collectionName: string }

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/$/, '')
}

function buildUrl(baseUrl: string, path: string): string {
  return `${normalizeBaseUrl(baseUrl)}${path.startsWith('/') ? path : `/${path}`}`
}

function buildServiceUrl(baseUrl: string, path: string): string {
  const normalized = normalizeBaseUrl(baseUrl).replace(/\/api(?:\/v\d+)?$/, '')
  return `${normalized}${path.startsWith('/') ? path : `/${path}`}`
}

function buildHeaders(connection: Partial<ConnectionOptions>): HeadersInit {
  const headers: Record<string, string> = {}
  if (connection.tenantId?.trim()) {
    headers['X-Tenant-ID'] = connection.tenantId.trim()
  }
  if (connection.apiKey?.trim()) {
    headers['X-API-Key'] = connection.apiKey.trim()
  }
  return headers
}

function buildQuery(values: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined) {
      params.set(key, String(value))
    }
  }
  const query = params.toString()
  return query ? `?${query}` : ''
}

async function requestJson<T>(baseUrl: string, path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers)
  let body = init.body

  if (init.json !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(init.json)
  }

  const response = await fetch(buildUrl(baseUrl, path), { ...init, headers, body })

  if (!response.ok) {
    throw new Error(await response.text())
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

async function readSseText(response: Response): Promise<string> {
  if (!response.body) {
    return ''
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let fullText = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const event of events) {
      for (const line of event.split('\n')) {
        if (!line.startsWith('data: ')) {
          continue
        }
        const token = line.slice(6)
        if (token === '[DONE]') {
          return fullText
        }
        fullText += token
      }
    }
  }

  for (const line of buffer.split('\n')) {
    if (line.startsWith('data: ')) {
      const token = line.slice(6)
      if (token !== '[DONE]') {
        fullText += token
      }
    }
  }

  return fullText
}

export const ragApi = createApi({
  reducerPath: 'ragApi',
  baseQuery: fetchBaseQuery({ baseUrl: '' }),
  tagTypes: ['Health', 'Collections', 'Documents', 'Chunks', 'Rag', 'Agents', 'Analytics'],
  endpoints: (builder) => ({
    checkHealth: builder.query<boolean, BaseArg>({
      async queryFn(arg) {
        const response = await fetch(buildServiceUrl(arg.baseUrl, '/health'), { headers: buildHeaders(arg) })
        return { data: response.ok }
      },
      providesTags: ['Health'],
    }),
    getCollections: builder.query<CollectionSummary[], BaseArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<CollectionSummary[]>(arg.baseUrl, '/documents/collections', { headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: ['Collections'],
    }),
    getDocuments: builder.query<DocumentSummary[], DocumentsArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<DocumentSummary[]>(arg.baseUrl, `/documents/${encodeURIComponent(arg.collectionName)}`, { headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: (_result, _error, arg) => [{ type: 'Documents', id: arg.collectionName }],
    }),
    getDocumentChunks: builder.query<DocumentChunk[], ChunksArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<DocumentChunk[]>(arg.baseUrl, `/documents/${encodeURIComponent(arg.collectionName)}/documents/${encodeURIComponent(arg.documentId)}/chunks${buildQuery({ limit: arg.limit ?? 100 })}`, { headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: (_result, _error, arg) => [{ type: 'Chunks', id: `${arg.collectionName}:${arg.documentId}` }],
    }),
    getRagStrategies: builder.query<string[], BaseArg>({
      async queryFn(arg) {
        try {
          const response = await requestJson<{ strategies: Array<{ name?: string }> }>(arg.baseUrl, '/rag/strategies', { headers: buildHeaders(arg) })
          return { data: response.strategies.flatMap((item) => (item.name ? [item.name] : [])) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: ['Rag'],
    }),
    getAgentTypes: builder.query<string[], BaseArg>({
      async queryFn(arg) {
        try {
          const response = await requestJson<{ agent_types: Array<{ name?: string }> }>(arg.baseUrl, '/agents/types', { headers: buildHeaders(arg) })
          return { data: response.agent_types.flatMap((item) => (item.name ? [item.name] : [])) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: ['Agents'],
    }),
    getQueryLogs: builder.query<QueryLog[], ListArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<QueryLog[]>(arg.baseUrl, `/analytics/queries${buildQuery({ limit: arg.limit ?? 100 })}`, { headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: ['Analytics'],
    }),
    getFeedback: builder.query<FeedbackItem[], ListArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<FeedbackItem[]>(arg.baseUrl, `/analytics/feedback${buildQuery({ limit: arg.limit ?? 100 })}`, { headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      providesTags: ['Analytics'],
    }),
    ingestDocument: builder.mutation<IngestResult, IngestArg>({
      async queryFn(arg) {
        try {
          const formData = new FormData()
          formData.set('file', arg.file)
          formData.set('collection_name', arg.collectionName)
          formData.set('chunk_strategy', arg.chunkStrategy)
          formData.set('chunk_size', String(arg.chunkSize))
          formData.set('chunk_overlap', String(arg.chunkOverlap))
          return { data: await requestJson<IngestResult>(arg.baseUrl, '/documents/ingest', { method: 'POST', body: formData, headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Collections', 'Documents', 'Chunks', 'Analytics'],
    }),
    deleteDocument: builder.mutation<DeleteDocumentResult, DeleteArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<DeleteDocumentResult>(arg.baseUrl, `/documents/${encodeURIComponent(arg.collectionName)}/documents/${encodeURIComponent(arg.documentId)}`, { method: 'DELETE', headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Collections', 'Documents', 'Chunks', 'Analytics'],
    }),
    deleteCollection: builder.mutation<DeleteCollectionResult, DocumentsArg>({
      async queryFn(arg) {
        try {
          return { data: await requestJson<DeleteCollectionResult>(arg.baseUrl, `/documents/collection/${encodeURIComponent(arg.collectionName)}`, { method: 'DELETE', headers: buildHeaders(arg) }) }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Collections', 'Documents', 'Chunks', 'Analytics'],
    }),
    runRagQuery: builder.mutation<RagQueryResponse, RagQueryArg>({
      async queryFn(arg) {
        try {
          return {
            data: await requestJson<RagQueryResponse>(arg.baseUrl, '/rag/query', {
              method: 'POST',
              headers: buildHeaders(arg),
              json: {
                query: arg.query,
                strategy: arg.strategy,
                collection_name: arg.collectionName,
                top_k: arg.topK,
                include_sources: arg.includeSources,
                rerank: arg.rerank,
              },
            }),
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
    compareRagStrategies: builder.mutation<RagCompareResponse, RagCompareArg>({
      async queryFn(arg) {
        try {
          return {
            data: await requestJson<RagCompareResponse>(arg.baseUrl, '/rag/compare', {
              method: 'POST',
              headers: buildHeaders(arg),
              json: {
                query: arg.query,
                strategies: arg.strategies,
                collection_name: arg.collectionName,
                top_k: arg.topK,
              },
            }),
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
    runAgent: builder.mutation<AgentResult, AgentArg>({
      async queryFn(arg) {
        try {
          return {
            data: await requestJson<AgentResult>(arg.baseUrl, '/agents/run', {
              method: 'POST',
              headers: buildHeaders(arg),
              json: {
                task: arg.task,
                agent_type: arg.agentType,
                max_iterations: arg.maxIterations,
                tools: arg.tools,
                session_id: arg.sessionId ?? undefined,
                use_memory: arg.useMemory ?? true,
              },
            }),
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
    submitFeedback: builder.mutation<FeedbackItem, FeedbackArg>({
      async queryFn(arg) {
        try {
          return {
            data: await requestJson<FeedbackItem>(arg.baseUrl, '/analytics/feedback', {
              method: 'POST',
              headers: buildHeaders(arg),
              json: {
                query_log_id: arg.queryLogId,
                rating: arg.rating,
                comment: arg.comment,
                query: arg.query,
                strategy: arg.strategy,
                collection_name: arg.collectionName,
              },
            }),
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
    chat: builder.mutation<ChatResult, ChatArg>({
      async queryFn(arg) {
        try {
          return {
            data: await requestJson<ChatResult>(arg.baseUrl, '/chat/', {
              method: 'POST',
              headers: buildHeaders(arg),
              json: {
                messages: arg.messages,
                session_id: arg.sessionId ?? undefined,
                rag_strategy: arg.ragStrategy ?? undefined,
                collection_name: arg.collectionName,
                stream: false,
              },
            }),
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
    streamChat: builder.mutation<StreamChatResult, ChatArg>({
      async queryFn(arg) {
        try {
          const response = await fetch(buildUrl(arg.baseUrl, '/chat/stream'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...buildHeaders(arg) },
            body: JSON.stringify({
              messages: arg.messages,
              session_id: arg.sessionId ?? undefined,
              rag_strategy: arg.ragStrategy ?? undefined,
              collection_name: arg.collectionName,
              stream: true,
            }),
          })

          if (!response.ok) {
            return { error: { status: response.status, data: await response.text() } as never }
          }

          return {
            data: {
              text: await readSseText(response),
              session_id: response.headers.get('X-Session-ID') ?? arg.sessionId ?? '',
            },
          }
        } catch (error) {
          return { error: error as never }
        }
      },
      invalidatesTags: ['Analytics'],
    }),
  }),
})

export const {
  useCheckHealthQuery,
  useGetCollectionsQuery,
  useGetDocumentsQuery,
  useGetDocumentChunksQuery,
  useGetRagStrategiesQuery,
  useGetAgentTypesQuery,
  useGetQueryLogsQuery,
  useGetFeedbackQuery,
  useIngestDocumentMutation,
  useDeleteDocumentMutation,
  useDeleteCollectionMutation,
  useRunRagQueryMutation,
  useCompareRagStrategiesMutation,
  useRunAgentMutation,
  useSubmitFeedbackMutation,
  useChatMutation,
  useStreamChatMutation,
} = ragApi

export const store = configureStore({
  reducer: {
    [ragApi.reducerPath]: ragApi.reducer,
  },
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(ragApi.middleware),
})

export type AppStore = typeof store
export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
