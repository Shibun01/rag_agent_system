import { startTransition, useEffect, useMemo, useState, type ReactNode } from 'react'
import './App.css'
import {
  AGENT_TYPES,
  CHUNK_STRATEGIES,
  DEFAULT_API_URL,
  type AgentResult,
  type ChatMessage,
  type ChunkStrategy,
  type ConnectionOptions,
  type FeedbackItem,
  type RagCompareResponse,
  type RagQueryResponse,
  type RagSource,
  useCheckHealthQuery,
  useChatMutation,
  useCompareRagStrategiesMutation,
  useDeleteCollectionMutation,
  useDeleteDocumentMutation,
  useGetAgentTypesQuery,
  useGetCollectionsQuery,
  useGetDocumentChunksQuery,
  useGetDocumentsQuery,
  useGetFeedbackQuery,
  useGetQueryLogsQuery,
  useGetRagStrategiesQuery,
  useIngestDocumentMutation,
  useRunAgentMutation,
  useRunRagQueryMutation,
  useStreamChatMutation,
  useSubmitFeedbackMutation,
} from './app/store'

type Tab = 'flow' | 'rag' | 'chat' | 'agents' | 'ingest' | 'manage' | 'analytics'

type FlowActivity = {
  connectionChecked: boolean
  ingestCompleted: boolean
  chunkPreviewed: boolean
  ragQueried: boolean
  ragCompared: boolean
  chatSent: boolean
  agentRan: boolean
  feedbackSubmitted: boolean
  tabsVisited: Partial<Record<Tab, boolean>>
}

type FlowStep = {
  id: string
  step: string
  title: string
  summary: string
  demo: string
  checkpoint: string
  ready: boolean
  complete: boolean
  primaryLabel: string
  onPrimary: () => void
  secondaryLabel?: string
  onSecondary?: () => void
}

const TAB_LABELS: Record<Tab, string> = {
  flow: 'Flow',
  rag: 'Retrieval',
  chat: 'Chat',
  agents: 'Agents',
  ingest: 'Ingest',
  manage: 'Collections',
  analytics: 'Analytics',
}

const CHUNK_STRATEGY_COPY: Record<ChunkStrategy, string> = {
  recursive: 'Balanced chunking for general document ingestion.',
  semantic: 'Sentence-aware splitting that favors meaning boundaries.',
  parent_child: 'Large parent context with compact child retrieval units.',
  sentence_window: 'Fine-grained retrieval with local context windows.',
}

const inputCls = 'w-full rounded-2xl border border-white/10 bg-white/6 px-4 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60 focus:bg-slate-900/70 focus:ring-2 focus:ring-cyan-300/20'
const textareaCls = `${inputCls} resize-none`
const subtleInputCls = 'w-full rounded-2xl border border-white/10 bg-slate-950/55 px-4 py-2.5 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/20'
const primaryBtn = 'inline-flex items-center justify-center rounded-2xl bg-linear-to-r from-cyan-300 via-sky-300 to-emerald-300 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(34,211,238,0.24)] transition hover:-translate-y-0.5 hover:shadow-[0_18px_55px_rgba(34,211,238,0.32)] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:translate-y-0'
const ghostBtn = 'inline-flex items-center justify-center rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:border-cyan-300/30 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-45'
const dangerBtn = 'inline-flex items-center justify-center rounded-2xl border border-rose-400/20 bg-rose-400/8 px-4 py-2.5 text-sm font-medium text-rose-100 transition hover:border-rose-300/30 hover:bg-rose-400/14 disabled:cursor-not-allowed disabled:opacity-45'


function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}


function getErrorMessage(error: unknown): string | null {
  if (!error) return null
  if (typeof error === 'string') return error
  if (typeof error === 'object') {
    if ('data' in error && typeof (error as { data?: unknown }).data === 'string') {
      return (error as { data: string }).data
    }
    if ('error' in error && typeof (error as { error?: unknown }).error === 'string') {
      return (error as { error: string }).error
    }
    if ('message' in error && typeof (error as { message?: unknown }).message === 'string') {
      return (error as { message: string }).message
    }
    if ('status' in error) {
      return `Request failed (${String((error as { status: unknown }).status)})`
    }
  }
  return 'Request failed.'
}


function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-2.5">
      <span className="flex items-center justify-between gap-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        <span>{label}</span>
        {hint ? <span className="text-slate-500">{hint}</span> : null}
      </span>
      {children}
    </label>
  )
}


function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cx('aurora-panel relative overflow-hidden rounded-[28px] border border-white/10 p-5 sm:p-6', className)}>{children}</section>
}


function PanelHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="space-y-2">
        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/72">{eyebrow}</p>
        <div>
          <h2 className="text-xl font-semibold text-white sm:text-2xl">{title}</h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-400">{description}</p>
        </div>
      </div>
      {action}
    </div>
  )
}


function MetricTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="metric-tile rounded-3xl border border-white/10 bg-slate-950/45 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-sm text-slate-400">{detail}</p>
    </div>
  )
}


function StatusPill({ tone, children }: { tone: 'good' | 'warn' | 'alert' | 'info'; children: ReactNode }) {
  const toneClass = {
    good: 'border-emerald-300/20 bg-emerald-300/10 text-emerald-100',
    warn: 'border-amber-300/20 bg-amber-300/10 text-amber-100',
    alert: 'border-rose-300/20 bg-rose-300/10 text-rose-100',
    info: 'border-cyan-300/20 bg-cyan-300/10 text-cyan-100',
  }[tone]

  return <span className={cx('inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium', toneClass)}>{children}</span>
}


function EmptyState({ title, copy }: { title: string; copy: string }) {
  return (
    <div className="rounded-3xl border border-dashed border-white/10 bg-slate-950/35 px-5 py-8 text-center">
      <p className="text-sm font-semibold text-slate-200">{title}</p>
      <p className="mt-2 text-sm leading-6 text-slate-500">{copy}</p>
    </div>
  )
}


function SourceCards({ sources }: { sources: RagSource[] }) {
  if (sources.length === 0) {
    return <EmptyState title="No retrieved sources" copy="This response was generated without attached passages." />
  }

  return (
    <div className="grid gap-3">
      {sources.map((source, index) => {
        const score = typeof source.score === 'number' ? `${Math.round(source.score * 100)}% match` : 'Retrieved passage'
        return (
          <article key={`${source.id ?? 'source'}-${index}`} className="rounded-3xl border border-white/10 bg-slate-950/45 p-4">
            <div className="mb-3 flex items-center justify-between gap-3 text-xs text-slate-400">
              <span className="font-medium text-slate-300">Source {index + 1}</span>
              <span>{score}</span>
            </div>
            <p className="line-clamp-6 whitespace-pre-wrap text-sm leading-6 text-slate-300">{String(source.content ?? 'No content available.')}</p>
          </article>
        )
      })}
    </div>
  )
}


export default function DashboardApp() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_URL)
  const [tenantId, setTenantId] = useState('default')
  const [apiKey, setApiKey] = useState('')
  const [activeTab, setActiveTab] = useState<Tab>('flow')
  const [showGuide, setShowGuide] = useState(false)
  const [flowActivity, setFlowActivity] = useState<FlowActivity>({
    connectionChecked: false,
    ingestCompleted: false,
    chunkPreviewed: false,
    ragQueried: false,
    ragCompared: false,
    chatSent: false,
    agentRan: false,
    feedbackSubmitted: false,
    tabsVisited: { flow: true },
  })

  const [selectedCollection, setSelectedCollection] = useState('default')
  const [ingestCollection, setIngestCollection] = useState('default')
  const [chatCollection, setChatCollection] = useState('default')
  const [manageCollection, setManageCollection] = useState('default')
  const [feedbackCollection, setFeedbackCollection] = useState('default')

  const [ingestFile, setIngestFile] = useState<File | null>(null)
  const [ingestChunkStrategy, setIngestChunkStrategy] = useState<ChunkStrategy>('recursive')
  const [ingestChunkSize, setIngestChunkSize] = useState(512)
  const [ingestChunkOverlap, setIngestChunkOverlap] = useState(64)

  const [ragQuery, setRagQuery] = useState('')
  const [ragStrategy, setRagStrategy] = useState('advanced')
  const [ragTopK, setRagTopK] = useState(5)
  const [ragAnswer, setRagAnswer] = useState<RagQueryResponse | null>(null)
  const [ragCompareQuery, setRagCompareQuery] = useState('')
  const [ragCompareChoices, setRagCompareChoices] = useState<string[]>(['naive', 'advanced'])
  const [ragCompareResult, setRagCompareResult] = useState<RagCompareResponse | null>(null)

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatSessionId, setChatSessionId] = useState('')
  const [chatUseRag, setChatUseRag] = useState(true)
  const [chatUseStream, setChatUseStream] = useState(false)
  const [chatStrategy, setChatStrategy] = useState('advanced')
  const [chatSources, setChatSources] = useState<RagSource[]>([])

  const [agentTask, setAgentTask] = useState('')
  const [agentType, setAgentType] = useState('react')
  const [agentIterations, setAgentIterations] = useState(10)
  const [agentToolsInput, setAgentToolsInput] = useState('search, calculator')
  const [agentSessionId, setAgentSessionId] = useState('')
  const [agentUseMemory, setAgentUseMemory] = useState(true)
  const [agentResponse, setAgentResponse] = useState<AgentResult | null>(null)

  const [manageDocumentId, setManageDocumentId] = useState('')
  const [previewDocumentId, setPreviewDocumentId] = useState('')

  const [feedbackRating, setFeedbackRating] = useState<'up' | 'down'>('up')
  const [feedbackComment, setFeedbackComment] = useState('')
  const [feedbackQuery, setFeedbackQuery] = useState('')
  const [feedbackStrategy, setFeedbackStrategy] = useState('advanced')

  const connection = useMemo<ConnectionOptions>(() => ({
    baseUrl: apiBase,
    tenantId: tenantId.trim() || undefined,
    apiKey: apiKey.trim() || undefined,
  }), [apiBase, tenantId, apiKey])

  const collectionsQuery = useGetCollectionsQuery(connection)
  const ragStrategiesQuery = useGetRagStrategiesQuery(connection)
  const agentTypesQuery = useGetAgentTypesQuery(connection)
  const healthQuery = useCheckHealthQuery(connection, { pollingInterval: 30000 })
  const queryLogsQuery = useGetQueryLogsQuery({ ...connection, limit: 50 })
  const feedbackQueryState = useGetFeedbackQuery({ ...connection, limit: 50 })
  const documentsQuery = useGetDocumentsQuery({ ...connection, collectionName: manageCollection }, { skip: !manageCollection })
  const chunksQuery = useGetDocumentChunksQuery(
    { ...connection, collectionName: manageCollection, documentId: previewDocumentId, limit: 64 },
    { skip: !manageCollection || !previewDocumentId },
  )

  const [ingestDocument, ingestState] = useIngestDocumentMutation()
  const [deleteDocument, deleteDocumentState] = useDeleteDocumentMutation()
  const [deleteCollection, deleteCollectionState] = useDeleteCollectionMutation()
  const [runRagQuery, ragState] = useRunRagQueryMutation()
  const [compareRagStrategies, compareState] = useCompareRagStrategiesMutation()
  const [runAgent, agentState] = useRunAgentMutation()
  const [submitFeedback, feedbackState] = useSubmitFeedbackMutation()
  const [chat, chatState] = useChatMutation()
  const [streamChat, streamChatState] = useStreamChatMutation()

  const collections = collectionsQuery.data ?? []
  const ragStrategies = ragStrategiesQuery.data ?? ['naive', 'advanced', 'multi_query']
  const agentTypes = agentTypesQuery.data ?? [...AGENT_TYPES]
  const queryLogs = queryLogsQuery.data ?? []
  const feedbackItems = feedbackQueryState.data ?? []
  const documents = documentsQuery.data ?? []
  const chunks = chunksQuery.data ?? []
  const collectionNames = useMemo(() => collections.map((item) => item.name), [collections])
  const collectionOptions = collectionNames.length > 0 ? collectionNames : ['default']
  const preferredCollection = useMemo(() => {
    if (collectionNames.includes(selectedCollection)) {
      return selectedCollection
    }
    if (collectionNames.length > 0) {
      return collectionNames[0]
    }
    const draftCollection = ingestCollection.trim()
    return draftCollection || 'demo-handbook'
  }, [collectionNames, ingestCollection, selectedCollection])

  const isBusy =
    ingestState.isLoading || deleteDocumentState.isLoading || deleteCollectionState.isLoading ||
    ragState.isLoading || compareState.isLoading || agentState.isLoading ||
    feedbackState.isLoading || chatState.isLoading || streamChatState.isLoading

  const dashboardMetrics = useMemo(() => {
    const totalChunks = collections.reduce((sum, item) => sum + item.chunk_count, 0)
    const totalQueries = queryLogs.length
    const successfulQueries = queryLogs.filter((item) => item.status === 'success').length
    const successRate = totalQueries > 0 ? Math.round((successfulQueries / totalQueries) * 100) : 0
    const positiveFeedback = feedbackItems.filter((item) => item.rating === 'up').length

    return {
      totalChunks,
      totalQueries,
      successRate,
      positiveFeedback,
    }
  }, [collections, feedbackItems, queryLogs])

  const latestQuery = queryLogs[0]
  const healthTone: 'good' | 'warn' | 'alert' = healthQuery.data === true ? 'good' : healthQuery.isError ? 'alert' : 'warn'
  const healthLabel = healthQuery.data === true ? 'healthy' : healthQuery.isError ? 'offline' : 'checking'
  const firstError = [
    collectionsQuery.error,
    ragStrategiesQuery.error,
    agentTypesQuery.error,
    queryLogsQuery.error,
    feedbackQueryState.error,
    documentsQuery.error,
    chunksQuery.error,
    ingestState.error,
    deleteDocumentState.error,
    deleteCollectionState.error,
    ragState.error,
    compareState.error,
    agentState.error,
    feedbackState.error,
    chatState.error,
    streamChatState.error,
  ].map(getErrorMessage).find(Boolean) ?? null

  useEffect(() => {
    const fallback = collectionNames[0] ?? 'default'
    if (collectionNames.length > 0) {
      if (!collectionNames.includes(selectedCollection)) setSelectedCollection(fallback)
      if (!collectionNames.includes(ingestCollection)) setIngestCollection(fallback)
      if (!collectionNames.includes(chatCollection)) setChatCollection(fallback)
      if (!collectionNames.includes(manageCollection)) setManageCollection(fallback)
      if (!collectionNames.includes(feedbackCollection)) setFeedbackCollection(fallback)
    }
    if (!ragStrategies.includes(ragStrategy)) setRagStrategy(ragStrategies[0] ?? 'advanced')
    if (!ragStrategies.includes(chatStrategy)) setChatStrategy(ragStrategies[0] ?? 'advanced')
    if (!ragStrategies.includes(feedbackStrategy)) setFeedbackStrategy(ragStrategies[0] ?? 'advanced')
    if (!agentTypes.includes(agentType)) setAgentType(agentTypes[0] ?? 'react')
  }, [agentType, agentTypes, chatCollection, chatStrategy, collectionNames, feedbackCollection, feedbackStrategy, ingestCollection, manageCollection, ragStrategies, ragStrategy, selectedCollection])

  useEffect(() => {
    if (documents.length === 0) {
      setManageDocumentId('')
      setPreviewDocumentId('')
      return
    }
    if (!documents.some((item) => item.document_id === manageDocumentId)) {
      const next = documents[0]?.document_id ?? ''
      setManageDocumentId(next)
      setPreviewDocumentId(next)
    }
  }, [documents, manageDocumentId])

  async function handleRefresh() {
    await Promise.allSettled([
      healthQuery.refetch(),
      collectionsQuery.refetch(),
      ragStrategiesQuery.refetch(),
      agentTypesQuery.refetch(),
      queryLogsQuery.refetch(),
      feedbackQueryState.refetch(),
    ])
    setFlowActivity((previous) => ({
      ...previous,
      connectionChecked: true,
    }))
  }

  async function handleIngest() {
    if (!ingestFile) return
    await ingestDocument({
      ...connection,
      file: ingestFile,
      collectionName: ingestCollection,
      chunkStrategy: ingestChunkStrategy,
      chunkSize: ingestChunkSize,
      chunkOverlap: ingestChunkOverlap,
    }).unwrap()
    setFlowActivity((previous) => ({
      ...previous,
      ingestCompleted: true,
      tabsVisited: {
        ...previous.tabsVisited,
        ingest: true,
      },
    }))
    handleTabChange('manage')
  }

  async function handleRagQuery() {
    if (!ragQuery.trim()) return
    const result = await runRagQuery({
      ...connection,
      query: ragQuery,
      strategy: ragStrategy,
      collectionName: selectedCollection,
      topK: ragTopK,
      includeSources: true,
      rerank: true,
    }).unwrap()
    setRagAnswer(result)
    setFlowActivity((previous) => ({
      ...previous,
      ragQueried: true,
      tabsVisited: {
        ...previous.tabsVisited,
        rag: true,
      },
    }))
  }

  async function handleCompare() {
    if (!ragCompareQuery.trim() || ragCompareChoices.length === 0) return
    const result = await compareRagStrategies({
      ...connection,
      query: ragCompareQuery,
      strategies: ragCompareChoices,
      collectionName: selectedCollection,
      topK: ragTopK,
    }).unwrap()
    setRagCompareResult(result)
    setFlowActivity((previous) => ({
      ...previous,
      ragCompared: true,
      tabsVisited: {
        ...previous.tabsVisited,
        rag: true,
      },
    }))
  }

  async function handleSendChat() {
    if (!chatInput.trim()) return
    const nextMessages = [...chatMessages, { role: 'user', content: chatInput.trim() } as ChatMessage]
    setChatMessages(nextMessages)
    setChatInput('')

    if (chatUseStream) {
      const result = await streamChat({
        ...connection,
        messages: nextMessages,
        sessionId: chatSessionId || undefined,
        ragStrategy: chatUseRag ? chatStrategy : null,
        collectionName: chatCollection,
      }).unwrap()
      setChatSessionId(result.session_id)
      setChatSources([])
      setChatMessages((previous) => [...previous, { role: 'assistant', content: result.text }])
      setFlowActivity((previous) => ({
        ...previous,
        chatSent: true,
        tabsVisited: {
          ...previous.tabsVisited,
          chat: true,
        },
      }))
      return
    }

    const result = await chat({
      ...connection,
      messages: nextMessages,
      sessionId: chatSessionId || undefined,
      ragStrategy: chatUseRag ? chatStrategy : null,
      collectionName: chatCollection,
    }).unwrap()
    setChatSessionId(result.session_id)
    setChatSources(result.sources)
    setChatMessages((previous) => [...previous, result.message])
    setFlowActivity((previous) => ({
      ...previous,
      chatSent: true,
      tabsVisited: {
        ...previous.tabsVisited,
        chat: true,
      },
    }))
  }

  async function handleRunAgent() {
    if (!agentTask.trim()) return
    const tools = agentToolsInput
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    const result = await runAgent({
      ...connection,
      task: agentTask,
      agentType,
      maxIterations: agentIterations,
      tools,
      sessionId: agentSessionId || undefined,
      useMemory: agentUseMemory,
    }).unwrap()
    setAgentResponse(result)
    setFlowActivity((previous) => ({
      ...previous,
      agentRan: true,
      tabsVisited: {
        ...previous.tabsVisited,
        agents: true,
      },
    }))
  }

  function handlePreviewChunks() {
    if (!manageDocumentId) return
    setPreviewDocumentId(manageDocumentId)
    setFlowActivity((previous) => ({
      ...previous,
      chunkPreviewed: true,
      tabsVisited: {
        ...previous.tabsVisited,
        manage: true,
      },
    }))
  }

  async function handleDeleteDocument() {
    if (!manageCollection || !manageDocumentId) return
    await deleteDocument({ ...connection, collectionName: manageCollection, documentId: manageDocumentId }).unwrap()
    setManageDocumentId('')
    setPreviewDocumentId('')
  }

  async function handleDeleteCollection() {
    if (!manageCollection) return
    await deleteCollection({ ...connection, collectionName: manageCollection }).unwrap()
    setManageDocumentId('')
    setPreviewDocumentId('')
  }

  async function handleSubmitFeedback() {
    await submitFeedback({
      ...connection,
      rating: feedbackRating,
      comment: feedbackComment,
      query: feedbackQuery || undefined,
      strategy: feedbackStrategy,
      collectionName: feedbackCollection,
    }).unwrap()
    setFeedbackComment('')
    setFeedbackQuery('')
    setFlowActivity((previous) => ({
      ...previous,
      feedbackSubmitted: true,
      tabsVisited: {
        ...previous.tabsVisited,
        analytics: true,
      },
    }))
  }

  function handleTabChange(tab: Tab) {
    setFlowActivity((previous) => ({
      ...previous,
      tabsVisited: {
        ...previous.tabsVisited,
        [tab]: true,
      },
    }))
    startTransition(() => setActiveTab(tab))
  }

  function prepareIngestDemo() {
    setIngestCollection(preferredCollection)
    setIngestChunkStrategy('sentence_window')
    setIngestChunkSize(512)
    setIngestChunkOverlap(64)
    handleTabChange('ingest')
  }

  function prepareManageDemo() {
    setManageCollection(preferredCollection)
    if (documents.length > 0) {
      const firstDocument = documents[0]
      setManageDocumentId(firstDocument.document_id)
      setPreviewDocumentId(firstDocument.document_id)
      setFlowActivity((previous) => ({
        ...previous,
        chunkPreviewed: true,
      }))
    }
    handleTabChange('manage')
  }

  function prepareRetrievalDemo() {
    setSelectedCollection(preferredCollection)
    setRagStrategy('advanced')
    setRagTopK(6)
    setRagQuery('What are the main responsibilities, deadlines, and open risks in this collection?')
    setRagCompareQuery('Which strategy produces the clearest grounded summary for this collection?')
    setRagCompareChoices(['naive', 'advanced', 'corrective'])
    handleTabChange('rag')
  }

  function prepareChatDemo() {
    setChatCollection(preferredCollection)
    setChatUseRag(true)
    setChatUseStream(false)
    setChatStrategy('advanced')
    setChatInput('Summarize this collection for a new teammate and list the top two follow-up actions.')
    handleTabChange('chat')
  }

  function prepareAgentDemo() {
    setAgentType('rag_agent')
    setAgentToolsInput('search, calculator')
    setAgentUseMemory(true)
    setAgentSessionId('')
    setAgentTask('Use the uploaded collection to summarize the main themes, identify key action items, and produce a short execution plan.')
    handleTabChange('agents')
  }

  function prepareAnalyticsDemo() {
    setFeedbackCollection(preferredCollection)
    setFeedbackStrategy('advanced')
    setFeedbackQuery('What are the main responsibilities, deadlines, and open risks?')
    setFeedbackComment('Use this after reviewing answer quality and source grounding.')
    handleTabChange('analytics')
  }

  const flowSteps: FlowStep[] = [
    {
      id: 'connect',
      step: '01',
      title: 'Connect the workspace',
      summary: 'Confirm the API base, tenant, and optional API key, then refresh the backend status and scoped metadata.',
      demo: 'Use the Environment panel to point the UI at the correct backend and tenant before doing anything else.',
      checkpoint: flowActivity.connectionChecked
        ? healthQuery.data === true
          ? `Connection confirmed for ${apiBase}.`
          : 'A refresh was attempted, but the backend is not currently healthy.'
        : 'Refresh the connection once to validate the target backend and tenant scope.',
      ready: healthQuery.data === true,
      complete: flowActivity.connectionChecked && healthQuery.data === true,
      primaryLabel: healthQuery.isFetching ? 'Refreshing...' : 'Refresh connection',
      onPrimary: () => { void handleRefresh() },
    },
    {
      id: 'ingest',
      step: '02',
      title: 'Ingest a document',
      summary: 'Open the Ingest tab, pick a chunking strategy, and upload a PDF, DOCX, TXT, or Markdown file.',
      demo: `Preload collection ${preferredCollection} with sentence-window chunking for a strong default demo.`,
      checkpoint: flowActivity.ingestCompleted
        ? 'A document was ingested during this walkthrough.'
        : collections.length > 0 || documents.length > 0
          ? 'This tenant already has corpus data. Ingest one file in this session to complete the guided step.'
          : 'No corpus has been ingested in this walkthrough yet.',
      ready: collections.length > 0 || documents.length > 0 || flowActivity.ingestCompleted,
      complete: flowActivity.ingestCompleted,
      primaryLabel: 'Load ingest demo',
      onPrimary: prepareIngestDemo,
      secondaryLabel: 'Open ingest tab',
      onSecondary: () => handleTabChange('ingest'),
    },
    {
      id: 'manage',
      step: '03',
      title: 'Inspect chunks and documents',
      summary: 'Use Collections to verify the stored document inventory and preview the chunk text that retrieval will operate on.',
      demo: 'Jump to the collection inventory, select a document, and preview its stored chunk payload.',
      checkpoint: flowActivity.chunkPreviewed || (Boolean(flowActivity.tabsVisited.manage) && chunks.length > 0)
        ? 'Chunk preview was opened from the collection view.'
        : documents.length > 0
          ? 'Documents are available. Open Collections and preview chunk text to complete this step.'
          : 'No document inventory is available to inspect yet.',
      ready: documents.length > 0 || chunks.length > 0 || flowActivity.chunkPreviewed,
      complete: flowActivity.chunkPreviewed || (Boolean(flowActivity.tabsVisited.manage) && chunks.length > 0),
      primaryLabel: 'Open collection review',
      onPrimary: prepareManageDemo,
      secondaryLabel: 'Open collections tab',
      onSecondary: () => handleTabChange('manage'),
    },
    {
      id: 'retrieval',
      step: '04',
      title: 'Run retrieval and compare strategies',
      summary: 'Ask a grounded question, inspect retrieved sources, then compare multiple pipelines against the same prompt.',
      demo: 'Load a sample question plus a comparison prompt so the Retrieval tab demonstrates both answer generation and strategy benchmarking.',
      checkpoint: flowActivity.ragQueried || flowActivity.ragCompared
        ? 'A retrieval action was executed in this walkthrough.'
        : ragAnswer || ragCompareResult || queryLogs.length > 0
          ? 'Retrieval history exists. Run a query or comparison from this session to complete the guide step.'
          : 'No retrieval action has been run yet.',
      ready: Boolean(ragAnswer) || Boolean(ragCompareResult) || queryLogs.length > 0 || flowActivity.ragQueried || flowActivity.ragCompared,
      complete: flowActivity.ragQueried || flowActivity.ragCompared,
      primaryLabel: 'Load retrieval demo',
      onPrimary: prepareRetrievalDemo,
      secondaryLabel: 'Open retrieval tab',
      onSecondary: () => handleTabChange('rag'),
    },
    {
      id: 'chat',
      step: '05',
      title: 'Start grounded chat',
      summary: 'Use the Chat tab to carry a multi-turn session over the same collection with optional streaming.',
      demo: 'Preload a grounded chat prompt that asks for an onboarding summary and explicit action items.',
      checkpoint: flowActivity.chatSent
        ? 'A chat turn was sent during this walkthrough.'
        : chatSessionId || chatMessages.length > 0
          ? 'Chat state exists. Send one message from this session to complete the guide step.'
          : 'No active chat turn has been sent yet.',
      ready: Boolean(chatSessionId) || chatMessages.length > 0 || flowActivity.chatSent,
      complete: flowActivity.chatSent,
      primaryLabel: 'Load chat demo',
      onPrimary: prepareChatDemo,
      secondaryLabel: 'Open chat tab',
      onSecondary: () => handleTabChange('chat'),
    },
    {
      id: 'agent',
      step: '06',
      title: 'Run an agent workflow',
      summary: 'Move from direct retrieval into agent orchestration when you need planning, tools, or multi-step synthesis.',
      demo: 'Preload the RAG agent with a task that turns the collection into a short execution plan.',
      checkpoint: flowActivity.agentRan
        ? 'An agent execution was completed in this walkthrough.'
        : agentResponse
          ? 'An agent response exists. Run one agent action in this session to complete the guide step.'
          : 'No agent run has been captured yet.',
      ready: Boolean(agentResponse) || flowActivity.agentRan,
      complete: flowActivity.agentRan,
      primaryLabel: 'Load agent demo',
      onPrimary: prepareAgentDemo,
      secondaryLabel: 'Open agents tab',
      onSecondary: () => handleTabChange('agents'),
    },
    {
      id: 'analytics',
      step: '07',
      title: 'Close the loop with analytics',
      summary: 'Review query logs, latency, and qualitative feedback so you can tune chunking, retrieval, and prompting.',
      demo: 'Prefill the feedback form so the Analytics tab demonstrates the final operator review step.',
      checkpoint: flowActivity.feedbackSubmitted || (Boolean(flowActivity.tabsVisited.analytics) && (queryLogs.length > 0 || feedbackItems.length > 0))
        ? 'Analytics were reviewed or feedback was submitted from this walkthrough.'
        : queryLogs.length > 0 || feedbackItems.length > 0
          ? 'Telemetry exists. Open Analytics or submit feedback to complete the guide step.'
          : 'No telemetry has been reviewed yet.',
      ready: queryLogs.length > 0 || feedbackItems.length > 0 || flowActivity.feedbackSubmitted,
      complete: flowActivity.feedbackSubmitted || (Boolean(flowActivity.tabsVisited.analytics) && (queryLogs.length > 0 || feedbackItems.length > 0)),
      primaryLabel: 'Load analytics demo',
      onPrimary: prepareAnalyticsDemo,
      secondaryLabel: 'Open analytics tab',
      onSecondary: () => handleTabChange('analytics'),
    },
  ]

  const completedFlowSteps = flowSteps.filter((step) => step.complete).length
  const readyFlowSteps = flowSteps.filter((step) => step.ready).length
  const nextPendingFlowStep = flowSteps.find((step) => !step.complete) ?? flowSteps[flowSteps.length - 1]

  function renderFlowTab() {
    return (
      <div className="space-y-6">
        <Panel>
          <PanelHeader
            eyebrow="Guided walkthrough"
            title="Use the entire project end to end"
            description="This flow demonstrates how the product fits together in practice: connect the backend, ingest documents, inspect chunks, run retrieval, chat over the corpus, invoke agents, and review analytics. The walkthrough now marks completion off actual user interactions while still surfacing existing backend evidence."
            action={<StatusPill tone="info">{completedFlowSteps} / {flowSteps.length} steps completed</StatusPill>}
          />

          <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricTile label="Flow progress" value={`${completedFlowSteps}/${flowSteps.length}`} detail="Completed from actual UI actions in this session" />
            <MetricTile label="Ready signals" value={`${readyFlowSteps}/${flowSteps.length}`} detail="Steps with enough data or telemetry to inspect" />
            <MetricTile label="Preferred collection" value={preferredCollection} detail="Used when loading demo states into each tab" />
            <MetricTile label="Next step" value={nextPendingFlowStep.step} detail={nextPendingFlowStep.title} />
          </div>

          <div className="mt-6 grid gap-3 lg:grid-cols-7">
            {flowSteps.map((step) => (
              <div key={step.step} className={cx(
                'rounded-[24px] border px-4 py-4 transition',
                step.complete
                  ? 'border-cyan-300/24 bg-cyan-300/10'
                  : step.ready
                    ? 'border-sky-300/18 bg-sky-300/8'
                    : 'border-white/10 bg-slate-950/35',
              )}>
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">{step.step}</div>
                <div className="mt-2 text-sm font-semibold text-white">{step.title}</div>
                <div className="mt-2 text-xs leading-5 text-slate-400">{step.complete ? 'Completed' : step.ready ? 'Ready to review' : 'Pending step'}</div>
              </div>
            ))}
          </div>
        </Panel>

        <div className="grid gap-4 xl:grid-cols-2">
          {flowSteps.map((step) => (
            <article key={step.step} className={cx('aurora-panel relative overflow-hidden rounded-[28px] border p-5 sm:p-6', step.complete ? 'border-cyan-300/18' : 'border-white/10')}>
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">Step {step.step}</div>
                  <h3 className="mt-2 text-xl font-semibold text-white">{step.title}</h3>
                </div>
                <StatusPill tone={step.complete ? 'good' : step.ready ? 'info' : 'warn'}>{step.complete ? 'Completed' : step.ready ? 'Detected' : 'Next step'}</StatusPill>
              </div>

              <p className="mt-4 text-sm leading-6 text-slate-300">{step.summary}</p>

              <div className="mt-4 rounded-[24px] border border-white/10 bg-slate-950/45 p-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">What the UI will demonstrate</div>
                <p className="mt-2 text-sm leading-6 text-slate-300">{step.demo}</p>
              </div>

              <div className="mt-4 rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Checkpoint</div>
                <p className="mt-2 text-sm leading-6 text-slate-300">{step.checkpoint}</p>
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                <button type="button" className={primaryBtn} onClick={step.onPrimary}>{step.primaryLabel}</button>
                {step.secondaryLabel && step.onSecondary ? <button type="button" className={ghostBtn} onClick={step.onSecondary}>{step.secondaryLabel}</button> : null}
              </div>
            </article>
          ))}
        </div>
      </div>
    )
  }

  function renderGuideDrawer() {
    if (!showGuide) {
      return null
    }

    return (
      <div className="fixed inset-0 z-50 flex justify-end">
        <button
          type="button"
          aria-label="Close guide"
          className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"
          onClick={() => setShowGuide(false)}
        />

        <aside className="relative z-10 flex h-full w-full max-w-xl flex-col border-l border-white/10 bg-[linear-gradient(180deg,rgba(5,11,20,0.98),rgba(3,7,15,0.98))] shadow-[-24px_0_80px_rgba(2,6,23,0.42)]">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-4 sm:px-6">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">Compact guide</p>
              <h2 className="mt-1 text-xl font-semibold text-white">Project flow</h2>
            </div>
            <button type="button" className={ghostBtn} onClick={() => setShowGuide(false)}>Close</button>
          </div>

          <div className="data-scroll flex-1 overflow-y-auto px-5 py-5 sm:px-6">
            <div className="rounded-[28px] border border-cyan-300/16 bg-cyan-300/10 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-100/70">Session progress</div>
                  <div className="mt-2 text-3xl font-semibold text-white">{completedFlowSteps}/{flowSteps.length}</div>
                </div>
                <StatusPill tone="info">Next {nextPendingFlowStep.step}</StatusPill>
              </div>
              <p className="mt-3 text-sm leading-6 text-cyan-50/80">{nextPendingFlowStep.title}: {nextPendingFlowStep.summary}</p>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-950/45">
                <div className="h-full rounded-full bg-linear-to-r from-cyan-300 via-sky-300 to-emerald-300 transition-all" style={{ width: `${(completedFlowSteps / flowSteps.length) * 100}%` }} />
              </div>
            </div>

            <div className="mt-5 space-y-4">
              {flowSteps.map((step) => (
                <article key={step.id} className={cx(
                  'rounded-[26px] border p-4 transition',
                  step.complete
                    ? 'border-cyan-300/20 bg-cyan-300/10'
                    : step.ready
                      ? 'border-sky-300/18 bg-sky-300/8'
                      : 'border-white/10 bg-slate-950/45',
                )}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Step {step.step}</div>
                      <h3 className="mt-2 text-base font-semibold text-white">{step.title}</h3>
                    </div>
                    <StatusPill tone={step.complete ? 'good' : step.ready ? 'info' : 'warn'}>{step.complete ? 'Done' : step.ready ? 'Ready' : 'Pending'}</StatusPill>
                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-300">{step.checkpoint}</p>

                  <div className="mt-4 flex flex-wrap gap-3">
                    <button
                      type="button"
                      className={primaryBtn}
                      onClick={() => {
                        step.onPrimary()
                        setShowGuide(false)
                      }}
                    >
                      {step.primaryLabel}
                    </button>
                    {step.onSecondary && step.secondaryLabel ? (
                      <button
                        type="button"
                        className={ghostBtn}
                        onClick={() => {
                          step.onSecondary?.()
                          setShowGuide(false)
                        }}
                      >
                        {step.secondaryLabel}
                      </button>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          </div>
        </aside>
      </div>
    )
  }

  function renderRagTab() {
    return (
      <div className="space-y-6">
        <Panel>
          <PanelHeader
            eyebrow="Grounded retrieval"
            title="Interrogate the corpus"
            description="Run a focused question through a chosen retrieval strategy, inspect the answer, then compare neighboring pipelines without leaving the same workspace state."
            action={latestQuery ? <StatusPill tone="info">Last latency {Math.round(latestQuery.latency_ms)} ms</StatusPill> : undefined}
          />

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="lg:col-span-2">
              <Field label="Question" hint="Grounded prompt">
                <textarea rows={5} value={ragQuery} onChange={(event) => setRagQuery(event.target.value)} className={textareaCls} placeholder="Ask a document-grounded question with enough specificity to stress retrieval quality." />
              </Field>
            </div>
            <Field label="Collection">
              <select value={selectedCollection} onChange={(event) => setSelectedCollection(event.target.value)} className={inputCls}>
                {collectionOptions.map((name) => <option key={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="Strategy">
              <select value={ragStrategy} onChange={(event) => setRagStrategy(event.target.value)} className={inputCls}>
                {ragStrategies.map((strategy) => <option key={strategy}>{strategy}</option>)}
              </select>
            </Field>
            <div className="lg:col-span-2">
              <Field label="Retrieval depth" hint={`Top K ${ragTopK}`}>
                <input type="range" min={1} max={20} value={ragTopK} onChange={(event) => setRagTopK(Number(event.target.value))} className="w-full accent-cyan-300" />
              </Field>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button type="button" className={primaryBtn} onClick={handleRagQuery} disabled={!ragQuery.trim() || isBusy}>
              {ragState.isLoading ? 'Running retrieval...' : 'Run query'}
            </button>
            <StatusPill tone="info">Tenant {connection.tenantId ?? 'default'}</StatusPill>
            <StatusPill tone="good">{ragStrategies.length} strategies online</StatusPill>
          </div>

          {ragAnswer ? (
            <div className="mt-6 grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(19rem,0.8fr)]">
              <article className="rounded-[26px] border border-white/10 bg-slate-950/45 p-5">
                <div className="mb-4 flex flex-wrap items-center gap-3 text-xs text-slate-400">
                  <StatusPill tone="info">{ragAnswer.strategy}</StatusPill>
                  <span>{ragAnswer.sources.length} passages</span>
                </div>
                <p className="whitespace-pre-wrap text-sm leading-7 text-slate-200">{ragAnswer.answer}</p>
              </article>
              <div>
                <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Retrieved context</p>
                <SourceCards sources={ragAnswer.sources} />
              </div>
            </div>
          ) : null}
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Strategy bench"
            title="Compare pipelines side by side"
            description="Probe the same question through multiple retrieval modes to expose latency, grounding, and answer-shape differences."
          />

          <div className="mt-6 space-y-4">
            <Field label="Comparison prompt">
              <textarea rows={4} value={ragCompareQuery} onChange={(event) => setRagCompareQuery(event.target.value)} className={textareaCls} placeholder="Use a question that reveals retrieval tradeoffs across methods." />
            </Field>

            <div className="space-y-3">
              <span className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Strategies in the bench</span>
              <div className="flex flex-wrap gap-2">
                {ragStrategies.map((strategy) => {
                  const selected = ragCompareChoices.includes(strategy)
                  return (
                    <button
                      key={strategy}
                      type="button"
                      className={cx(
                        'rounded-full border px-3.5 py-1.5 text-xs font-medium transition',
                        selected
                          ? 'border-cyan-300/30 bg-cyan-300/14 text-cyan-50 shadow-[0_12px_28px_rgba(34,211,238,0.12)]'
                          : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/8',
                      )}
                      onClick={() => setRagCompareChoices((previous) => selected ? previous.filter((item) => item !== strategy) : [...previous, strategy])}
                    >
                      {strategy}
                    </button>
                  )
                })}
              </div>
            </div>

            <button type="button" className={primaryBtn} onClick={handleCompare} disabled={!ragCompareQuery.trim() || ragCompareChoices.length === 0 || isBusy}>
              {compareState.isLoading ? 'Comparing pipelines...' : 'Run comparison'}
            </button>

            {ragCompareResult ? (
              <div className="grid gap-4 lg:grid-cols-2">
                {ragCompareResult.results.map((item) => (
                  <article key={item.strategy} className="rounded-[26px] border border-white/10 bg-slate-950/45 p-5">
                    <div className="mb-4 flex flex-wrap items-center gap-3 text-xs text-slate-400">
                      <StatusPill tone={item.status === 'success' ? 'good' : 'alert'}>{item.strategy}</StatusPill>
                      <span>{Math.round(item.latency_ms)} ms</span>
                      <span>{item.source_count} sources</span>
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-6 text-slate-200">{item.status === 'success' ? item.answer : item.error ?? 'Pipeline failed.'}</p>
                  </article>
                ))}
              </div>
            ) : null}
          </div>
        </Panel>
      </div>
    )
  }

  function renderChatTab() {
    return (
      <Panel className="overflow-hidden">
        <PanelHeader
          eyebrow="Conversation"
          title="Run grounded chat sessions"
          description="Switch between streaming and buffered chat, keep a persistent session, and optionally ground each turn in the currently scoped document collection."
          action={chatSessionId ? <StatusPill tone="info">Session {chatSessionId.slice(0, 8)}</StatusPill> : undefined}
        />

        <div className="mt-6 grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(18rem,0.75fr)]">
          <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-4 sm:p-5">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setChatUseRag((value) => !value)} className={cx(ghostBtn, chatUseRag && 'border-cyan-300/30 bg-cyan-300/12 text-cyan-50')}>
                {chatUseRag ? 'Grounded' : 'Ungrounded'}
              </button>
              <button type="button" onClick={() => setChatUseStream((value) => !value)} className={cx(ghostBtn, chatUseStream && 'border-cyan-300/30 bg-cyan-300/12 text-cyan-50')}>
                {chatUseStream ? 'Streaming' : 'Buffered'}
              </button>
              {chatSources.length > 0 ? <StatusPill tone="good">{chatSources.length} sources on last reply</StatusPill> : null}
            </div>

            <div className="data-scroll min-h-[24rem] space-y-3 overflow-y-auto rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(12,20,36,0.95),rgba(5,10,20,0.98))] p-4 sm:p-5">
              {chatMessages.length === 0 ? (
                <EmptyState title="No conversation yet" copy="Start with a concrete question and decide whether you want retrieval grounding turned on." />
              ) : chatMessages.map((message, index) => (
                <div key={`${message.role}-${index}`} className={cx('max-w-[85%] rounded-[24px] px-4 py-3 text-sm leading-7 shadow-[0_18px_40px_rgba(2,6,23,0.18)]', message.role === 'user' ? 'ml-auto bg-linear-to-r from-cyan-300 via-sky-300 to-emerald-300 text-slate-950' : message.role === 'system' ? 'border border-amber-300/20 bg-amber-300/10 text-amber-50' : 'border border-white/10 bg-white/6 text-slate-100')}>
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] opacity-70">{message.role}</div>
                  <div className="whitespace-pre-wrap">{message.content}</div>
                </div>
              ))}
            </div>

            <div className="mt-4 flex flex-col gap-3 lg:flex-row">
              <textarea
                rows={3}
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    void handleSendChat()
                  }
                }}
                className={cx(textareaCls, 'flex-1')}
                placeholder="Type a message. Press Enter to send and Shift+Enter for a new line."
              />
              <div className="flex flex-row gap-2 lg:w-40 lg:flex-col">
                <button type="button" className={primaryBtn} onClick={handleSendChat} disabled={!chatInput.trim() || isBusy}>Send</button>
                <button type="button" className={ghostBtn} onClick={() => { setChatMessages([]); setChatSessionId(''); setChatSources([]) }}>Clear</button>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Chat controls</p>
              <div className="mt-4 space-y-4">
                <Field label="Grounding strategy">
                  <select value={chatStrategy} onChange={(event) => setChatStrategy(event.target.value)} className={inputCls} disabled={!chatUseRag}>
                    {ragStrategies.map((strategy) => <option key={strategy}>{strategy}</option>)}
                  </select>
                </Field>
                <Field label="Collection">
                  <select value={chatCollection} onChange={(event) => setChatCollection(event.target.value)} className={inputCls} disabled={!chatUseRag}>
                    {collectionOptions.map((name) => <option key={name}>{name}</option>)}
                  </select>
                </Field>
                <Field label="Session id" hint="Optional override">
                  <input value={chatSessionId} onChange={(event) => setChatSessionId(event.target.value)} className={inputCls} placeholder="Autogenerated if left blank" />
                </Field>
              </div>
            </div>

            <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Retrieved context</p>
              <div className="mt-4">
                <SourceCards sources={chatSources} />
              </div>
            </div>
          </div>
        </div>
      </Panel>
    )
  }

  function renderAgentsTab() {
    const stepPreview = agentResponse?.steps.slice(0, 5) ?? []

    return (
      <div className="space-y-6">
        <Panel>
          <PanelHeader
            eyebrow="Agent orchestration"
            title="Dispatch a task to an agent pattern"
            description="Use the corrected request plumbing to pass tool preferences, explicit session IDs, and memory choices into the underlying agent implementations."
          />

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="lg:col-span-2">
              <Field label="Task">
                <textarea rows={5} value={agentTask} onChange={(event) => setAgentTask(event.target.value)} className={textareaCls} placeholder="Describe the outcome, constraints, and any useful context for the selected agent." />
              </Field>
            </div>
            <Field label="Agent type">
              <select value={agentType} onChange={(event) => setAgentType(event.target.value)} className={inputCls}>
                {agentTypes.map((type) => <option key={type}>{type}</option>)}
              </select>
            </Field>
            <Field label="Max iterations">
              <input type="number" min={1} max={30} value={agentIterations} onChange={(event) => setAgentIterations(Number(event.target.value))} className={inputCls} />
            </Field>
            <Field label="Allowed tools" hint="Comma separated">
              <input value={agentToolsInput} onChange={(event) => setAgentToolsInput(event.target.value)} className={inputCls} placeholder="search, calculator" />
            </Field>
            <Field label="Session id" hint="Optional memory lane">
              <input value={agentSessionId} onChange={(event) => setAgentSessionId(event.target.value)} className={inputCls} placeholder="agent-session-01" />
            </Field>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button type="button" className={cx(ghostBtn, agentUseMemory && 'border-cyan-300/30 bg-cyan-300/12 text-cyan-50')} onClick={() => setAgentUseMemory((value) => !value)}>
              {agentUseMemory ? 'Memory enabled' : 'Memory disabled'}
            </button>
            <button type="button" className={primaryBtn} onClick={handleRunAgent} disabled={!agentTask.trim() || isBusy}>
              {agentState.isLoading ? 'Running agent...' : 'Run agent'}
            </button>
          </div>
        </Panel>

        {agentResponse ? (
          <Panel>
            <PanelHeader
              eyebrow="Agent output"
              title="Execution trace"
              description="Inspect the final answer together with a lightweight view of the step trace and the effective agent settings."
              action={<StatusPill tone="info">{agentResponse.iterations} iterations</StatusPill>}
            />

            <div className="mt-6 grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(18rem,0.85fr)]">
              <article className="rounded-[26px] border border-white/10 bg-slate-950/45 p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Final answer</p>
                <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-slate-100">{agentResponse.result}</p>
              </article>

              <div className="space-y-4">
                <article className="rounded-[26px] border border-white/10 bg-slate-950/45 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Run shape</p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <MetricTile label="Steps" value={String(agentResponse.steps.length)} detail="Recorded trace entries" />
                    <MetricTile label="Tool calls" value={String(agentResponse.tool_calls.length)} detail="Action events captured" />
                  </div>
                </article>

                <article className="rounded-[26px] border border-white/10 bg-slate-950/45 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Trace preview</p>
                  <div className="mt-4 space-y-3">
                    {stepPreview.length === 0 ? <EmptyState title="No steps emitted" copy="This agent returned a result without a detailed trace." /> : stepPreview.map((step, index) => (
                      <div key={index} className="rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm text-slate-300">
                        <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{String(step.type ?? step.tool ?? `step ${index + 1}`)}</div>
                        <div className="line-clamp-4 whitespace-pre-wrap">{JSON.stringify(step)}</div>
                      </div>
                    ))}
                  </div>
                </article>
              </div>
            </div>
          </Panel>
        ) : null}
      </div>
    )
  }

  function renderIngestTab() {
    return (
      <Panel>
        <PanelHeader
          eyebrow="Corpus intake"
          title="Stage new documents for retrieval"
          description="Upload a file, choose a chunking strategy, and push it into the tenant-scoped collection namespace that now backs the service."
        />

        <div className="mt-6 grid gap-5 xl:grid-cols-[minmax(0,1.1fr)_minmax(18rem,0.9fr)]">
          <div className="space-y-4">
            <Field label="Document file" hint={ingestFile?.name ?? 'PDF, DOCX, TXT, MD'}>
              <label className="block cursor-pointer rounded-[28px] border border-dashed border-cyan-300/20 bg-linear-to-br from-cyan-300/8 via-transparent to-emerald-300/8 p-5 transition hover:border-cyan-300/40 hover:bg-cyan-300/10">
                <input type="file" accept=".pdf,.doc,.docx,.txt,.md" onChange={(event) => setIngestFile(event.target.files?.[0] ?? null)} className="hidden" />
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-white">Drop in a document or browse locally</p>
                  <p className="text-sm leading-6 text-slate-400">The uploaded file will be extracted, chunked, and embedded into the chosen collection.</p>
                </div>
              </label>
            </Field>

            <div className="grid gap-4 lg:grid-cols-2">
              <Field label="Collection">
                <input value={ingestCollection} onChange={(event) => setIngestCollection(event.target.value)} className={inputCls} placeholder="collection name" />
              </Field>
              <Field label="Chunk strategy">
                <select value={ingestChunkStrategy} onChange={(event) => setIngestChunkStrategy(event.target.value as ChunkStrategy)} className={inputCls}>
                  {CHUNK_STRATEGIES.map((strategy) => <option key={strategy}>{strategy}</option>)}
                </select>
              </Field>
              <Field label="Chunk size">
                <input type="number" min={128} max={4096} step={64} value={ingestChunkSize} onChange={(event) => setIngestChunkSize(Number(event.target.value))} className={inputCls} />
              </Field>
              <Field label="Chunk overlap">
                <input type="number" min={0} max={512} step={16} value={ingestChunkOverlap} onChange={(event) => setIngestChunkOverlap(Number(event.target.value))} className={inputCls} />
              </Field>
            </div>

            <button type="button" className={primaryBtn} onClick={handleIngest} disabled={!ingestFile || isBusy}>
              {ingestState.isLoading ? 'Embedding document...' : 'Ingest document'}
            </button>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Strategy notes</p>
            <div className="mt-4 space-y-4">
              <article className="rounded-3xl border border-cyan-300/18 bg-cyan-300/10 p-4">
                <p className="text-sm font-semibold text-cyan-50">{ingestChunkStrategy}</p>
                <p className="mt-2 text-sm leading-6 text-cyan-100/78">{CHUNK_STRATEGY_COPY[ingestChunkStrategy]}</p>
              </article>
              <div className="grid gap-3 sm:grid-cols-2">
                <MetricTile label="Tenant" value={connection.tenantId ?? 'default'} detail="Collection names are automatically scoped." />
                <MetricTile label="Overlap" value={String(ingestChunkOverlap)} detail="Tokens shared across adjacent chunks." />
              </div>
            </div>
          </div>
        </div>
      </Panel>
    )
  }

  function renderManageTab() {
    const selectedDocument = documents.find((item) => item.document_id === manageDocumentId)

    return (
      <div className="space-y-6">
        <Panel>
          <PanelHeader
            eyebrow="Collection operations"
            title="Inspect and prune stored chunks"
            description="Review the active tenant collection, preview chunked content, and remove stale documents or entire collections when needed."
          />

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <Field label="Collection">
              <select value={manageCollection} onChange={(event) => setManageCollection(event.target.value)} className={inputCls}>
                {collectionOptions.map((name) => <option key={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="Document">
              <select value={manageDocumentId} onChange={(event) => setManageDocumentId(event.target.value)} className={inputCls}>
                {documents.length > 0
                  ? documents.map((item) => <option key={item.document_id} value={item.document_id}>{item.source}</option>)
                  : <option value="">No documents</option>}
              </select>
            </Field>
          </div>

          <div className="mt-5 flex flex-wrap gap-3">
              <button type="button" className={ghostBtn} onClick={handlePreviewChunks} disabled={!manageDocumentId}>Preview chunks</button>
            <button type="button" className={ghostBtn} onClick={handleDeleteDocument} disabled={!manageDocumentId || isBusy}>Delete document</button>
            <button type="button" className={dangerBtn} onClick={handleDeleteCollection} disabled={!manageCollection || isBusy}>Delete collection</button>
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Stored documents"
            title="Collection inventory"
            description="The table below reflects the scoped collection view returned by the API, including chunk counts and stored metadata."
          />

          <div className="mt-6 overflow-hidden rounded-[26px] border border-white/10 bg-slate-950/45">
            {documents.length === 0 ? (
              <div className="p-5">
                <EmptyState title="No documents available" copy="Ingest a file first, then return here to inspect chunk structure and metadata." />
              </div>
            ) : (
              <table className="min-w-full text-sm">
                <thead className="border-b border-white/10 bg-white/6 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Source</th>
                    <th className="px-4 py-3">Chunks</th>
                    <th className="px-4 py-3">Document id</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/8 text-slate-300">
                  {documents.map((item) => (
                    <tr key={item.document_id} className={cx('cursor-pointer transition hover:bg-white/6', item.document_id === manageDocumentId && 'bg-cyan-300/8')} onClick={() => setManageDocumentId(item.document_id)}>
                      <td className="px-4 py-3 font-medium text-slate-100">{item.source}</td>
                      <td className="px-4 py-3 text-slate-400">{item.chunk_count}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-500">{item.document_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Chunk preview"
            title={selectedDocument ? selectedDocument.source : 'Choose a document'}
            description={selectedDocument ? `${selectedDocument.chunk_count} stored chunks for the selected document.` : 'Select and preview a document to inspect the retrieved chunk content.'}
          />

          <div className="mt-6 space-y-3">
            {chunks.length === 0 ? (
              <EmptyState title="No chunks loaded" copy="Click Preview chunks after selecting a document to inspect the stored retrieval units." />
            ) : chunks.map((chunk) => (
              <article key={chunk.id} className="rounded-[24px] border border-white/10 bg-slate-950/45 p-4">
                <div className="mb-2 font-mono text-[11px] text-slate-500">{chunk.id}</div>
                <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">{chunk.content}</p>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    )
  }

  function renderAnalyticsTab() {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricTile label="Queries" value={String(queryLogs.length)} detail="Recent log entries in scope" />
          <MetricTile label="Success rate" value={`${dashboardMetrics.successRate}%`} detail="Based on the recent query log window" />
          <MetricTile label="Positive feedback" value={String(dashboardMetrics.positiveFeedback)} detail="Thumbs-up ratings in current scope" />
          <MetricTile label="Tenant" value={connection.tenantId ?? 'default'} detail="Analytics are filtered by tenant id" />
        </div>

        <Panel>
          <PanelHeader
            eyebrow="Feedback"
            title="Capture operator signals"
            description="Record qualitative feedback alongside query and strategy context so you can correlate user sentiment with retrieval behavior."
          />

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <Field label="Query" hint="Optional">
              <input value={feedbackQuery} onChange={(event) => setFeedbackQuery(event.target.value)} className={inputCls} placeholder="What was the user trying to do?" />
            </Field>
            <Field label="Collection">
              <select value={feedbackCollection} onChange={(event) => setFeedbackCollection(event.target.value)} className={inputCls}>
                {collectionOptions.map((name) => <option key={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="Strategy">
              <select value={feedbackStrategy} onChange={(event) => setFeedbackStrategy(event.target.value)} className={inputCls}>
                {ragStrategies.map((strategy) => <option key={strategy}>{strategy}</option>)}
              </select>
            </Field>
            <Field label="Rating">
              <div className="flex gap-2">
                <button type="button" className={cx(ghostBtn, feedbackRating === 'up' && 'border-emerald-300/30 bg-emerald-300/12 text-emerald-50')} onClick={() => setFeedbackRating('up')}>Useful</button>
                <button type="button" className={cx(ghostBtn, feedbackRating === 'down' && 'border-amber-300/30 bg-amber-300/12 text-amber-50')} onClick={() => setFeedbackRating('down')}>Needs work</button>
              </div>
            </Field>
            <div className="lg:col-span-2">
              <Field label="Comment">
                <textarea rows={4} value={feedbackComment} onChange={(event) => setFeedbackComment(event.target.value)} className={textareaCls} placeholder="Capture what worked, what failed, and whether the issue was retrieval, ranking, or response quality." />
              </Field>
            </div>
          </div>

          <div className="mt-5">
            <button type="button" className={primaryBtn} onClick={handleSubmitFeedback} disabled={isBusy}>
              {feedbackState.isLoading ? 'Submitting feedback...' : 'Submit feedback'}
            </button>
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Telemetry"
            title="Recent query log"
            description="Use the log stream to spot failure clusters, slow strategies, and collections that need better chunking or filtering."
          />

          <div className="mt-6 overflow-hidden rounded-[26px] border border-white/10 bg-slate-950/45">
            {queryLogs.length === 0 ? (
              <div className="p-5">
                <EmptyState title="No query logs yet" copy="Run a retrieval query or chat turn to start building the scoped analytics trail." />
              </div>
            ) : (
              <table className="min-w-full text-sm">
                <thead className="border-b border-white/10 bg-white/6 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Query</th>
                    <th className="px-4 py-3">Strategy</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/8 text-slate-300">
                  {queryLogs.map((item) => (
                    <tr key={item.id}>
                      <td className="px-4 py-3">
                        <div className="line-clamp-2 max-w-xl text-slate-100">{item.query}</div>
                        <div className="mt-1 text-xs text-slate-500">{item.collection_name}</div>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{item.strategy}</td>
                      <td className="px-4 py-3">
                        <StatusPill tone={item.status === 'success' ? 'good' : 'alert'}>{item.status}</StatusPill>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{Math.round(item.latency_ms)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Feedback stream"
            title="Recent feedback entries"
            description="Pair the operator comment stream with the query log to identify whether issues are rooted in retrieval, ranking, or response formulation."
          />

          <div className="mt-6 overflow-hidden rounded-[26px] border border-white/10 bg-slate-950/45">
            {feedbackItems.length === 0 ? (
              <div className="p-5">
                <EmptyState title="No feedback recorded" copy="Submit a few operator notes to start validating which strategies produce the most reliable user outcomes." />
              </div>
            ) : (
              <table className="min-w-full text-sm">
                <thead className="border-b border-white/10 bg-white/6 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Rating</th>
                    <th className="px-4 py-3">Comment</th>
                    <th className="px-4 py-3">Collection</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/8 text-slate-300">
                  {feedbackItems.map((item: FeedbackItem) => (
                    <tr key={item.id}>
                      <td className="px-4 py-3"><StatusPill tone={item.rating === 'up' ? 'good' : 'warn'}>{item.rating}</StatusPill></td>
                      <td className="px-4 py-3 text-slate-100">{item.comment || 'No comment provided.'}</td>
                      <td className="px-4 py-3 text-slate-400">{item.collection_name}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>
      </div>
    )
  }

  function renderSideRail() {
    return (
      <div className="space-y-6 xl:sticky xl:top-6">
        <Panel>
          <PanelHeader
            eyebrow="Environment"
            title="Connection details"
            description="Drive the UI against different backends, tenant scopes, and authentication envelopes without editing code."
          />

          <div className="mt-6 space-y-4">
            <Field label="API base">
              <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} className={subtleInputCls} placeholder="http://localhost:8000/api/v1" />
            </Field>
            <Field label="Tenant id">
              <input value={tenantId} onChange={(event) => setTenantId(event.target.value)} className={subtleInputCls} placeholder="default" />
            </Field>
            <Field label="API key" hint="Optional">
              <input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} className={subtleInputCls} placeholder="Attach when auth is enabled" />
            </Field>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <StatusPill tone={healthTone}>{healthLabel}</StatusPill>
            <button type="button" className={ghostBtn} onClick={() => { void handleRefresh() }} disabled={healthQuery.isFetching}>
              {healthQuery.isFetching ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Collections"
            title="Scoped inventory"
            description="What the current tenant can see right now."
          />

          <div className="mt-6 space-y-3">
            {collections.length === 0 ? (
              <EmptyState title="No collections visible" copy="Create or ingest into a collection in the current tenant scope to populate this rail." />
            ) : collections.slice(0, 6).map((item) => (
              <button
                key={item.name}
                type="button"
                onClick={() => {
                  setSelectedCollection(item.name)
                  setChatCollection(item.name)
                  setManageCollection(item.name)
                  startTransition(() => setActiveTab('manage'))
                }}
                className="w-full rounded-[24px] border border-white/10 bg-slate-950/45 px-4 py-4 text-left transition hover:border-cyan-300/24 hover:bg-cyan-300/8"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold text-white">{item.name}</span>
                  <span className="text-xs text-slate-500">{item.document_count} docs</span>
                </div>
                <div className="mt-2 text-sm text-slate-400">{item.chunk_count} chunks</div>
              </button>
            ))}
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            eyebrow="Recent activity"
            title="Query pulse"
            description="A quick view of what just happened inside this tenant scope."
          />

          <div className="mt-6 space-y-3">
            {queryLogs.length === 0 ? (
              <EmptyState title="No recent queries" copy="Run a retrieval query or chat turn and the most recent requests will show up here." />
            ) : queryLogs.slice(0, 5).map((item) => (
              <article key={item.id} className="rounded-[24px] border border-white/10 bg-slate-950/45 px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <StatusPill tone={item.status === 'success' ? 'good' : 'alert'}>{item.strategy}</StatusPill>
                  <span className="text-xs text-slate-500">{Math.round(item.latency_ms)} ms</span>
                </div>
                <p className="mt-3 line-clamp-3 text-sm leading-6 text-slate-300">{item.query}</p>
              </article>
            ))}
          </div>
        </Panel>
      </div>
    )
  }

  function renderActiveTab() {
    switch (activeTab) {
      case 'flow':
        return renderFlowTab()
      case 'rag':
        return renderRagTab()
      case 'chat':
        return renderChatTab()
      case 'agents':
        return renderAgentsTab()
      case 'ingest':
        return renderIngestTab()
      case 'manage':
        return renderManageTab()
      case 'analytics':
        return renderAnalyticsTab()
      default:
        return null
    }
  }

  return (
    <div className="dashboard-shell min-h-screen text-slate-100">
      <div className="relative mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(19rem,0.95fr)]">
          <Panel className="hero-panel">
            <div className="relative z-10 space-y-6">
              <div className="flex flex-wrap items-start justify-between gap-6">
                <div className="max-w-2xl space-y-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.26em] text-cyan-200/76">Operational studio</p>
                  <div>
                    <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl">RAG Agent Control Room</h1>
                    <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                      Multi-strategy retrieval, grounded chat, document operations, and agent orchestration in one tenant-aware workspace.
                    </p>
                  </div>
                </div>

                <div className="grid w-full gap-3 sm:grid-cols-2 xl:w-[23rem]">
                  <MetricTile label="Collections" value={String(collections.length)} detail="Visible in current scope" />
                  <MetricTile label="Chunks" value={String(dashboardMetrics.totalChunks)} detail="Indexed retrieval units" />
                  <MetricTile label="Queries" value={String(dashboardMetrics.totalQueries)} detail="Recent analytics window" />
                  <MetricTile label="Success rate" value={`${dashboardMetrics.successRate}%`} detail="Recent retrieval outcomes" />
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-3xl border border-white/10 bg-slate-950/35 px-4 py-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">RAG surface</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">{ragStrategies.length} retrieval strategies wired through query and chat flows.</p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-slate-950/35 px-4 py-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Agent layer</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">{agentTypes.length} agent patterns with tool, session, and memory controls.</p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-slate-950/35 px-4 py-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Tenant scope</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">Current tenant <span className="font-semibold text-white">{connection.tenantId ?? 'default'}</span> across chat, storage, and analytics.</p>
                </div>
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader
              eyebrow="System state"
              title="Runtime pulse"
              description="Quick telemetry from the latest activity window, including operator feedback and the last observed retrieval latency."
            />
            <div className="mt-6 space-y-4">
              <div className="rounded-[24px] border border-white/10 bg-slate-950/45 p-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold text-white">Backend health</span>
                  <StatusPill tone={healthTone}>{healthLabel}</StatusPill>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-400">API root {apiBase}. Health is checked against the service root, while requests flow through the configured API prefix.</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                <MetricTile label="Positive feedback" value={String(dashboardMetrics.positiveFeedback)} detail="Operator thumbs-up count" />
                <MetricTile label="Last latency" value={latestQuery ? `${Math.round(latestQuery.latency_ms)} ms` : '--'} detail="Most recent logged retrieval request" />
              </div>
            </div>
          </Panel>
        </header>

        {firstError ? (
          <div className="mt-6 rounded-[24px] border border-rose-300/20 bg-rose-400/10 px-5 py-4 text-sm text-rose-100 shadow-[0_14px_38px_rgba(244,63,94,0.12)]">
            {firstError}
          </div>
        ) : null}

        <div className="mt-6 flex flex-col gap-3 xl:flex-row xl:items-center">
          <nav className="flex-1 rounded-[28px] border border-white/10 bg-slate-950/65 p-2 shadow-[0_24px_70px_rgba(2,6,23,0.32)] backdrop-blur-xl">
            <div className="flex flex-wrap gap-2">
              {(Object.keys(TAB_LABELS) as Tab[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => handleTabChange(tab)}
                  className={cx(
                    'rounded-[20px] px-4 py-3 text-sm font-medium transition',
                    activeTab === tab
                      ? 'tab-active text-white'
                      : 'text-slate-400 hover:bg-white/6 hover:text-slate-100',
                  )}
                >
                  {TAB_LABELS[tab]}
                </button>
              ))}
            </div>
          </nav>

          <button type="button" className={cx(ghostBtn, 'xl:shrink-0')} onClick={() => setShowGuide(true)}>
            Open guide {completedFlowSteps}/{flowSteps.length}
          </button>
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_22rem]">
          <div className="space-y-6">{renderActiveTab()}</div>
          {renderSideRail()}
        </div>
      </div>

      {renderGuideDrawer()}
    </div>
  )
}