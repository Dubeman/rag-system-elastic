"use client"

import { useState, useEffect, useRef } from "react"
import { track } from "@/lib/pendo"
import { PendoEvents } from "@/lib/pendo-events"

interface Source {
  title: string
  score: number
  text: string
}

interface QueryResult {
  answer: string
  sources: Source[]
  latency_ms: number
}

/** FastAPI /query returns v1/v2 JSON with `results` + `llm_response`, not this shape. */
function normalizeQueryResponse(raw: unknown, latencyMs: number): QueryResult {
  const r = raw as Record<string, unknown>
  const llm = r.llm_response as Record<string, unknown> | undefined
  const answer =
    typeof llm?.answer === "string"
      ? llm.answer
      : typeof r.answer === "string"
        ? r.answer
        : ""

  const rows = Array.isArray(r.results) ? r.results : []
  const sources: Source[] = rows.map((item) => {
    const row = item as Record<string, unknown>
    const score = typeof row._score === "number" ? row._score : 0
    const title =
      typeof row.filename === "string" && row.filename
        ? row.filename
        : typeof row.doc_id === "string"
          ? `doc ${row.doc_id}`
          : "Source"
    const text = typeof row.content === "string" ? row.content : ""
    return { title, score, text }
  })

  return { answer, sources, latency_ms: latencyMs }
}

async function readApiErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body === "object" && body !== null && "detail" in body) {
      const d = (body as { detail: unknown }).detail
      if (typeof d === "string") return d
      if (Array.isArray(d))
        return d.map((e) => JSON.stringify(e)).join("; ")
    }
  } catch {
    /* ignore */
  }
  return res.statusText || `HTTP ${res.status}`
}

export default function Home() {
  const [query, setQuery] = useState("")
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const queryCount = useRef(0)
  const lastQuery = useRef("")

  useEffect(() => {
    let cancelled = false
    const params = new URLSearchParams(window.location.search)
    const isAgent = params.get("agent") === "true"
    const agentVersion = params.get("agent_version") ?? "v1"

    ;(async () => {
      const pendo = await import("@/lib/pendo")
      if (cancelled) return
      if (isAgent) {
        await pendo.initAgentSession(agentVersion)
      } else {
        await pendo.initHumanSession()
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return

    // Detect agent loop — same query submitted 3+ times
    if (query.trim() === lastQuery.current) {
      queryCount.current += 1
      if (queryCount.current >= 3) {
        track(PendoEvents.AGENT_LOOP_DETECTED, {
          query,
          attempts: queryCount.current,
        })
      }
    } else {
      queryCount.current = 1
      lastQuery.current = query.trim()
    }

    const startTime = Date.now()
    setLoading(true)
    setError(null)
    setResult(null)

    track(PendoEvents.QUERY_SUBMITTED, {
      query,
      queryLength: query.length,
    })

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: query.trim(),
            pipeline_version: "v1",
          }),
        }
      )

      if (!res.ok) {
        const detail = await readApiErrorMessage(res)
        throw new Error(`API error: ${res.status} — ${detail}`)
      }
      const raw = await res.json()
      const latency = Date.now() - startTime
      const data = normalizeQueryResponse(raw, latency)

      setResult(data)
      track(PendoEvents.RESULT_RECEIVED, {
        resultCount: data.sources.length,
        latencyMs: latency,
        hasAnswer: !!data.answer,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error"
      setError(message)
      track(PendoEvents.AGENT_TIMEOUT, { query, error: message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-8">
      <div className="max-w-3xl mx-auto space-y-8">

        {/* Header */}
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Aletheia</h1>
          <p className="text-gray-400 text-sm">
            ἀλήθεια — ask, and what was always there is revealed.
          </p>
        </div>

        {/* Query form */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            id="query-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask something about your documents..."
            className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-sm
                       placeholder:text-gray-600 focus:outline-none focus:border-gray-600"
          />
          <button
            id="submit-button"
            type="submit"
            disabled={loading || !query.trim()}
            className="px-5 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40
                       rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="bg-red-950 border border-red-800 rounded-lg p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
              <p className="text-sm leading-relaxed">{result.answer}</p>
            </div>

            {result.sources.length > 0 && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Sources</p>
                {result.sources.map((source, i) => (
                  <div
                    key={i}
                    id={`source-${i}`}
                    onClick={() =>
                      track(PendoEvents.SOURCE_CLICKED, {
                        sourceIndex: i,
                        sourceTitle: source.title,
                        score: source.score,
                      })
                    }
                    className="bg-gray-900 border border-gray-800 rounded-lg p-4
                               cursor-pointer hover:border-gray-700 transition-colors"
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-medium text-indigo-400">
                        {source.title}
                      </span>
                      <span className="text-xs text-gray-600">
                        score: {source.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 leading-relaxed line-clamp-3">
                      {source.text}
                    </p>
                  </div>
                ))}
              </div>
            )}

            <p className="text-xs text-gray-600">
              Retrieved in {result.latency_ms}ms
            </p>
          </div>
        )}
      </div>
    </main>
  )
}
