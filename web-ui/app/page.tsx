'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Sparkles } from 'lucide-react'
import { AutoModeIndicator } from '@/components/AutoModeIndicator'
import { IntentBreakdown } from '@/components/IntentBreakdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  intent?: string
  confidence?: number
}

interface ChatState {
  messages: Message[]
  isLoading: boolean
  inferredTone: string | null
}

export default function Home() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    inferredTone: null,
  })
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state.messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || state.isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    }

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
    }))
    setInput('')

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: [...state.messages, userMessage].map(m => ({
            role: m.role,
            content: m.content,
          })),
        }),
      })

      const data = await response.json()
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.choices?.[0]?.message?.content || 'No response',
        intent: data.intent,
        confidence: data.confidence,
      }

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        isLoading: false,
        inferredTone: data.tone || null,
      }))
    } catch (error) {
      console.error('Chat error:', error)
      setState(prev => ({
        ...prev,
        messages: [...prev.messages, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Error: Unable to get response. Please try again.',
        }],
        isLoading: false,
      }))
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <main className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary-600" />
            <h1 className="text-lg font-semibold text-gray-900">SPSE Engine</h1>
          </div>
          <AutoModeIndicator />
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-4xl space-y-6">
          {state.messages.length === 0 ? (
            <div className="flex h-[60vh] flex-col items-center justify-center text-center">
              <Sparkles className="h-12 w-12 text-primary-400 mb-4" />
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                Welcome to SPSE Engine
              </h2>
              <p className="text-gray-500 max-w-md">
                Auto-Intelligence is active. The system will automatically adapt 
                its reasoning and response style based on your query.
              </p>
            </div>
          ) : (
            state.messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.intent && (
                    <IntentBreakdown 
                      intent={message.intent} 
                      confidence={message.confidence || 0} 
                    />
                  )}
                </div>
              </div>
            ))
          )}
          {state.isLoading && (
            <div className="flex justify-start">
              <div className="rounded-2xl bg-gray-100 px-4 py-3">
                <Loader2 className="h-5 w-5 animate-spin text-primary-600" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="sticky bottom-0 border-t bg-white/80 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="mx-auto max-w-4xl px-4 py-4">
          <div className="flex items-end gap-3">
            <div className="relative flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message..."
                rows={1}
                className="w-full resize-none rounded-xl border border-gray-200 bg-white px-4 py-3 pr-12 text-gray-900 placeholder-gray-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
                style={{ minHeight: '48px', maxHeight: '200px' }}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || state.isLoading}
              className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-600 text-white transition-colors hover:bg-primary-700 disabled:bg-gray-200 disabled:text-gray-400"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
          {state.inferredTone && (
            <p className="mt-2 text-sm text-gray-500">
              Inferred tone: <span className="font-medium">{state.inferredTone}</span>
            </p>
          )}
        </form>
      </div>
    </main>
  )
}
