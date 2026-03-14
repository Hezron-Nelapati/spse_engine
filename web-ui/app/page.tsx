'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Sparkles, Upload, FileText, X, AlertCircle, Wifi, WifiOff } from 'lucide-react'
import { AutoModeIndicator } from '@/components/AutoModeIndicator'
import { IntentBreakdown } from '@/components/IntentBreakdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  intent?: string
  confidence?: number
  documents?: string[]
}

interface ChatState {
  messages: Message[]
  isLoading: boolean
  inferredTone: string | null
  backendStatus: 'unknown' | 'connected' | 'disconnected'
}

export default function Home() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    inferredTone: null,
    backendStatus: 'unknown',
  })
  const [input, setInput] = useState('')
  const [uploadedDocs, setUploadedDocs] = useState<File[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state.messages])

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('/api/health', { method: 'GET' })
      if (response.ok) {
        setState(prev => ({ ...prev, backendStatus: 'connected' }))
      } else {
        setState(prev => ({ ...prev, backendStatus: 'disconnected' }))
      }
    } catch {
      setState(prev => ({ ...prev, backendStatus: 'disconnected' }))
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) {
      const newFiles = Array.from(files)
      setUploadedDocs(prev => [...prev, ...newFiles])
    }
  }

  const removeUploadedDoc = (index: number) => {
    setUploadedDocs(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if ((!input.trim() && uploadedDocs.length === 0) || state.isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim() || (uploadedDocs.length > 0 ? `Uploaded ${uploadedDocs.length} document(s)` : ''),
      documents: uploadedDocs.map(f => f.name),
    }

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isLoading: true,
    }))
    
    const currentDocs = [...uploadedDocs]
    setInput('')
    setUploadedDocs([])

    // If backend is disconnected, show demo response
    if (state.backendStatus === 'disconnected') {
      setTimeout(() => {
        const demoMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: getDemoResponse(userMessage.content, currentDocs),
          intent: 'Factual',
          confidence: 0.85,
        }
        setState(prev => ({
          ...prev,
          messages: [...prev.messages, demoMessage],
          isLoading: false,
          inferredTone: 'NeutralProfessional',
        }))
      }, 1000)
      return
    }

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: [...state.messages, userMessage].map(m => ({
            role: m.role,
            content: m.content,
          })),
          documents: currentDocs.map(f => f.name),
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.choices?.[0]?.message?.content || 'I processed your request.',
        intent: data.intent,
        confidence: data.confidence,
      }

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        isLoading: false,
        inferredTone: data.tone || null,
        backendStatus: 'connected',
      }))
    } catch (error) {
      console.error('Chat error:', error)
      setState(prev => ({
        ...prev,
        messages: [...prev.messages, {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: '⚠️ Unable to connect to SPSE Engine backend. Please ensure the engine is running on port 3000, or use the demo mode to explore the UI.',
          intent: 'Error',
          confidence: 0,
        }],
        isLoading: false,
        backendStatus: 'disconnected',
      }))
    }
  }

  const getDemoResponse = (query: string, docs: File[]): string => {
    if (docs.length > 0) {
      return `📄 Document Processing Demo\n\nI've received ${docs.length} document(s): ${docs.map(d => d.name).join(', ')}.\n\nIn a live environment, I would:\n1. Parse and index the document content\n2. Extract key entities and concepts\n3. Store in memory for future retrieval\n4. Enable question-answering over the documents\n\nConnect to the SPSE Engine backend to enable full document processing capabilities.`
    }
    
    const lowerQuery = query.toLowerCase()
    if (lowerQuery.includes('hello') || lowerQuery.includes('hi')) {
      return 'Hello! I\'m the SPSE Engine demo mode. The backend is not connected, but I can show you how the UI works. Try uploading a document or asking a question!'
    }
    if (lowerQuery.includes('help')) {
      return 'SPSE Engine Demo Mode\n\nFeatures:\n• Auto-Intelligence Active - no manual mode selection\n• Document Upload - attach files for processing\n• Intent Detection - shows query classification\n• Tone Inference - adapts response style\n\nTo enable full functionality, start the SPSE Engine backend.'
    }
    return `Demo Response\n\nYou asked: "${query}"\n\nI detected this as a Factual query with 85% confidence.\n\nIn demo mode, I provide placeholder responses. Connect to the SPSE Engine backend for real AI-powered responses with:\n• Dynamic reasoning\n• Memory-augmented retrieval\n• Multi-source consensus\n• Adaptive tone matching`
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
          <div className="flex items-center gap-3">
            {/* Backend Status Indicator */}
            <div className={`flex items-center gap-1.5 text-xs ${
              state.backendStatus === 'connected' ? 'text-green-600' :
              state.backendStatus === 'disconnected' ? 'text-amber-600' : 'text-gray-400'
            }`}>
              {state.backendStatus === 'connected' ? (
                <Wifi className="h-3.5 w-3.5" />
              ) : state.backendStatus === 'disconnected' ? (
                <WifiOff className="h-3.5 w-3.5" />
              ) : null}
              <span className="hidden sm:inline">
                {state.backendStatus === 'connected' ? 'Connected' :
                 state.backendStatus === 'disconnected' ? 'Demo Mode' : 'Checking...'}
              </span>
            </div>
            <AutoModeIndicator />
          </div>
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
              <p className="text-gray-500 max-w-md mb-6">
                Auto-Intelligence is active. The system will automatically adapt 
                its reasoning and response style based on your query.
              </p>
              
              {/* Quick Start Guide */}
              <div className="grid gap-3 sm:grid-cols-2 max-w-lg w-full">
                <div className="rounded-lg border border-gray-200 p-4 text-left">
                  <div className="flex items-center gap-2 mb-2">
                    <Send className="h-4 w-4 text-primary-600" />
                    <span className="font-medium text-gray-900">Ask a Question</span>
                  </div>
                  <p className="text-sm text-gray-500">Type any question and get an AI-powered response.</p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4 text-left">
                  <div className="flex items-center gap-2 mb-2">
                    <Upload className="h-4 w-4 text-primary-600" />
                    <span className="font-medium text-gray-900">Upload Documents</span>
                  </div>
                  <p className="text-sm text-gray-500">Attach files for processing and analysis.</p>
                </div>
              </div>

              {/* Demo Mode Notice */}
              {state.backendStatus === 'disconnected' && (
                <div className="mt-6 flex items-center gap-2 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700">
                  <AlertCircle className="h-4 w-4" />
                  <span>Running in demo mode. Start the SPSE Engine backend for full functionality.</span>
                </div>
              )}
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
                      : message.intent === 'Error'
                      ? 'bg-red-50 text-red-900 border border-red-200'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  {/* Show attached documents */}
                  {message.documents && message.documents.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-1">
                      {message.documents.map((doc, i) => (
                        <span key={i} className="inline-flex items-center gap-1 rounded bg-white/20 px-2 py-0.5 text-xs">
                          <FileText className="h-3 w-3" />
                          {doc}
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.intent && message.intent !== 'Error' && (
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
          {/* Uploaded Documents Display */}
          {uploadedDocs.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {uploadedDocs.map((doc, index) => (
                <div key={index} className="flex items-center gap-1 rounded-lg bg-primary-50 px-3 py-1.5 text-sm text-primary-700">
                  <FileText className="h-4 w-4" />
                  <span className="max-w-[150px] truncate">{doc.name}</span>
                  <button
                    type="button"
                    onClick={() => removeUploadedDoc(index)}
                    className="ml-1 rounded-full p-0.5 hover:bg-primary-200"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
          
          <div className="flex items-end gap-3">
            {/* Upload Button */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.pdf,.doc,.docx,.md,.json,.csv"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex h-12 w-12 items-center justify-center rounded-xl border border-gray-200 text-gray-500 transition-colors hover:bg-gray-50 hover:text-gray-700"
              title="Upload documents"
            >
              <Upload className="h-5 w-5" />
            </button>
            
            {/* Text Input */}
            <div className="relative flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={uploadedDocs.length > 0 ? "Add a message about your documents..." : "Type your message..."}
                rows={1}
                className="w-full resize-none rounded-xl border border-gray-200 bg-white px-4 py-3 pr-12 text-gray-900 placeholder-gray-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
                style={{ minHeight: '48px', maxHeight: '200px' }}
              />
            </div>
            
            {/* Send Button */}
            <button
              type="submit"
              disabled={(!input.trim() && uploadedDocs.length === 0) || state.isLoading}
              className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-600 text-white transition-colors hover:bg-primary-700 disabled:bg-gray-200 disabled:text-gray-400"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
          
          {/* Status Bar */}
          <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
            <span>Press Enter to send, Shift+Enter for new line</span>
            {state.inferredTone && (
              <span className="text-gray-500">
                Inferred tone: <span className="font-medium text-gray-600">{state.inferredTone}</span>
              </span>
            )}
          </div>
        </form>
      </div>
    </main>
  )
}
