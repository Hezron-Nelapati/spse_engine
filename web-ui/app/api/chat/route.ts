import { NextRequest, NextResponse } from 'next/server'

const SPSE_API_URL = process.env.SPSE_API_URL || 'http://localhost:3001'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface ChatCompletionRequest {
  model?: string
  messages: Message[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
}

/**
 * API Route: /api/chat
 * 
 * Phase 6.1: Proxy to SPSE OpenAI-compatible API.
 * All parameters except messages are ignored (Auto-Mode only).
 */
export async function POST(request: NextRequest) {
  try {
    const body: ChatCompletionRequest = await request.json()
    
    // Forward to SPSE OpenAI-compatible endpoint
    const response = await fetch(`${SPSE_API_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'spse-auto', // Ignored by SPSE, but required by OpenAI spec
        messages: body.messages,
        // temperature, max_tokens, stream - all ignored in Auto-Mode
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      return NextResponse.json(
        { error: `SPSE API error: ${error}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    
    // Add SPSE-specific fields to response
    return NextResponse.json({
      ...data,
      // These fields are populated by SPSE engine
      intent: data.intent || 'factual',
      confidence: data.confidence || 0.85,
      tone: data.tone || 'NeutralProfessional',
    })
  } catch (error) {
    console.error('Chat API error:', error)
    return NextResponse.json(
      { error: 'Failed to process chat request' },
      { status: 500 }
    )
  }
}
