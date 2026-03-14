import { NextRequest, NextResponse } from 'next/server'

const SPSE_API_URL = process.env.SPSE_API_URL || 'http://localhost:3001'

/**
 * API Route: /api/health
 * 
 * Health check endpoint to verify SPSE Engine backend connectivity.
 */
export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${SPSE_API_URL}/api/v1/status`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json({
        status: 'healthy',
        backend: 'connected',
        ...data,
      })
    } else {
      return NextResponse.json({
        status: 'unhealthy',
        backend: 'disconnected',
        error: `Backend returned ${response.status}`,
      }, { status: 503 })
    }
  } catch (error) {
    return NextResponse.json({
      status: 'unavailable',
      backend: 'disconnected',
      error: error instanceof Error ? error.message : 'Unknown error',
      hint: 'Start the SPSE Engine backend on port 3000, or use demo mode',
    }, { status: 503 })
  }
}
