interface IntentBreakdownProps {
  intent: string
  confidence: number
}

/**
 * IntentBreakdown - Displays detected intent and confidence
 * 
 * Phase 6.1: Shows the inferred intent breakdown for the response.
 */
export function IntentBreakdown({ intent, confidence }: IntentBreakdownProps) {
  const confidencePercent = Math.round(confidence * 100)
  
  return (
    <div className="mt-2 flex items-center gap-2 text-xs opacity-80">
      <span className="rounded bg-white/20 px-1.5 py-0.5 font-mono">
        {intent}
      </span>
      <span className="text-gray-500">
        {confidencePercent}% confidence
      </span>
    </div>
  )
}
