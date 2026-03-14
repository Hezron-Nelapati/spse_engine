/**
 * AutoModeIndicator - Static display component
 * 
 * Phase 6.1: Auto-Mode indicator only - no mode toggles.
 * Displays "Auto-Intelligence Active" badge to indicate the engine
 * is locked to auto mode with no user overrides.
 */
export function AutoModeIndicator() {
  return (
    <div className="flex items-center gap-2 rounded-full bg-green-50 px-3 py-1.5 text-sm font-medium text-green-700 ring-1 ring-inset ring-green-600/20">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-pulse rounded-full bg-green-400 opacity-75"></span>
        <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500"></span>
      </span>
      Auto-Intelligence Active
    </div>
  )
}
