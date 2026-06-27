function ProgressBar({ progress = 0, animated = true }) {
  const clampedProgress = Math.min(100, Math.max(0, progress));
  const isComplete = clampedProgress === 100;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">Progress</span>
        <span className={`font-bold tabular-nums ${isComplete ? 'text-accent-green' : 'text-brand-400'}`}>
          {clampedProgress}%
        </span>
      </div>

      <div className="relative h-3 bg-surface-700 rounded-full overflow-hidden border border-surface-600">
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out ${
            isComplete
              ? 'bg-gradient-to-r from-accent-green to-emerald-400'
              : 'bg-gradient-to-r from-brand-700 to-brand-400'
          }`}
          style={{ width: `${clampedProgress}%` }}
          role="progressbar"
          aria-valuenow={clampedProgress}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          {animated && !isComplete && clampedProgress > 0 && (
            <div className="absolute inset-0 shimmer" />
          )}
        </div>

        {clampedProgress > 0 && !isComplete && (
          <div
            className="absolute top-0 bottom-0 w-4 blur-sm opacity-60 bg-brand-400 transition-all duration-700"
            style={{ left: `calc(${clampedProgress}% - 8px)` }}
          />
        )}
      </div>

      <div className="flex justify-between">
        {[0, 25, 50, 75, 100].map((tick) => (
          <div
            key={tick}
            className={`w-1 h-1 rounded-full transition-all duration-500 ${
              clampedProgress >= tick ? (isComplete ? 'bg-accent-green' : 'bg-brand-500') : 'bg-surface-500'
            }`}
          />
        ))}
      </div>
    </div>
  );
}

export default ProgressBar;
