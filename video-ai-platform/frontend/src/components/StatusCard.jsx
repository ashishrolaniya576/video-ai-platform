const STATUS_CONFIG = {
  idle: {
    label: 'Idle',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    containerClass: 'bg-surface-700 border-surface-600',
    iconClass: 'text-slate-500',
    labelClass: 'text-slate-400',
    dot: null,
  },
  accepted: {
    label: 'Job Accepted',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    containerClass: 'bg-amber-900/20 border-amber-800',
    iconClass: 'text-accent-amber',
    labelClass: 'text-amber-300',
    dot: 'bg-accent-amber animate-pulse',
  },
  processing: {
    label: 'Processing',
    icon: (
      <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    ),
    containerClass: 'bg-brand-900/20 border-brand-700',
    iconClass: 'text-brand-400',
    labelClass: 'text-brand-300',
    dot: 'bg-brand-400 animate-pulse',
  },
  completed: {
    label: 'Completed',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    ),
    containerClass: 'bg-emerald-900/20 border-emerald-700',
    iconClass: 'text-accent-green',
    labelClass: 'text-emerald-300',
    dot: 'bg-accent-green',
  },
  failed: {
    label: 'Failed',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    containerClass: 'bg-rose-900/20 border-rose-700',
    iconClass: 'text-accent-rose',
    labelClass: 'text-rose-300',
    dot: 'bg-accent-rose animate-pulse',
  },
};

function StatusCard({ status = 'idle', currentStage = '', jobId = null }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.idle;

  return (
    <div className={`flex items-center gap-4 p-4 rounded-xl border transition-all duration-300 ${config.containerClass}`}>
      <div className={`flex-shrink-0 ${config.iconClass} transition-colors duration-300`}>
        {config.icon}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {config.dot && (
            <span className={`inline-block w-2 h-2 rounded-full ${config.dot}`} />
          )}
          <p className={`font-semibold text-sm ${config.labelClass}`}>{config.label}</p>
        </div>

        {currentStage && (
          <p className="text-xs text-slate-400 mt-0.5 truncate">{currentStage}</p>
        )}

        {jobId && (
          <p className="text-xs text-slate-600 mt-0.5 font-mono truncate">Job: {jobId}</p>
        )}
      </div>
    </div>
  );
}

export default StatusCard;
