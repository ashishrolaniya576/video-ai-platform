const FEATURES = [
  {
    id: 'stabilization',
    label: 'Video Stabilization',
    description: 'Remove camera shake and unwanted motion from video using the RAFT optical flow model.',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        {/* Camera / motion icon */}
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M4 8h8a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4a2 2 0 012-2z" />
      </svg>
    ),
    color: 'brand',
  },
  {
    id: 'heavyRainRemoval',
    label: 'Heavy Rain Removal',
    description: 'Remove heavy rain streaks from video frames using the Heavy Rain Removal neural network.',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        {/* Rain / cloud icon */}
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 19l-1 3M12 19l-1 3M16 19l-1 3" />
      </svg>
    ),
    color: 'cyan',
  },
  {
    id: 'videoVisibility',
    label: 'Video Visibility Enhancement',
    description: 'Improve visibility, restore degraded frames, enhance contrast and sharpness using PromptIR.',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        {/* Sparkle / eye icon */}
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 2l.5 1.5M12 22l.5-1.5M4.22 4.22l1.06 1.06M18.72 18.72l1.06 1.06" />
      </svg>
    ),
    color: 'purple',
  },
  {
    id: 'distanceEstimation',
    label: 'Distance Estimation',
    description: 'Detect objects and estimate their distance using the DistanceEstimation_d2 model.',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        {/* Bounding box / scan icon */}
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7V5a2 2 0 012-2h2M17 3h2a2 2 0 012 2v2M21 17v2a2 2 0 01-2 2h-2M7 21H5a2 2 0 01-2-2v-2" />
        <rect x="8" y="8" width="8" height="8" rx="1" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
      </svg>
    ),
    color: 'orange',
  },
];

const colorMap = {
  brand: {
    checked: 'bg-brand-600/20 border-brand-500 text-brand-400',
    unchecked: 'bg-surface-700 border-surface-500 text-slate-400',
    icon: 'text-brand-400',
    dot: 'bg-brand-500',
  },
  cyan: {
    checked: 'bg-cyan-900/20 border-cyan-600 text-cyan-400',
    unchecked: 'bg-surface-700 border-surface-500 text-slate-400',
    icon: 'text-accent-cyan',
    dot: 'bg-accent-cyan',
  },
  purple: {
    checked: 'bg-purple-900/20 border-purple-600 text-purple-400',
    unchecked: 'bg-surface-700 border-surface-500 text-slate-400',
    icon: 'text-accent-purple',
    dot: 'bg-accent-purple',
  },
  orange: {
    checked: 'bg-orange-900/20 border-orange-500 text-orange-400',
    unchecked: 'bg-surface-700 border-surface-500 text-slate-400',
    icon: 'text-orange-400',
    dot: 'bg-orange-500',
  },
};

function FeaturePanel({ features, onChange, disabled }) {
  const handleToggle = (featureId) => {
    if (disabled) return;
    onChange({ ...features, [featureId]: !features[featureId] });
  };

  const selectedCount = Object.values(features).filter(Boolean).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="section-label">Processing Features</p>
        {selectedCount > 0 && (
          <span className="badge bg-brand-900/50 text-brand-300 border border-brand-800">
            {selectedCount} selected
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3">
        {FEATURES.map((feature) => {
          const isChecked = features[feature.id];
          const colors = colorMap[feature.color];

          return (
            <button
              key={feature.id}
              id={`feature-${feature.id}`}
              type="button"
              onClick={() => handleToggle(feature.id)}
              disabled={disabled}
              className={`flex items-center gap-4 p-4 rounded-xl border-2 transition-all duration-200 text-left w-full
                ${isChecked ? colors.checked : colors.unchecked}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-opacity-80 cursor-pointer'}`}
            >
              <div className={`flex-shrink-0 ${isChecked ? colors.icon : 'text-slate-500'} transition-colors duration-200`}>
                {feature.icon}
              </div>

              <div className="flex-1 min-w-0">
                <p className={`font-semibold text-sm ${isChecked ? 'text-slate-100' : 'text-slate-400'} transition-colors duration-200`}>
                  {feature.label}
                </p>
                <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">{feature.description}</p>
              </div>

              <div
                className={`flex-shrink-0 w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all duration-200
                  ${isChecked
                    ? `border-current ${colors.dot.replace('bg-', 'border-').replace('500', '500')}`
                    : 'border-surface-400 bg-surface-600'
                  }`}
              >
                {isChecked && (
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {selectedCount === 0 && (
        <p className="text-xs text-rose-400 flex items-center gap-1.5 mt-1">
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          Select at least one feature to proceed.
        </p>
      )}
    </div>
  );
}

export default FeaturePanel;
