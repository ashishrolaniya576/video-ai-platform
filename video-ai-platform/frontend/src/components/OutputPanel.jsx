import VideoPlayer from './VideoPlayer.jsx';

function OutputPanel({ outputVideo, detectionSummary }) {
  const hasOutput = Boolean(outputVideo);
  const hasDetections = detectionSummary && Object.keys(detectionSummary).length > 0;

  // Sort detections by count descending
  const sortedDetections = hasDetections
    ? Object.entries(detectionSummary).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="space-y-6">
      <VideoPlayer src={outputVideo} title="Output Video" />

      {/* Processing complete indicator */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="section-label">Video Processing Complete</p>
          {hasOutput && (
            <span className="badge bg-emerald-900/50 text-emerald-300 border border-emerald-800">
              Ready
            </span>
          )}
        </div>

        {!hasOutput ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8 bg-surface-900 rounded-xl border-2 border-dashed border-surface-600">
            <svg className="w-8 h-8 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              {/* Eye / visibility icon */}
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <p className="text-xs text-slate-600">
              No output yet. Start processing to see your enhanced video here.
            </p>
          </div>
        ) : (
          <div id="processing-complete" className="flex items-center gap-3 p-3 bg-emerald-900/10 rounded-xl border border-emerald-800/50 animate-slide-up">
            <svg className="w-5 h-5 text-emerald-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <p className="text-sm text-emerald-300 font-medium">
              AI enhancement pipeline finished successfully.
            </p>
          </div>
        )}
      </div>

      {/* Detection Summary Panel */}
      {hasDetections && (
        <div className="space-y-3 animate-slide-up">
          <div className="flex items-center justify-between">
            <p className="section-label flex items-center gap-2">
              <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 7V5a2 2 0 012-2h2M17 3h2a2 2 0 012 2v2M21 17v2a2 2 0 01-2 2h-2M7 21H5a2 2 0 01-2-2v-2" />
                <rect x="8" y="8" width="8" height="8" rx="1" />
              </svg>
              Detection Summary
            </p>
            <span className="badge bg-orange-900/50 text-orange-300 border border-orange-800">
              {sortedDetections.length} {sortedDetections.length === 1 ? 'class' : 'classes'}
            </span>
          </div>

          <div className="rounded-xl border border-surface-600 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-surface-700 border-b border-surface-600">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Object Class
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Detections
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-600">
                {sortedDetections.map(([className, count], idx) => (
                  <tr
                    key={className}
                    className={`transition-colors ${idx % 2 === 0 ? 'bg-surface-800' : 'bg-surface-900'} hover:bg-surface-700`}
                  >
                    <td className="px-4 py-2.5 text-slate-300 font-medium capitalize">
                      {className}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <span className="inline-flex items-center justify-center min-w-[2rem] px-2 py-0.5 rounded-md bg-orange-900/40 text-orange-300 text-xs font-bold border border-orange-800/50">
                        {count}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default OutputPanel;

