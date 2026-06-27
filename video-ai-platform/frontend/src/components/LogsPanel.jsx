import { useEffect, useRef } from 'react';

const LOG_LEVEL_STYLES = {
  info: 'text-brand-400',
  success: 'text-accent-green',
  warning: 'text-accent-amber',
  error: 'text-accent-rose',
  stage: 'text-accent-purple',
  system: 'text-slate-500',
};

const LOG_PREFIXES = {
  info: '›',
  success: '✓',
  warning: '⚠',
  error: '✗',
  stage: '◈',
  system: '·',
};

function LogsPanel({ logs = [] }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="section-label">Processing Logs</p>
        <span className="text-xs text-slate-600 font-mono">{logs.length} entries</span>
      </div>

      <div
        id="logs-panel"
        className="bg-surface-900 border border-surface-700 rounded-xl h-52 overflow-y-auto font-mono text-xs p-3 space-y-1"
      >
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-600 text-xs">No logs yet. Start processing to see output here.</p>
          </div>
        ) : (
          logs.map((log, idx) => {
            const level = log.level || 'info';
            const color = LOG_LEVEL_STYLES[level] || LOG_LEVEL_STYLES.info;
            const prefix = LOG_PREFIXES[level] || '›';

            return (
              <div key={idx} className="flex items-start gap-2 animate-fade-in leading-relaxed">
                <span className="flex-shrink-0 text-slate-600 tabular-nums">{log.timestamp}</span>
                <span className={`flex-shrink-0 font-bold ${color}`}>{prefix}</span>
                <span className={color}>{log.message}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

export default LogsPanel;
