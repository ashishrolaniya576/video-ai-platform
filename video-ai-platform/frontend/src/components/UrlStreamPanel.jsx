import { useState, useEffect } from 'react';
import { stopUrlLiveStream } from '../services/api.js';

function UrlStreamPanel({ jobId, features, onStop, streamUrl }) {
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Connecting to stream...');
  const [imgSrc, setImgSrc] = useState('');

  useEffect(() => {
    // Generate MJPEG URL from backend
    // Since we are proxying /api through Vite, we construct the URL
    const backendUrl = import.meta.env.VITE_API_URL || '/api';
    const mjpegUrl = `${backendUrl}/live/stream/${jobId}`;
    
    setImgSrc(mjpegUrl);
    setStatus('Live Stream Connected');

    return () => {
      // Cleanup on unmount
      stopUrlLiveStream(jobId).catch(err => {
        console.error("Failed to cleanly stop URL stream:", err);
      });
    };
  }, [jobId]);

  const handleStop = async () => {
    try {
      await stopUrlLiveStream(jobId);
    } catch (err) {
      console.error(err);
    }
    if (onStop) onStop();
  };

  return (
    <div className="space-y-4">
      {error && (
        <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-800 text-xs text-rose-300">
          <strong>Error:</strong> {error}
        </div>
      )}
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status === 'Live Stream Connected' ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
          <span className="text-sm font-medium text-slate-300">{status}</span>
        </div>
        
        <button
          onClick={handleStop}
          className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-rose-900/50 text-rose-300 hover:bg-rose-800 hover:text-white transition-colors border border-rose-800"
        >
          Stop Stream
        </button>
      </div>

      <div className="space-y-2">
        <span className="text-xs font-semibold uppercase tracking-widest text-emerald-500">AI Output</span>
        <div className="aspect-video bg-black rounded-xl border border-surface-600 overflow-hidden relative flex items-center justify-center">
          {imgSrc ? (
            <img 
              src={imgSrc} 
              alt="Live Stream" 
              className="w-full h-full object-contain"
              onError={(e) => {
                setStatus('Disconnected');
                setError('Failed to load stream or stream ended.');
                // Optionally hide image or show fallback icon
              }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <svg className="w-8 h-8 text-brand-400 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default UrlStreamPanel;
