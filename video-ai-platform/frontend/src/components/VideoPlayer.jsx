import { useRef, useState } from 'react';

function VideoPlayer({ src, title = 'Processed Video' }) {
  const videoRef = useRef(null);
  const [error, setError] = useState(false);

  if (!src) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 h-52 bg-surface-900 rounded-xl border-2 border-dashed border-surface-600">
        <div className="w-14 h-14 rounded-2xl bg-surface-700 flex items-center justify-center">
          <svg className="w-7 h-7 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-slate-500">No output yet</p>
          <p className="text-xs text-slate-600 mt-1">Processed video will appear here</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 h-52 bg-rose-950/20 rounded-xl border border-rose-900">
        <svg className="w-8 h-8 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
        </svg>
        <p className="text-sm text-rose-400">Failed to load video</p>
        <a
          href={src}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary text-xs"
        >
          Open in new tab
        </a>
      </div>
    );
  }

  return (
    <div className="space-y-2 animate-fade-in">
      <div className="flex items-center justify-between">
        <p className="section-label">{title}</p>
        <div className="flex gap-3">
          <a
            href={src}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1 transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            Open
          </a>
          <a
            href={src}
            download="processed_video.mp4"
            className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition-colors font-medium"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download
          </a>
        </div>
      </div>

      <div className="relative bg-black rounded-xl overflow-hidden border border-surface-600 shadow-2xl">
        <video
          ref={videoRef}
          id="output-video-player"
          src={src}
          controls
          className="w-full max-h-64 object-contain"
          onError={() => setError(true)}
        />
      </div>
    </div>
  );
}

export default VideoPlayer;
