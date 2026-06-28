import { useState, useRef, useCallback, useEffect } from 'react';

// ── Supported formats and platforms ──────────────────────────────────────────
const SUPPORTED_UPLOAD_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
const ACCEPT_STRING = SUPPORTED_UPLOAD_EXTS.join(',');

const URL_PLATFORMS = [
  { label: 'YouTube', color: 'text-red-400 bg-red-950/40 border-red-800' },
  { label: 'Vimeo', color: 'text-sky-400 bg-sky-950/40 border-sky-800' },
  { label: 'Google Drive', color: 'text-blue-400 bg-blue-950/40 border-blue-800' },
  { label: 'Dropbox', color: 'text-blue-300 bg-blue-950/40 border-blue-800' },
  { label: 'OneDrive', color: 'text-cyan-400 bg-cyan-950/40 border-cyan-800' },
  { label: 'Direct MP4', color: 'text-emerald-400 bg-emerald-950/40 border-emerald-800' },
  { label: 'Direct MOV/AVI', color: 'text-teal-400 bg-teal-950/40 border-teal-800' },
  { label: 'MKV/WEBM', color: 'text-violet-400 bg-violet-950/40 border-violet-800' },
];

// ── URL source detection (mirrors backend logic) ──────────────────────────────
const URL_PATTERNS = {
  youtube: /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/)|youtu\.be\/)/i,
  vimeo: /^(https?:\/\/)?(www\.)?vimeo\.com\//i,
  gdrive: /^(https?:\/\/)?(drive|docs)\.google\.com\//i,
  dropbox: /^(https?:\/\/)?(www\.)?dropbox\.com\//i,
  onedrive: /^(https?:\/\/)?(onedrive\.live\.com|1drv\.ms)\//i,
  direct: /\.(mp4|mov|avi|mkv|webm)(\?.*)?$/i,
};

const SOURCE_LABELS = {
  youtube: { label: 'YouTube', color: 'text-red-400 border-red-800 bg-red-950/30' },
  vimeo: { label: 'Vimeo', color: 'text-sky-400 border-sky-800 bg-sky-950/30' },
  gdrive: { label: 'Google Drive', color: 'text-blue-400 border-blue-800 bg-blue-950/30' },
  dropbox: { label: 'Dropbox', color: 'text-blue-300 border-blue-800 bg-blue-950/30' },
  onedrive: { label: 'OneDrive', color: 'text-cyan-400 border-cyan-800 bg-cyan-950/30' },
  direct: { label: 'Direct Video', color: 'text-emerald-400 border-emerald-800 bg-emerald-950/30' },
};

function detectUrlSource(url) {
  for (const [key, regex] of Object.entries(URL_PATTERNS)) {
    if (regex.test(url)) return key;
  }
  return null;
}

// ── Utility helpers ───────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (!bytes) return '—';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDuration(seconds) {
  if (!seconds || isNaN(seconds)) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ── Tab icon components ───────────────────────────────────────────────────────
function UploadIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4-4m0 0L8 12m4-4v12" />
    </svg>
  );
}

function LinkIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
    </svg>
  );
}

function XIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function CheckCircleIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

// ── FilePreview sub-component ─────────────────────────────────────────────────
function FilePreview({ file, metadata, onClear }) {
  const previewRef = useRef(null);

  if (!file) return null;

  return (
    <div className="mt-4 rounded-xl border border-surface-500 bg-surface-700/50 overflow-hidden animate-fade-in">
      {/* Thumbnail + Video preview */}
      {metadata.previewUrl && (
        <div className="relative bg-black aspect-video">
          <video
            ref={previewRef}
            src={metadata.previewUrl}
            className="w-full h-full object-contain"
            controls
            playsInline
            preload="metadata"
          />
        </div>
      )}

      {/* File info bar */}
      <div className="px-4 py-3 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <CheckCircleIcon className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span className="text-sm font-medium text-slate-200 truncate" title={file.name}>
              {file.name}
            </span>
          </div>
          <button
            type="button"
            onClick={onClear}
            className="flex-shrink-0 text-slate-500 hover:text-rose-400 transition-colors duration-150 p-0.5 rounded"
            title="Remove file"
          >
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
          <span>
            <span className="text-slate-500">Size </span>
            <span className="font-mono text-slate-300">{formatBytes(file.size)}</span>
          </span>
          {metadata.duration && (
            <span>
              <span className="text-slate-500">Duration </span>
              <span className="font-mono text-slate-300">{formatDuration(metadata.duration)}</span>
            </span>
          )}
          {metadata.width && metadata.height && (
            <span>
              <span className="text-slate-500">Resolution </span>
              <span className="font-mono text-slate-300">{metadata.width}×{metadata.height}</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Drag-and-drop upload zone ─────────────────────────────────────────────────
function UploadZone({ onFileSelect, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  }, [onFileSelect, disabled]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
    e.target.value = '';
  }, [onFileSelect]);

  return (
    <div>
      <div
        onClick={() => !disabled && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative flex flex-col items-center justify-center gap-3
          border-2 border-dashed rounded-xl p-8 text-center
          transition-all duration-200 cursor-pointer
          ${isDragging
            ? 'border-brand-400 bg-brand-900/20 scale-[1.01]'
            : 'border-surface-500 bg-surface-700/30 hover:border-brand-600 hover:bg-surface-700/50'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        {/* Cloud upload icon */}
        <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200
          ${isDragging ? 'bg-brand-600/30 text-brand-300' : 'bg-surface-600 text-slate-400'}`}>
          <UploadIcon className="w-6 h-6" />
        </div>

        <div>
          <p className="text-sm font-semibold text-slate-300">
            {isDragging ? 'Drop your video here' : 'Drag & drop or click to choose'}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {SUPPORTED_UPLOAD_EXTS.join(' · ').toUpperCase()}
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          id="video-file-input"
          accept={ACCEPT_STRING}
          onChange={handleChange}
          className="sr-only"
          disabled={disabled}
        />
      </div>
    </div>
  );
}

// ── URL Input sub-component ───────────────────────────────────────────────────
function UrlInput({ value, onChange, error, disabled }) {
  const sourceType = value.trim() ? detectUrlSource(value.trim()) : null;
  const sourceInfo = sourceType ? SOURCE_LABELS[sourceType] : null;

  return (
    <div className="space-y-3">
      <div className="relative">
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none">
          <LinkIcon className="w-4 h-4" />
        </div>
        <input
          id="video-url-input"
          type="url"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="https://youtube.com/watch?v=... or https://example.com/video.mp4"
          disabled={disabled}
          className={`
            w-full pl-10 pr-4 py-3 rounded-xl bg-surface-700 border text-slate-100
            placeholder-slate-500 text-sm
            focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent
            transition-all duration-200
            ${error ? 'border-rose-600 focus:ring-rose-500' : 'border-surface-500'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          aria-describedby={error ? 'url-error-msg' : undefined}
        />
      </div>

      {/* Detected source badge */}
      {sourceInfo && !error && (
        <div className="flex items-center gap-2 animate-fade-in">
          <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-400" />
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${sourceInfo.color}`}>
            {sourceInfo.label}
          </span>
          <span className="text-xs text-slate-500">detected</span>
        </div>
      )}

      {/* Supported platforms */}
      <div className="space-y-1.5">
        <p className="text-xs text-slate-500 font-medium">Supported sources:</p>
        <div className="flex flex-wrap gap-1.5">
          {URL_PLATFORMS.map(({ label, color }) => (
            <span
              key={label}
              className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${color}`}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main VideoInputPanel ──────────────────────────────────────────────────────
/**
 * @param {object} props
 * @param {function} props.onSourceReady - Called with { type: 'file'|'url', file?, url? } or null
 * @param {boolean} props.disabled
 * @param {string} [props.externalError] - Error message from parent (validation / API)
 */
function VideoInputPanel({ onSourceReady, disabled, externalError }) {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'url'
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileMetadata, setFileMetadata] = useState({});
  const [fileError, setFileError] = useState('');
  const [urlValue, setUrlValue] = useState('');
  const [urlError, setUrlError] = useState('');

  // Notify parent whenever the source changes
  const notifyParent = useCallback((type, file = null, url = '') => {
    if (type === 'file' && file) {
      onSourceReady({ type: 'file', file });
    } else if (type === 'url' && url.trim()) {
      onSourceReady({ type: 'url', url: url.trim() });
    } else {
      onSourceReady(null);
    }
  }, [onSourceReady]);

  // ── File selection ──────────────────────────────────────────────────────────
  const handleFileSelect = useCallback((file) => {
    setFileError('');

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_UPLOAD_EXTS.includes(ext)) {
      setFileError(`Unsupported format "${ext}". Allowed: ${SUPPORTED_UPLOAD_EXTS.join(', ')}`);
      notifyParent(null);
      return;
    }

    setSelectedFile(file);

    // Extract metadata using a temporary <video> element
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = url;
    video.onloadedmetadata = () => {
      setFileMetadata({
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
        previewUrl: url,
      });
    };
    video.onerror = () => {
      setFileMetadata({ previewUrl: url });
    };

    notifyParent('file', file);
  }, [notifyParent]);

  const handleFileClear = useCallback(() => {
    if (fileMetadata.previewUrl) {
      URL.revokeObjectURL(fileMetadata.previewUrl);
    }
    setSelectedFile(null);
    setFileMetadata({});
    setFileError('');
    notifyParent(null);
  }, [fileMetadata, notifyParent]);

  // ── URL input ───────────────────────────────────────────────────────────────
  const handleUrlChange = useCallback((val) => {
    setUrlValue(val);
    setUrlError('');

    if (!val.trim()) {
      notifyParent(null);
      return;
    }

    // Basic URL validation
    try {
      new URL(val.trim());
    } catch {
      setUrlError('Invalid URL format.');
      notifyParent(null);
      return;
    }

    const source = detectUrlSource(val.trim());
    if (!source) {
      setUrlError('Unsupported URL source. See supported sources below.');
      notifyParent(null);
      return;
    }

    notifyParent('url', null, val);
  }, [notifyParent]);

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      if (fileMetadata.previewUrl) URL.revokeObjectURL(fileMetadata.previewUrl);
    };
  }, [fileMetadata.previewUrl]);

  // Tab switch clears the other input
  const switchTab = useCallback((tab) => {
    if (tab === activeTab) return;
    setActiveTab(tab);
    if (tab === 'upload') {
      setUrlValue('');
      setUrlError('');
      notifyParent(selectedFile ? 'file' : null, selectedFile);
    } else {
      handleFileClear();
      notifyParent(null);
    }
  }, [activeTab, selectedFile, handleFileClear, notifyParent]);

  const displayError = fileError || urlError || externalError || '';

  return (
    <div className="space-y-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Video Source</p>

      {/* ── Tab switcher ── */}
      <div className="flex rounded-xl border border-surface-600 bg-surface-800 p-1 gap-1">
        <button
          id="tab-upload"
          type="button"
          onClick={() => switchTab('upload')}
          disabled={disabled}
          className={`
            flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold
            transition-all duration-200
            ${activeTab === 'upload'
              ? 'bg-brand-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-surface-700'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <UploadIcon className="w-4 h-4" />
          Upload Video
        </button>
        <button
          id="tab-url"
          type="button"
          onClick={() => switchTab('url')}
          disabled={disabled}
          className={`
            flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold
            transition-all duration-200
            ${activeTab === 'url'
              ? 'bg-brand-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-surface-700'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <LinkIcon className="w-4 h-4" />
          Video URL
        </button>
      </div>

      {/* ── Tab content ── */}
      <div className="animate-fade-in">
        {activeTab === 'upload' ? (
          <div>
            {!selectedFile ? (
              <UploadZone onFileSelect={handleFileSelect} disabled={disabled} />
            ) : (
              <FilePreview
                file={selectedFile}
                metadata={fileMetadata}
                onClear={handleFileClear}
              />
            )}
          </div>
        ) : (
          <UrlInput
            value={urlValue}
            onChange={handleUrlChange}
            error={urlError || externalError}
            disabled={disabled}
          />
        )}
      </div>

      {/* ── Error message ── */}
      {displayError && (
        <p id="input-error-msg" role="alert" className="text-xs text-rose-400 flex items-start gap-1.5 animate-fade-in">
          <svg className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          {displayError}
        </p>
      )}
    </div>
  );
}

export default VideoInputPanel;
