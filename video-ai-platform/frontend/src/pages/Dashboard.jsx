import { useState, useCallback } from 'react';
import FeaturePanel from '../components/FeaturePanel.jsx';
import ProgressBar from '../components/ProgressBar.jsx';
import StatusCard from '../components/StatusCard.jsx';
import LogsPanel from '../components/LogsPanel.jsx';
import OutputPanel from '../components/OutputPanel.jsx';
import VideoInputPanel from '../components/VideoInputPanel.jsx';
import { startProcessing, uploadVideoFile, downloadVideoUrl, getJobStatus, getJobResult } from '../services/api.js';
import { subscribeToJob } from '../services/socket.js';

const INITIAL_FEATURES = { stabilization: false, heavyRainRemoval: false, videoVisibility: false, distanceEstimation: false };

// Maps feature key → human-readable pipeline label (ordered for display)
const PIPELINE_LABELS = [
  { key: 'stabilization',    label: 'Video Stabilization' },
  { key: 'heavyRainRemoval', label: 'Heavy Rain Removal' },
  { key: 'videoVisibility',  label: 'Video Visibility Enhancement' },
  { key: 'distanceEstimation',  label: 'Distance Estimation' },
];

function createLog(message, level = 'info') {
  const now = new Date();
  const timestamp = now.toLocaleTimeString('en-US', { hour12: false });
  return { timestamp, message, level };
}

function Dashboard() {
  // ── Input source state ────────────────────────────────────────────────────
  // inputSource: null | { type: 'file', file: File } | { type: 'url', url: string }
  const [inputSource, setInputSource] = useState(null);

  // ── Processing state ──────────────────────────────────────────────────────
  const [features, setFeatures] = useState(INITIAL_FEATURES);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [logs, setLogs] = useState([]);
  const [outputVideo, setOutputVideo] = useState(null);
  const [detectionSummary, setDetectionSummary] = useState(null);

  // ── Validation error state ─────────────────────────────────────────────────
  const [sourceError, setSourceError] = useState('');
  const [featureError, setFeatureError] = useState('');
  const [apiError, setApiError] = useState('');

  const appendLog = useCallback((message, level = 'info') => {
    setLogs((prev) => [...prev, createLog(message, level)]);
  }, []);

  const validate = useCallback(() => {
    let valid = true;

    if (!inputSource) {
      setSourceError('Please select a video file or enter a video URL.');
      valid = false;
    } else {
      setSourceError('');
    }

    const hasFeature = Object.values(features).some(Boolean);
    if (!hasFeature) {
      setFeatureError('Select at least one processing feature.');
      valid = false;
    } else {
      setFeatureError('');
    }

    return valid;
  }, [inputSource, features]);

  const resetState = useCallback(() => {
    setProgress(0);
    setCurrentStage('');
    setLogs([]);
    setOutputVideo(null);
    setDetectionSummary(null);
    setApiError('');
    setStatus('idle');
    setJobId(null);
  }, []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!validate()) return;

    resetState();
    setProcessing(true);
    setStatus('accepted');

    let tempPath = null;

    try {
      // ── Step 1: Upload file or download URL ─────────────────────────────
      if (inputSource.type === 'file') {
        appendLog(`Uploading "${inputSource.file.name}"…`, 'info');
        const uploadResult = await uploadVideoFile(inputSource.file, (pct) => {
          if (pct < 100) appendLog(`Upload progress: ${pct}%`, 'info');
        });
        tempPath = uploadResult.tempPath;
        appendLog(`File uploaded successfully (${(uploadResult.sizeBytes / (1024 * 1024)).toFixed(1)} MB).`, 'success');
      } else {
        appendLog(`Downloading video from URL…`, 'info');
        appendLog(`Source: ${inputSource.url}`, 'system');
        const downloadResult = await downloadVideoUrl(inputSource.url);
        tempPath = downloadResult.tempPath;
        appendLog(`Video downloaded (${downloadResult.sourceType}, ${(downloadResult.sizeBytes / (1024 * 1024)).toFixed(1)} MB).`, 'success');
      }

      // ── Step 2: Submit processing job ───────────────────────────────────
      appendLog('Submitting job to AI pipeline…', 'info');
      const data = await startProcessing({ tempPath, ...features });
      const newJobId = data.jobId;
      setJobId(newJobId);
      appendLog(`Job accepted. ID: ${newJobId}`, 'success');
      appendLog('Connecting to processing pipeline…', 'stage');

      const unsubscribe = subscribeToJob(newJobId, {
        onSubscribed: ({ message }) => {
          appendLog(message, 'system');
          setStatus('processing');
        },
        onProgress: (payload) => {
          if (payload.jobId !== newJobId) return;
          setProgress(payload.progress);
          setCurrentStage(payload.currentStage);
          setStatus(payload.status === 'completed' ? 'completed' : 'processing');
          const level = payload.currentStage.toLowerCase().includes('running') ? 'stage' : 'info';
          appendLog(`${payload.currentStage} — ${payload.progress}%`, level);
        },
        onCompleted: (payload) => {
          if (payload.jobId !== newJobId) return;
          setStatus('completed');
          setProgress(100);
          setCurrentStage('Completed');
          setOutputVideo(payload.outputVideo);
          if (payload.detectionSummary && Object.keys(payload.detectionSummary).length > 0) {
            setDetectionSummary(payload.detectionSummary);
          }
          appendLog('Processing completed successfully.', 'success');
          setProcessing(false);
          unsubscribe();
        },
        onFailed: (payload) => {
          if (payload.jobId !== newJobId) return;
          setStatus('failed');
          setApiError(payload.error || 'An unknown error occurred.');
          appendLog(`Error: ${payload.error || 'Processing failed.'}`, 'error');
          setProcessing(false);
          unsubscribe();
        },
      });

      // Fallback Polling Mechanism
      // Safeguard in case WebSockets fail to connect, drop, or miss events due to race conditions.
      let pollInterval = setInterval(async () => {
        try {
          const res = await getJobStatus(newJobId);
          if (res.status === 'completed') {
            clearInterval(pollInterval);
            const resultRes = await getJobResult(newJobId);
            setStatus('completed');
            setProgress(100);
            setCurrentStage('Completed');
            setOutputVideo(resultRes.outputVideo);
            if (resultRes.detectionSummary && Object.keys(resultRes.detectionSummary).length > 0) {
              setDetectionSummary(resultRes.detectionSummary);
            }
            appendLog('Processing completed successfully.', 'success');
            setProcessing(false);
            unsubscribe();
          } else if (res.status === 'failed') {
            clearInterval(pollInterval);
            setStatus('failed');
            setApiError('Processing failed.');
            appendLog(`Error: Processing failed on backend.`, 'error');
            setProcessing(false);
            unsubscribe();
          } else {
            // Update progress only if it's strictly greater than what we already have
            setProgress((prev) => Math.max(prev, res.progress || 0));
            if (res.currentStage) setCurrentStage(res.currentStage);
            if (res.status) setStatus(res.status);
          }
        } catch (pollErr) {
          // Ignore polling errors to prevent interrupting the flow if a single request fails
        }
      }, 2000);

      // Clean up polling if the user resets
      window.__currentJobPoll = pollInterval;

    } catch (err) {
      setStatus('failed');
      setApiError(err.message);
      appendLog(`Failed: ${err.message}`, 'error');
      setProcessing(false);
    }
  }, [inputSource, features, validate, resetState, appendLog]);

  const handleReset = useCallback(() => {
    if (window.__currentJobPoll) {
      clearInterval(window.__currentJobPoll);
      window.__currentJobPoll = null;
    }
    resetState();
    setInputSource(null);
    setFeatures(INITIAL_FEATURES);
    setSourceError('');
    setFeatureError('');
    setProcessing(false);
  }, [resetState]);

  const isSubmitDisabled = processing;

  // Build ordered pipeline steps based on active toggles
  const activePipelineSteps = PIPELINE_LABELS.filter(({ key }) => features[key]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8 animate-fade-in">
        <h1 className="text-3xl font-extrabold tracking-tight">
          <span className="gradient-text">AI Video Processing</span>{' '}
          <span className="text-slate-300">Dashboard</span>
        </h1>
        <p className="mt-2 text-slate-400 text-base">
          Process recorded videos using AI-powered video stabilization, heavy rain removal, visibility enhancement, and object detection.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* ── Left Panel — Input & Config ── */}
        <div className="xl:col-span-2 space-y-5">
          <form onSubmit={handleSubmit} noValidate id="processing-form">
            <div className="card-glow p-6 space-y-6">

              {/* ── Video Source Input ── */}
              <VideoInputPanel
                onSourceReady={setInputSource}
                disabled={isSubmitDisabled}
                externalError={sourceError}
              />

              {/* ── Feature Selection ── */}
              <FeaturePanel
                features={features}
                onChange={setFeatures}
                disabled={isSubmitDisabled}
              />
              {featureError && (
                <p className="text-xs text-rose-400 flex items-center gap-1">
                  <svg className="w-3.5 h-3.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  {featureError}
                </p>
              )}

              {/* ── API Error ── */}
              {apiError && (
                <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-800 text-xs text-rose-300">
                  <strong>Error:</strong> {apiError}
                </div>
              )}

              {/* ── Dynamic Pipeline Preview ── */}
              {activePipelineSteps.length > 0 && (
                <div className="space-y-2">
                  <p className="section-label">Pipeline Preview</p>
                  <div className="flex flex-col gap-1.5">
                    {/* Input node */}
                    <div className="flex items-center gap-2">
                      <span className="badge bg-surface-700 border-surface-500 text-slate-400">
                        {inputSource?.type === 'file'
                          ? `📁 ${inputSource.file.name.length > 20 ? inputSource.file.name.slice(0, 17) + '…' : inputSource.file.name}`
                          : inputSource?.type === 'url'
                          ? '🔗 Video URL'
                          : 'Input'
                        }
                      </span>
                    </div>
                    {activePipelineSteps.map(({ key, label }) => (
                      <div key={key} className="flex flex-col gap-1.5">
                        {/* Arrow */}
                        <div className="flex items-center pl-3">
                          <svg className="w-3 h-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                        <span className="badge badge-processing">{label}</span>
                      </div>
                    ))}
                    {/* Arrow + Output */}
                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center pl-3">
                        <svg className="w-3 h-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                      <span className="badge bg-emerald-900/50 text-emerald-300 border-emerald-800">Output</span>
                    </div>
                  </div>
                </div>
              )}

              {/* ── Action Buttons ── */}
              <div className="flex gap-3 pt-2">
                <button
                  id="start-processing-btn"
                  type="submit"
                  disabled={isSubmitDisabled}
                  className="btn-primary flex-1"
                >
                  {processing ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing…
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 3l14 9-14 9V3z" />
                      </svg>
                      Process Video
                    </>
                  )}
                </button>

                {(status !== 'idle') && (
                  <button
                    id="reset-btn"
                    type="button"
                    onClick={handleReset}
                    disabled={processing}
                    className="btn-secondary"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Reset
                  </button>
                )}
              </div>
            </div>
          </form>
        </div>

        {/* ── Right Panel — Status & Output ── */}
        <div className="xl:col-span-3 space-y-5">
          {/* Processing Status */}
          <div className="card-glow p-6 space-y-4 animate-slide-up">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
              <svg className="w-4 h-4 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Processing Status
            </h2>

            <StatusCard status={status} currentStage={currentStage} jobId={jobId} />
            <ProgressBar progress={progress} animated={processing} />
            <LogsPanel logs={logs} />
          </div>

          {/* Output Section */}
          <div className="card-glow p-6 animate-slide-up">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-4">
              <svg className="w-4 h-4 text-accent-purple" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
              </svg>
              Output
            </h2>
            <OutputPanel outputVideo={outputVideo} detectionSummary={detectionSummary} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
