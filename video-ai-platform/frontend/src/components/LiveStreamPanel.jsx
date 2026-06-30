import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

function LiveStreamPanel({ jobId, features, onStop }) {
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState('Initializing camera...');
  const [error, setError] = useState('');
  
  useEffect(() => {
    let pc = null;
    let localStream = null;

    async function startWebRTC() {
      try {
        // 1. Get user media (camera)
        setStatus('Requesting camera access...');
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, frameRate: 30 },
          audio: false,
        });

        if (localVideoRef.current) {
          localVideoRef.current.srcObject = localStream;
        }

        // 2. Setup RTCPeerConnection
        setStatus('Connecting to AI Server...');
        pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        pcRef.current = pc;

        // Add local tracks
        localStream.getTracks().forEach(track => {
          pc.addTrack(track, localStream);
        });

        // Listen for remote tracks
        pc.ontrack = (event) => {
          if (remoteVideoRef.current && event.streams && event.streams[0]) {
            remoteVideoRef.current.srcObject = event.streams[0];
            setStatus('Live Stream Connected');
          }
        };

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Send offer to FastAPI via Vite proxy
        const response = await axios.post('/rtc/offer', {
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          videoPath: jobId,
          stabilization: features.stabilization,
          heavyRainRemoval: features.heavyRainRemoval,
          videoVisibility: features.videoVisibility,
          distanceEstimation: features.distanceEstimation,
        });

        // Set remote answer
        await pc.setRemoteDescription(response.data);
      } catch (err) {
        console.error('WebRTC Error:', err);
        setError(err.message || 'Failed to start WebRTC session.');
        setStatus('Failed');
      }
    }

    startWebRTC();

    return () => {
      // Cleanup
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
      if (pc) {
        pc.close();
      }
    };
  }, [jobId, features]);

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
          onClick={onStop}
          className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-rose-900/50 text-rose-300 hover:bg-rose-800 hover:text-white transition-colors border border-rose-800"
        >
          Stop Stream
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Local Camera (Input) */}
        <div className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">Camera Input</span>
          <div className="aspect-video bg-black rounded-xl border border-surface-600 overflow-hidden relative">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-contain"
            />
          </div>
        </div>

        {/* Remote Stream (Output) */}
        <div className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-emerald-500">AI Output</span>
          <div className="aspect-video bg-black rounded-xl border border-surface-600 overflow-hidden relative">
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-contain"
            />
            {status !== 'Live Stream Connected' && !error && (
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
    </div>
  );
}

export default LiveStreamPanel;
