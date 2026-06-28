import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.message ||
      'An unexpected error occurred.';
    return Promise.reject(new Error(message));
  }
);

// ── Existing endpoints ────────────────────────────────────────────────────────

/**
 * Start processing using a pre-saved temp path (from upload or URL download).
 */
export async function startProcessing({ tempPath, videoUrl, stabilization, heavyRainRemoval, videoVisibility, objectDetection }) {
  const response = await api.post('/process', {
    tempPath,
    videoUrl,
    stabilization,
    heavyRainRemoval,
    videoVisibility,
    objectDetection,
  });
  return response.data;
}

export async function getJobStatus(jobId) {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
}

export async function getJobResult(jobId) {
  const response = await api.get(`/result/${jobId}`);
  return response.data;
}

export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

// ── New input endpoints ───────────────────────────────────────────────────────

/**
 * Upload a local video file via multipart/form-data.
 *
 * @param {File} file - The File object to upload.
 * @param {function} [onProgress] - Optional progress callback (0–100).
 * @returns {Promise<{ tempPath: string, filename: string, sizeBytes: number, source: string }>}
 */
export async function uploadVideoFile(file, onProgress) {
  const formData = new FormData();
  formData.append('video', file);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30 * 60 * 1000, // 30 min for large files
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(pct);
      }
    },
  });

  if (!response.data?.tempPath) {
    throw new Error('Upload succeeded but no tempPath was returned.');
  }

  return response.data;
}

/**
 * Request the backend to download a video from a URL (YouTube, Vimeo, direct links, etc.).
 *
 * @param {string} url - The video source URL.
 * @returns {Promise<{ tempPath: string, filename: string, sizeBytes: number, sourceType: string, source: string }>}
 */
export async function downloadVideoUrl(url) {
  const response = await api.post(
    '/download-url',
    { url },
    {
      timeout: 15 * 60 * 1000, // 15 min for large remote videos
    }
  );

  if (!response.data?.tempPath) {
    throw new Error('Download succeeded but no tempPath was returned.');
  }

  return response.data;
}
