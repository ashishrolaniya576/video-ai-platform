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

export async function startProcessing({ videoUrl, stabilization, heavyRainRemoval, videoVisibility, objectDetection }) {
  const response = await api.post('/process', {
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
