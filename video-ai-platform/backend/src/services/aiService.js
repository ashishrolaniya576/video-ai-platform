const axios = require('axios');
const logger = require('../utils/logger');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

const jobs = new Map();

function createJob(jobId, videoSource, features, isTempFile = false) {
  const job = {
    jobId,
    videoUrl: videoSource,
    isTempFile,
    features,
    status: 'accepted',
    progress: 0,
    currentStage: 'Queued',
    outputVideo: null,
    detectionSummary: null,
    createdAt: new Date().toISOString(),
    completedAt: null,
  };
  jobs.set(jobId, job);
  logger.info(`[Job ${jobId}] Created — source: ${isTempFile ? 'localFile' : 'url'} — ${videoSource}`);
  return job;
}

function getJob(jobId) {
  return jobs.get(jobId) || null;
}

async function processJob(jobId, videoUrl, features, emitProgress) {
  const job = jobs.get(jobId);
  if (!job) return;

  try {
    // 1. Emit Processing Started
    job.status = 'processing';
    job.progress = 10;
    job.currentStage = 'Running AI Pipeline on FastAPI';

    emitProgress(jobId, {
      jobId,
      status: job.status,
      progress: job.progress,
      currentStage: job.currentStage,
    });

    // Log individual stage starts
    if (features.stabilization) {
      logger.info(`[Job ${jobId}] Video Stabilization Started...`);
    }
    if (features.heavyRainRemoval) {
      logger.info(`[Job ${jobId}] Heavy Rain Removal Started...`);
    }
    if (features.videoVisibility) {
      logger.info(`[Job ${jobId}] Video Visibility Enhancement Started...`);
    }
    if (features.distanceEstimation) {
      logger.info(`[Job ${jobId}] Distance Estimation Started...`);
    }

    logger.info(`[Job ${jobId}] Calling FastAPI at ${AI_SERVICE_URL}/process`);

    // 2. Call FastAPI — send the feature flags directly as the API expects
    const payload = {
      videoPath: videoUrl,
      stabilization: Boolean(features.stabilization),
      heavyRainRemoval: Boolean(features.heavyRainRemoval),
      videoVisibility: Boolean(features.videoVisibility),
      distanceEstimation: Boolean(features.distanceEstimation),
    };

    // Polling mechanism to fetch real-time frame progress from FastAPI
    const progressInterval = setInterval(async () => {
      try {
        const res = await axios.get(`${AI_SERVICE_URL}/progress?video_path=${encodeURIComponent(videoUrl)}`);
        if (res.data && res.data.progress) {
          job.progress = res.data.progress;
          job.currentStage = 'Processing on GPU';
          emitProgress(jobId, {
            jobId,
            status: job.status,
            progress: job.progress,
            currentStage: job.currentStage,
          });
        }
      } catch (e) {
        // Ignore polling errors
      }
    }, 1000);

    let response;
    try {
      response = await axios.post(`${AI_SERVICE_URL}/process`, payload, {
        timeout: 30 * 60 * 1000, // 30 minutes — large videos can take significant time
      });
    } finally {
      clearInterval(progressInterval);
    }
    
    const data = response.data;

    if (data.status !== 'completed') {
      throw new Error(data.error || 'Pipeline failed in FastAPI');
    }

    // 3. Log stage completions
    if (features.stabilization) {
      logger.info(`[Job ${jobId}] Video Stabilization Finished...`);
    }
    if (features.heavyRainRemoval) {
      logger.info(`[Job ${jobId}] Heavy Rain Removal Finished...`);
    }
    if (features.videoVisibility) {
      logger.info(`[Job ${jobId}] Video Visibility Enhancement Finished...`);
    }
    if (features.distanceEstimation) {
      logger.info(`[Job ${jobId}] Distance Estimation Finished...`);
    }

    // 4. Build output URL
    job.status = 'completed';
    job.progress = 100;
    job.currentStage = 'Completed';
    job.completedAt = new Date().toISOString();

    // Store detection summary if returned by FastAPI
    if (data.detectionSummary && Object.keys(data.detectionSummary).length > 0) {
      job.detectionSummary = data.detectionSummary;
      logger.info(`[Job ${jobId}] Detection summary received: ${JSON.stringify(data.detectionSummary)}`);
    }

    // The FastAPI outputVideo is a local path like "output/heavy_rain_test.mp4".
    // We construct a URL the frontend can use to stream it.
    if (data.outputVideo) {
      // e.g. "output/video.mp4" -> "/api/media/output/video.mp4"
      const parts = data.outputVideo.split('output/');
      const filename = parts.length > 1 ? parts[1] : data.outputVideo;
      job.outputVideo = `/api/media/output/${filename}`;
    }

    logger.info(`[Job ${jobId}] Processing completed. Output: ${job.outputVideo}`);

    emitProgress(jobId, {
      jobId,
      status: job.status,
      progress: job.progress,
      currentStage: job.currentStage,
      detectionSummary: job.detectionSummary || null,
    });

    return job;

  } catch (error) {
    const errMsg = error.response?.data?.detail || error.message || 'Unknown error during AI processing';
    logger.error(`[Job ${jobId}] Processing failed: ${errMsg}`);

    job.status = 'failed';
    job.currentStage = 'Failed';
    job.completedAt = new Date().toISOString();

    emitProgress(jobId, {
      jobId,
      status: job.status,
      progress: job.progress,
      currentStage: job.currentStage,
      error: errMsg
    });
  }
}

module.exports = { createJob, getJob, processJob };
