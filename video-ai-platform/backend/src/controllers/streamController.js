const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const { createJob, getJob, processJob } = require('../services/aiService');
const { cleanupTempFile } = require('../utils/videoDownloader');

let io;

function setSocketIO(socketIO) {
  io = socketIO;
}

function emitProgress(jobId, data) {
  if (!io) return;

  if (data.status === 'completed') {
    const job = getJob(jobId);
    const payload = {
      ...data,
      outputVideo: job ? job.outputVideo : null,
      detectionSummary: data.detectionSummary || (job ? job.detectionSummary : null) || null,
    };
    io.to(jobId).emit('progress_update', payload);
    io.to(jobId).emit('processing_completed', payload);
  } else if (data.status === 'failed') {
    io.to(jobId).emit('processing_failed', data);
  } else {
    io.to(jobId).emit('progress_update', data);
  }
}

async function startProcessing(req, res, next) {
  try {
    // Accept either a pre-saved local path (from /upload or /download-url)
    // or a raw videoUrl for legacy compatibility.
    const {
      tempPath,
      videoUrl,
      stabilization,
      heavyRainRemoval,
      videoVisibility,
      objectDetection,
    } = req.body;

    const videoSource = (tempPath || videoUrl || '').trim();
    const isTempFile = Boolean(tempPath && tempPath.trim());

    if (isTempFile) {
      const path = require('path');
      const { TEMP_DIR } = require('../utils/videoDownloader');
      const resolvedPath = path.resolve(videoSource);
      if (!resolvedPath.startsWith(TEMP_DIR)) {
        return res.status(403).json({ error: 'Invalid tempPath. Path traversal is not allowed.' });
      }
    }

    logger.info(
      `Request received: POST /api/process — source: ${isTempFile ? `tempPath:${tempPath}` : `url:${videoUrl}`}`
    );

    if (!videoSource) {
      return res.status(400).json({
        error: 'Either tempPath (from upload/download) or videoUrl is required.',
      });
    }

    const hasFeature = stabilization || heavyRainRemoval || videoVisibility || objectDetection;
    if (!hasFeature) {
      return res.status(400).json({ error: 'At least one processing feature must be selected.' });
    }

    logger.info('Validation passed.');

    const jobId = uuidv4();
    const features = {
      stabilization: Boolean(stabilization),
      heavyRainRemoval: Boolean(heavyRainRemoval),
      videoVisibility: Boolean(videoVisibility),
      objectDetection: Boolean(objectDetection),
    };

    createJob(jobId, videoSource, features, isTempFile);
    logger.info(`Job created: ${jobId}`);

    res.status(202).json({ jobId, status: 'accepted' });

    logger.info(`Calling AI service for job: ${jobId}`);

    processJob(jobId, videoSource, features, (jid, progressData) => {
      emitProgress(jid, progressData);
    })
      .then(() => {
        // Clean up temp file after successful processing
        if (isTempFile) cleanupTempFile(videoSource);
      })
      .catch((err) => {
        logger.error(`Job ${jobId} failed: ${err.message}`);
        emitProgress(jobId, { jobId, status: 'failed', error: err.message });
        // Clean up temp file even on failure
        if (isTempFile) cleanupTempFile(videoSource);
      });
  } catch (err) {
    next(err);
  }
}

async function getStatus(req, res, next) {
  try {
    const { jobId } = req.params;
    logger.info(`Request received: GET /api/status/${jobId}`);

    const job = getJob(jobId);
    if (!job) {
      return res.status(404).json({ error: `Job ${jobId} not found.` });
    }

    res.json({
      status: job.status,
      progress: job.progress,
      currentStage: job.currentStage,
    });
  } catch (err) {
    next(err);
  }
}

async function getResult(req, res, next) {
  try {
    const { jobId } = req.params;
    logger.info(`Request received: GET /api/result/${jobId}`);

    const job = getJob(jobId);
    if (!job) {
      return res.status(404).json({ error: `Job ${jobId} not found.` });
    }

    if (job.status !== 'completed') {
      return res.status(202).json({ status: job.status, message: 'Processing not yet completed.' });
    }

    res.json({
      status: 'completed',
      outputVideo: job.outputVideo,
      detectionSummary: job.detectionSummary || null,
    });
  } catch (err) {
    next(err);
  }
}

async function healthCheck(req, res) {
  res.json({ status: 'running', timestamp: new Date().toISOString() });
}

module.exports = { startProcessing, getStatus, getResult, healthCheck, setSocketIO };
