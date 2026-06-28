/**
 * uploadRoutes.js
 *
 * Routes for two new input methods:
 *   POST /api/upload           — local file upload (multipart/form-data)
 *   POST /api/download-url     — download a video from a URL
 */

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const { downloadVideo, detectSourceType, TEMP_DIR } = require('../utils/videoDownloader');

// ── Multer configuration ──────────────────────────────────────────────────────
const SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
const MAX_FILE_SIZE_MB = parseInt(process.env.MAX_FILE_SIZE_MB || '2048', 10);

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    // Ensure TEMP_DIR exists (it is created in videoDownloader at startup)
    if (!fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
    }
    cb(null, TEMP_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${uuidv4()}${ext}`);
  },
});

function fileFilter(_req, file, cb) {
  const ext = path.extname(file.originalname).toLowerCase();
  if (SUPPORTED_EXTENSIONS.includes(ext)) {
    cb(null, true);
  } else {
    cb(new Error(`Unsupported file type "${ext}". Allowed: ${SUPPORTED_EXTENSIONS.join(', ')}`));
  }
}

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
});

// ── POST /api/upload ──────────────────────────────────────────────────────────
router.post('/upload', (req, res) => {
  upload.single('video')(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
          error: `File too large. Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`,
        });
      }
      return res.status(400).json({ error: `Upload error: ${err.message}` });
    }
    if (err) {
      return res.status(400).json({ error: err.message });
    }
    if (!req.file) {
      return res.status(400).json({ error: 'No video file received.' });
    }

    const { path: filePath, originalname, size } = req.file;
    logger.info(`[Upload] File saved: ${filePath} (${size} bytes, original: ${originalname})`);

    return res.status(200).json({
      tempPath: filePath,
      filename: originalname,
      sizeBytes: size,
      source: 'upload',
    });
  });
});

// ── POST /api/download-url ────────────────────────────────────────────────────
router.post('/download-url', async (req, res) => {
  const { url } = req.body;

  if (!url || typeof url !== 'string' || url.trim() === '') {
    return res.status(400).json({ error: 'url is required and must be a non-empty string.' });
  }

  const trimmedUrl = url.trim();

  // Basic URL format validation
  try {
    new URL(trimmedUrl); // throws if invalid
  } catch {
    return res.status(400).json({ error: 'Invalid URL format. Please provide a valid URL.' });
  }

  const sourceType = detectSourceType(trimmedUrl);
  if (sourceType === 'unknown') {
    return res.status(400).json({
      error:
        'Unsupported URL source. Supported: YouTube, Vimeo, Google Drive, Dropbox, OneDrive, and direct video links (.mp4, .mov, .avi, .mkv, .webm).',
    });
  }

  logger.info(`[DownloadURL] Starting download from ${sourceType}: ${trimmedUrl}`);

  try {
    const result = await downloadVideo(trimmedUrl);
    logger.info(`[DownloadURL] Success: ${result.filePath}`);

    return res.status(200).json({
      tempPath: result.filePath,
      filename: result.filename,
      sizeBytes: result.sizeBytes,
      sourceType: result.sourceType,
      source: 'url',
    });
  } catch (err) {
    logger.error(`[DownloadURL] Failed: ${err.message}`);
    return res.status(500).json({
      error: `Download failed: ${err.message}`,
    });
  }
});

module.exports = router;
