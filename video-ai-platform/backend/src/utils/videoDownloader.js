/**
 * videoDownloader.js
 *
 * Downloads a video from a given URL to a local temp directory.
 * - YouTube / Vimeo → uses yt-dlp (system command)
 * - Direct video URLs (mp4, mov, avi, mkv, webm) → streams via axios
 * - Google Drive / Dropbox / OneDrive public → attempts direct download
 */

const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const { promisify } = require('util');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const logger = require('./logger');

const execFileAsync = promisify(execFile);

// ── Temp directory ────────────────────────────────────────────────────────────
const TEMP_DIR = path.resolve(
  process.env.TEMP_UPLOAD_DIR ||
  path.join(__dirname, '../../uploads/temp')
);

// Ensure temp directory exists at module load time
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
  logger.info(`[Downloader] Created temp directory: ${TEMP_DIR}`);
}

// ── Source detection ──────────────────────────────────────────────────────────
const YOUTUBE_REGEX = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/)|youtu\.be\/)/i;
const VIMEO_REGEX = /^(https?:\/\/)?(www\.)?vimeo\.com\//i;
const GDRIVE_REGEX = /^(https?:\/\/)?(drive|docs)\.google\.com\//i;
const DROPBOX_REGEX = /^(https?:\/\/)?(www\.)?dropbox\.com\//i;
const ONEDRIVE_REGEX = /^(https?:\/\/)?(onedrive\.live\.com|1drv\.ms)\//i;
const DIRECT_VIDEO_REGEX = /\.(mp4|mov|avi|mkv|webm)(\?.*)?$/i;

/**
 * Detect the type of video source.
 * @param {string} url
 * @returns {'youtube'|'vimeo'|'gdrive'|'dropbox'|'onedrive'|'direct'|'unknown'}
 */
function detectSourceType(url) {
  if (YOUTUBE_REGEX.test(url)) return 'youtube';
  if (VIMEO_REGEX.test(url)) return 'vimeo';
  if (GDRIVE_REGEX.test(url)) return 'gdrive';
  if (DROPBOX_REGEX.test(url)) return 'dropbox';
  if (ONEDRIVE_REGEX.test(url)) return 'onedrive';
  if (DIRECT_VIDEO_REGEX.test(url)) return 'direct';
  return 'unknown';
}

// ── Normalize cloud share URLs to direct download ─────────────────────────────
function normalizeUrl(url, sourceType) {
  if (sourceType === 'gdrive') {
    // https://drive.google.com/file/d/FILE_ID/view → direct download
    const match = url.match(/\/file\/d\/([^/]+)/);
    if (match) {
      return `https://drive.google.com/uc?export=download&id=${match[1]}`;
    }
  }
  if (sourceType === 'dropbox') {
    // ?dl=0 → ?dl=1
    return url.replace(/[?&]dl=0/, '').replace(/dropbox\.com/, 'dl.dropboxusercontent.com');
  }
  if (sourceType === 'onedrive') {
    // 1drv.ms short links cannot be reliably transformed without a redirect;
    // we let axios follow redirects naturally.
    return url;
  }
  return url;
}

// ── yt-dlp downloader (YouTube / Vimeo) ──────────────────────────────────────
async function downloadWithYtDlp(url) {
  logger.info(`[Downloader] Using yt-dlp to extract stream URL for: ${url}`);

  const args = [
    '-f', 'bestvideo[ext=mp4]/best[ext=mp4]/best',
    '--no-playlist',
    '--no-warnings',
    '-g', // Get URL instead of downloading
    url,
  ];

  let streamUrl = '';
  try {
    const ytDlpBin = process.env.YTDLP_BIN || 'yt-dlp';
    const { stdout, stderr } = await execFileAsync(ytDlpBin, args, {
      timeout: 2 * 60 * 1000, // 2 minutes should be plenty for just resolving URL
    });
    if (stdout) {
      streamUrl = stdout.trim().split('\n')[0]; // Take the first URL (video)
    }
    if (stderr) logger.debug(`[yt-dlp stderr] ${stderr.trim()}`);
  } catch (err) {
    const msg = err.stderr || err.message || 'yt-dlp failed';
    throw new Error(`yt-dlp URL extraction failed: ${msg}`);
  }

  if (!streamUrl || !streamUrl.startsWith('http')) {
    throw new Error('yt-dlp completed but no valid stream URL was extracted.');
  }

  // We don't have local sizeBytes since it's a stream
  return { filePath: streamUrl, sizeBytes: 0 };
}

// ── Direct / cloud HTTP downloader ───────────────────────────────────────────
async function downloadDirect(url) {
  logger.info(`[Downloader] Resolving direct HTTP stream: ${url}`);
  
  // No need to actually download the file to disk!
  // OpenCV/FFmpeg can stream this directly.
  return { filePath: url, sizeBytes: 0 };
}

// ── Public API ────────────────────────────────────────────────────────────────
/**
 * Download a video from a URL to a local temp file.
 *
 * @param {string} url - The source URL
 * @returns {Promise<{ filePath: string, sizeBytes: number, sourceType: string, filename: string }>}
 */
async function downloadVideo(url) {
  if (!url || typeof url !== 'string') {
    throw new Error('A valid URL string is required.');
  }

  const sourceType = detectSourceType(url);
  logger.info(`[Downloader] Source type detected: ${sourceType} — ${url}`);

  if (sourceType === 'unknown') {
    throw new Error(
      'Unsupported URL. Supported sources: YouTube, Vimeo, Google Drive, Dropbox, OneDrive, and direct video links (.mp4, .mov, .avi, .mkv, .webm).'
    );
  }

  const uid = uuidv4();
  const usesYtDlp = sourceType === 'youtube' || sourceType === 'vimeo';

  // For yt-dlp, we provide a template path (it appends extension automatically)
  const outputPath = usesYtDlp
    ? path.join(TEMP_DIR, `${uid}.mp4`)
    : path.join(TEMP_DIR, `${uid}`);

  let result;
  if (usesYtDlp) {
    result = await downloadWithYtDlp(url);
  } else {
    const normalizedUrl = normalizeUrl(url, sourceType);
    result = await downloadDirect(normalizedUrl);
  }

  // If it's an HTTP stream, we don't have a strict filename, just use "stream.mp4"
  const filename = result.filePath.startsWith('http') ? 'stream.mp4' : path.basename(result.filePath);
  logger.info(`[Downloader] Stream URL resolved successfully!`);

  return {
    filePath: result.filePath,
    sizeBytes: result.sizeBytes,
    sourceType,
    filename,
  };
}

/**
 * Delete a temp file safely (no-throw).
 * @param {string} filePath
 */
function cleanupTempFile(filePath) {
  if (!filePath) return;
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      logger.info(`[Downloader] Cleaned up temp file: ${filePath}`);
    }
  } catch (err) {
    logger.warn(`[Downloader] Failed to clean up ${filePath}: ${err.message}`);
  }
}

module.exports = { downloadVideo, cleanupTempFile, detectSourceType, TEMP_DIR };
