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
async function downloadWithYtDlp(url, outputPath) {
  logger.info(`[Downloader] Using yt-dlp for: ${url}`);

  // yt-dlp options:
  // -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best  → prefer mp4
  // --merge-output-format mp4
  // -o <outputPath>
  const args = [
    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best',
    '--merge-output-format', 'mp4',
    '--no-playlist',
    '--no-warnings',
    '-o', outputPath,
    url,
  ];

  try {
    const ytDlpBin = process.env.YTDLP_BIN || 'yt-dlp';
    const { stdout, stderr } = await execFileAsync(ytDlpBin, args, {
      timeout: 10 * 60 * 1000, // 10 minutes
    });
    if (stdout) logger.debug(`[yt-dlp stdout] ${stdout.trim()}`);
    if (stderr) logger.debug(`[yt-dlp stderr] ${stderr.trim()}`);
  } catch (err) {
    const msg = err.stderr || err.message || 'yt-dlp failed';
    throw new Error(`yt-dlp download failed: ${msg}`);
  }

  if (!fs.existsSync(outputPath)) {
    throw new Error('yt-dlp completed but output file not found.');
  }

  const stat = fs.statSync(outputPath);
  return { filePath: outputPath, sizeBytes: stat.size };
}

// ── Direct / cloud HTTP downloader ───────────────────────────────────────────
async function downloadDirect(url, outputPath) {
  logger.info(`[Downloader] Direct HTTP download: ${url}`);

  const response = await axios.get(url, {
    responseType: 'stream',
    timeout: 10 * 60 * 1000, // 10 minutes
    maxRedirects: 10,
    headers: {
      'User-Agent': 'Mozilla/5.0 (compatible; VideoAIPlatform/1.0)',
    },
  });

  const contentType = response.headers['content-type'] || '';
  const isVideo = /video|octet-stream/.test(contentType);
  if (!isVideo) {
    logger.warn(`[Downloader] Unexpected content-type: ${contentType}`);
  }

  // Determine extension from content-disposition or URL
  let ext = '.mp4';
  const disposition = response.headers['content-disposition'] || '';
  const dispMatch = disposition.match(/filename[^;=\n]*=["']?([^"';\n]+)/i);
  if (dispMatch) {
    ext = path.extname(dispMatch[1]) || ext;
  } else {
    const urlExt = path.extname(url.split('?')[0]);
    if (urlExt) ext = urlExt;
  }

  // If outputPath has no extension, append one
  const finalPath = outputPath.endsWith('.mp4') || path.extname(outputPath)
    ? outputPath
    : outputPath + ext;

  await new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(finalPath);
    response.data.pipe(writer);
    writer.on('finish', resolve);
    writer.on('error', reject);
    response.data.on('error', reject);
  });

  const stat = fs.statSync(finalPath);
  return { filePath: finalPath, sizeBytes: stat.size };
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
    result = await downloadWithYtDlp(url, outputPath);
  } else {
    const normalizedUrl = normalizeUrl(url, sourceType);
    result = await downloadDirect(normalizedUrl, outputPath);
  }

  const filename = path.basename(result.filePath);
  logger.info(`[Downloader] Download complete: ${result.filePath} (${result.sizeBytes} bytes)`);

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
