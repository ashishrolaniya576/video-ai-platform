require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const streamRoutes = require('./routes/streamRoutes');
const uploadRoutes = require('./routes/uploadRoutes');
const logger = require('./utils/logger');

// Ensure temp upload directory exists at startup
const TEMP_DIR = path.resolve(
  process.env.TEMP_UPLOAD_DIR || path.join(__dirname, '../uploads/temp')
);
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
  logger.info(`[App] Created temp upload directory: ${TEMP_DIR}`);
}

const app = express();

app.use(cors({ origin: '*' }));
// Increase JSON / URL-encoded body limits for large metadata payloads
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

app.use((req, res, next) => {
  logger.debug(`${req.method} ${req.originalUrl}`);
  next();
});

app.use('/api', streamRoutes);
app.use('/api', uploadRoutes);

// Serve the ai-services output directory statically so the frontend can load the video
const mediaPath = path.join(__dirname, '../../ai-services');
app.use('/api/media', express.static(mediaPath, {
  setHeaders: (res, path) => {
    if (path.endsWith('.mp4')) {
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Cache-Control', 'public, max-age=31536000');
    }
  }
}));

app.use((req, res) => {
  res.status(404).json({ error: 'Route not found.' });
});

app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`, { stack: err.stack });
  res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});

module.exports = app;
