require('dotenv').config();
const express = require('express');
const cors = require('cors');
const streamRoutes = require('./routes/streamRoutes');
const logger = require('./utils/logger');

const app = express();

app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
  logger.debug(`${req.method} ${req.originalUrl}`);
  next();
});

app.use('/api', streamRoutes);

// Serve the ai-services output directory statically so the frontend can load the video
const path = require('path');
const mediaPath = path.join(__dirname, '../../../ai-services');
app.use('/api/media', express.static(mediaPath));

app.use((req, res) => {
  res.status(404).json({ error: 'Route not found.' });
});

app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`, { stack: err.stack });
  res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});

module.exports = app;
