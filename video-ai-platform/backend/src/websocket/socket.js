const { Server } = require('socket.io');
const logger = require('../utils/logger');
const { setSocketIO } = require('../controllers/streamController');
const { cancelJob } = require('../services/aiService');

const disconnectTimers = {};

function initSocket(httpServer) {
  const io = new Server(httpServer, {
    cors: {
      origin: process.env.NODE_ENV === 'production' ? false : '*',
      methods: ['GET', 'POST'],
    },
  });

  setSocketIO(io);

  io.on('connection', (socket) => {
    logger.info(`WebSocket client connected: ${socket.id}`);

    socket.on('subscribe_job', (jobId) => {
      socket.join(jobId);
      socket.currentJobId = jobId;
      logger.info(`Client ${socket.id} subscribed to job: ${jobId}`);
      
      // Clear any pending disconnect timer for this job (client reconnected)
      if (disconnectTimers[jobId]) {
        clearTimeout(disconnectTimers[jobId]);
        delete disconnectTimers[jobId];
        logger.info(`Cleared disconnect timer for job: ${jobId} (Client reconnected)`);
      }
      
      socket.emit('subscribed', { jobId, message: `Listening for updates on job ${jobId}` });
    });

    socket.on('unsubscribe_job', (jobId) => {
      socket.leave(jobId);
      if (socket.currentJobId === jobId) {
        socket.currentJobId = null;
      }
      logger.info(`Client ${socket.id} unsubscribed from job: ${jobId}`);
    });

    socket.on('cancel_job', (jobId) => {
      logger.info(`Client ${socket.id} requested cancellation for job: ${jobId}`);
      cancelJob(jobId);
    });

    socket.on('disconnect', () => {
      logger.info(`WebSocket client disconnected: ${socket.id}`);
      if (socket.currentJobId) {
        const jobId = socket.currentJobId;
        logger.info(`Client disconnected while on job ${jobId}. Starting 10s grace period...`);
        
        disconnectTimers[jobId] = setTimeout(() => {
          logger.info(`Grace period expired for job ${jobId}. Cancelling job...`);
          cancelJob(jobId);
          delete disconnectTimers[jobId];
        }, 10000);
      }
    });

    socket.on('error', (err) => {
      logger.error(`WebSocket error on client ${socket.id}: ${err.message}`);
    });
  });

  return io;
}

module.exports = { initSocket };
