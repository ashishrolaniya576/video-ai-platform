const { Server } = require('socket.io');
const logger = require('../utils/logger');
const { setSocketIO } = require('../controllers/streamController');

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
      logger.info(`Client ${socket.id} subscribed to job: ${jobId}`);
      socket.emit('subscribed', { jobId, message: `Listening for updates on job ${jobId}` });
    });

    socket.on('unsubscribe_job', (jobId) => {
      socket.leave(jobId);
      logger.info(`Client ${socket.id} unsubscribed from job: ${jobId}`);
    });

    socket.on('disconnect', () => {
      logger.info(`WebSocket client disconnected: ${socket.id}`);
    });

    socket.on('error', (err) => {
      logger.error(`WebSocket error on client ${socket.id}: ${err.message}`);
    });
  });

  return io;
}

module.exports = { initSocket };
