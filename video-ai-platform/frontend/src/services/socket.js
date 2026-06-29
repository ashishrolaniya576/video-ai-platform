import { io } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || '';

let socket = null;

export function getSocket() {
  if (!socket) {
    socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socket.on('connect', () => {
      console.info(`[Socket] Connected: ${socket.id}`);
    });

    socket.on('disconnect', (reason) => {
      console.info(`[Socket] Disconnected: ${reason}`);
    });

    socket.on('connect_error', (err) => {
      console.error(`[Socket] Connection error: ${err.message}`);
    });
  }
  return socket;
}

export function subscribeToJob(jobId, handlers) {
  const s = getSocket();
  s.emit('subscribe_job', jobId);

  const { onProgress, onCompleted, onFailed, onSubscribed, onConnect, onDisconnect, onConnectError } = handlers;

  if (onSubscribed) s.on('subscribed', onSubscribed);
  if (onProgress) s.on('progress_update', onProgress);
  if (onCompleted) s.on('processing_completed', onCompleted);
  if (onFailed) s.on('processing_failed', onFailed);
  if (onConnect) s.on('connect', onConnect);
  if (onDisconnect) s.on('disconnect', onDisconnect);
  if (onConnectError) s.on('connect_error', onConnectError);

  return () => {
    s.emit('unsubscribe_job', jobId);
    if (onSubscribed) s.off('subscribed', onSubscribed);
    if (onProgress) s.off('progress_update', onProgress);
    if (onCompleted) s.off('processing_completed', onCompleted);
    if (onFailed) s.off('processing_failed', onFailed);
    if (onConnect) s.off('connect', onConnect);
    if (onDisconnect) s.off('disconnect', onDisconnect);
    if (onConnectError) s.off('connect_error', onConnectError);
  };
}

export function cancelJobEvent(jobId) {
  const s = getSocket();
  if (s) {
    s.emit('cancel_job', jobId);
  }
}

export function isSocketConnected() {
  return socket ? socket.connected : false;
}

export function disconnectSocket() {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
}
