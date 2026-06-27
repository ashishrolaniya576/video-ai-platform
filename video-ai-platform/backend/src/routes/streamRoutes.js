const express = require('express');
const router = express.Router();
const { startProcessing, getStatus, getResult, healthCheck } = require('../controllers/streamController');

router.get('/health', healthCheck);
router.post('/process', startProcessing);
router.get('/status/:jobId', getStatus);
router.get('/result/:jobId', getResult);

module.exports = router;
