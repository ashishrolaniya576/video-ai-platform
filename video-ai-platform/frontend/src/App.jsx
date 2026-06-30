import { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import Navbar from './components/Navbar.jsx';
import Dashboard from './pages/Dashboard.jsx';

function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [backendStateInfo, setBackendStateInfo] = useState(null);

  useEffect(() => {
    let interval;
    const checkReady = async () => {
      try {
        const res = await axios.get('/ready');
        setBackendReady(res.data.ready);
        setBackendStateInfo(res.data);
        if (res.data.ready && interval) {
          clearInterval(interval);
        }
      } catch (err) {
        if (err.response?.data) {
          setBackendReady(false);
          setBackendStateInfo(err.response.data);
        }
      }
    };
    checkReady();
    interval = setInterval(checkReady, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar backendReady={backendReady} />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Dashboard backendReady={backendReady} backendStateInfo={backendStateInfo} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
