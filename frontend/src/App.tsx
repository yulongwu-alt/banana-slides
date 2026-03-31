import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Home } from './pages/Home';
import { History } from './pages/History';
import { OutlineEditor } from './pages/OutlineEditor';
import { DetailEditor } from './pages/DetailEditor';
import { SlidePreview } from './pages/SlidePreview';
import { SettingsPage } from './pages/Settings';
import { useProjectStore } from './store/useProjectStore';
import { useToast, GithubLink } from './components/shared';
import { ensureAuthenticated } from './utils/auth';

function AppRoutes() {
  const { currentProject, syncProject, error, setError } = useProjectStore();
  const { show, ToastContainer } = useToast();
  const location = useLocation();
  const navigate = useNavigate();
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const runAuthCheck = async () => {
      setAuthReady(false);

      try {
        const params = new URLSearchParams(location.search);
        const code = params.get('code');
        const result = await ensureAuthenticated(code);

        if (cancelled || result !== 'authenticated') {
          return;
        }

        if (code) {
          params.delete('code');
          params.delete('state');
          const nextSearch = params.toString();
          navigate(
            {
              pathname: location.pathname,
              search: nextSearch ? `?${nextSearch}` : '',
            },
            { replace: true },
          );
        }

        setAuthReady(true);
      } catch (authError) {
        console.error('Auth check failed', authError);
      }
    };

    void runAuthCheck();

    return () => {
      cancelled = true;
    };
  }, [location.pathname, location.search, navigate]);

  useEffect(() => {
    if (!authReady) {
      return;
    }

    const savedProjectId = localStorage.getItem('currentProjectId');
    if (savedProjectId && !currentProject) {
      syncProject();
    }
  }, [authReady, currentProject, syncProject]);

  useEffect(() => {
    if (!authReady) {
      return;
    }

    if (error) {
      show({ message: error, type: 'error' });
      setError(null);
    }
  }, [authReady, error, setError, show]);

  if (!authReady) {
    return (
      <>
        <div className="flex min-h-screen items-center justify-center bg-slate-950 text-sm text-slate-100">
          Authenticating...
        </div>
        <ToastContainer />
        <GithubLink />
      </>
    );
  }

  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/project/:projectId/outline" element={<OutlineEditor />} />
        <Route path="/project/:projectId/detail" element={<DetailEditor />} />
        <Route path="/project/:projectId/preview" element={<SlidePreview />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastContainer />
      <GithubLink />
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
