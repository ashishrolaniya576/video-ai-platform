import { Link, useLocation } from 'react-router-dom';

const NAV_LINKS = [
  { label: 'Dashboard', to: '/' },
];

function Navbar() {
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-50 border-b border-surface-700 bg-surface-900/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-brand-600 to-accent-purple shadow-lg shadow-brand-900/50">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight gradient-text">VideoAI</span>
              <span className="text-slate-400 font-medium text-lg ml-0.5"> Platform</span>
            </div>
          </div>

          <div className="flex items-center gap-1">
            {NAV_LINKS.map(({ label, to }) => (
              <Link
                key={to}
                to={to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  location.pathname === to
                    ? 'bg-brand-900/60 text-brand-300 border border-brand-800'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-surface-700'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-700 border border-surface-600">
              <span className="w-2 h-2 rounded-full bg-accent-green animate-pulse-slow" />
              <span className="text-xs text-slate-400 font-medium">AI Ready</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
