import { useEffect, useState } from 'react';
import { fetchSummary, type SummaryResponse } from './api/client';
import { DemoPage } from './pages/DemoPage';
import { HomePage } from './pages/HomePage';

type Page = 'home' | 'demo';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary()
      .then(setSummary)
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="app-shell">
      <header className="top-nav">
        <button className="brand" type="button" onClick={() => setPage('home')}>VeriLong-RL</button>
        <nav aria-label="主导航">
          <button className={page === 'home' ? 'nav-active' : ''} type="button" onClick={() => setPage('home')}>首页</button>
          <button className={page === 'demo' ? 'nav-active' : ''} type="button" onClick={() => setPage('demo')}>交互式演示</button>
        </nav>
      </header>

      {error ? <main className="section"><p className="error" role="alert">{error}</p></main> : null}
      {!error && !summary ? <main className="section"><p>正在加载 VeriLong-RL 基准数据...</p></main> : null}
      {!error && summary && page === 'home' ? <HomePage summary={summary} onOpenDemo={() => setPage('demo')} /> : null}
      {!error && summary && page === 'demo' ? <DemoPage /> : null}
    </div>
  );
}
