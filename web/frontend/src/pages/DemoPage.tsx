import { useEffect, useMemo, useState } from 'react';
import { fetchCase, fetchCases, runDryDemo, type DemoCase, type DemoCaseSummary, type DemoRunResponse } from '../api/client';
import { OutputPanel } from '../components/OutputPanel';
import { ScoreBreakdown } from '../components/ScoreBreakdown';
import { TaskViewer } from '../components/TaskViewer';

export function DemoPage() {
  const [cases, setCases] = useState<DemoCaseSummary[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string>('');
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [dryRun, setDryRun] = useState<DemoRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchCases()
      .then((items) => {
        if (cancelled) return;
        setCases(items);
        setSelectedTaskId(items[0]?.task_id ?? '');
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedTaskId) return;
    let cancelled = false;
    setDryRun(null);
    fetchCase(selectedTaskId)
      .then((item) => {
        if (!cancelled) setSelectedCase(item);
      })
      .catch((err: Error) => setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [selectedTaskId]);

  const selectedSummary = useMemo(
    () => cases.find((item) => item.task_id === selectedTaskId),
    [cases, selectedTaskId],
  );

  async function handleDryRun() {
    if (!selectedTaskId) return;
    setRunning(true);
    setError(null);
    try {
      setDryRun(await runDryDemo(selectedTaskId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  if (loading) {
    return <main className="section"><p>Loading demo cases…</p></main>;
  }

  return (
    <main className="demo-layout">
      <section className="section demo-intro">
        <p className="eyebrow">Interactive demo</p>
        <h1>Inspect a task, run the offline scorer, and see the reward decomposition.</h1>
        <p>
          This demo is offline-first. Dry-run uses the existing benchmark stub and scorer; no provider key is sent to the browser.
        </p>
        {error ? <p className="error" role="alert">{error}</p> : null}
      </section>

      <section className="panel controls" aria-label="Demo controls">
        <label htmlFor="case-select">Choose a tracked demo case</label>
        <select id="case-select" value={selectedTaskId} onChange={(event) => setSelectedTaskId(event.target.value)}>
          {cases.map((item) => (
            <option value={item.task_id} key={item.task_id}>
              {item.task_id} · {item.task_family} · reward {item.reward_total?.toFixed(3) ?? '—'}
            </option>
          ))}
        </select>
        {selectedSummary ? <p className="note">{selectedSummary.question}</p> : null}
        <button className="primary" type="button" onClick={handleDryRun} disabled={running || !selectedCase}>
          {running ? 'Running dry-run…' : 'Run dry-run scorer'}
        </button>
      </section>

      {selectedCase ? <TaskViewer demoCase={selectedCase} /> : null}

      {selectedCase ? (
        <div className="result-grid">
          <OutputPanel
            title={`Cached ${selectedCase.model}`}
            outputText={selectedCase.model_output}
            parsedOutput={selectedCase.parsed_output}
            promptPreview={selectedCase.prompt_preview}
          />
          <ScoreBreakdown metrics={selectedCase.metric_breakdown} errorType={selectedCase.error_type} />
        </div>
      ) : null}

      {dryRun ? (
        <div className="result-grid dry-run-grid">
          <OutputPanel
            title="Dry-run oracle stub"
            outputText={dryRun.output_text}
            parsedOutput={dryRun.parsed_output}
            promptPreview={dryRun.prompt_preview}
          />
          <ScoreBreakdown metrics={dryRun.metric_breakdown} errorType={dryRun.error_type} />
        </div>
      ) : null}
    </main>
  );
}
