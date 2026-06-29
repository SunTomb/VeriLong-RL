import { useEffect, useMemo, useState } from 'react';
import { fetchCase, fetchCases, runDryDemo, type DemoCase, type DemoCaseSummary, type DemoRunResponse } from '../api/client';
import { OutputPanel } from '../components/OutputPanel';
import { ScoreBreakdown } from '../components/ScoreBreakdown';
import { TaskViewer } from '../components/TaskViewer';

const FAMILY_SHORT_ZH: Record<string, string> = {
  anti_distractor_retrieval: '抗干扰检索',
  multi_hop_reasoning: '多跳推理',
  temporal_update: '时序更新',
};

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
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedTaskId) return;
    let cancelled = false;
    setDryRun(null);
    fetchCase(selectedTaskId)
      .then((item) => { if (!cancelled) setSelectedCase(item); })
      .catch((err: Error) => setError(err.message));
    return () => { cancelled = true; };
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

  if (loading) return <main className="section"><p>正在加载演示任务列表...</p></main>;

  return (
    <main className="demo-layout">
      <section className="section demo-intro">
        <p className="eyebrow">交互式评估演示</p>
        <h1>查看基准任务样例、运行离线评分器，并洞见奖励信号拆解细节。</h1>
        <p>本演示系统采用离线优先设计。"运行离线评分"将直接调用后端的基准评估模块；无须向浏览器或外网传输任何服务商的 API 密钥。</p>
        {error ? <p className="error" role="alert">{error}</p> : null}
      </section>

      <section className="panel controls" aria-label="演示控制面板">
        <label htmlFor="case-select">选择已录入的任务样例：</label>
        <select id="case-select" value={selectedTaskId} onChange={(e) => setSelectedTaskId(e.target.value)}>
          {cases.map((item) => (
            <option value={item.task_id} key={item.task_id}>
              {item.task_id} · {FAMILY_SHORT_ZH[item.task_family] ?? item.task_family} · reward {item.reward_total?.toFixed(3) ?? '—'}
            </option>
          ))}
        </select>
        {selectedSummary ? <p className="note">{selectedSummary.question}</p> : null}
        <button className="primary" type="button" onClick={handleDryRun} disabled={running || !selectedCase}>
          {running ? '正在进行离线评分...' : '运行离线评分器 (Dry-run)'}
        </button>
      </section>

      {selectedCase ? <TaskViewer demoCase={selectedCase} /> : null}

      {selectedCase ? (
        <div className="result-grid">
          <OutputPanel
            title={`已缓存输出 · ${selectedCase.model}`}
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
            title="离线评分器输出 (Dry-run Oracle Stub)"
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
