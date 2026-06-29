import { SmokeSummaryCards } from '../components/SmokeSummaryCards';
import { TaskFamilyCards } from '../components/TaskFamilyCards';
import type { SummaryResponse } from '../api/client';

interface Props {
  summary: SummaryResponse;
  onOpenDemo: () => void;
}

export function HomePage({ summary, onOpenDemo }: Props) {
  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Phase 1 pilot completed</p>
          <h1>{summary.project}</h1>
          <p className="hero-copy">{summary.tagline}</p>
          <div className="hero-actions">
            <button className="primary" type="button" onClick={onOpenDemo}>Try the interactive demo</button>
            <a className="secondary" href="https://github.com/SunTomb/VeriLong-RL" target="_blank" rel="noreferrer">GitHub repository</a>
          </div>
        </div>
        <aside className="status-card" aria-label="Project status">
          <h2>Current status</h2>
          <ul>
            <li><strong>Phase 1 pilot:</strong> {summary.status.phase1_pilot}</li>
            <li><strong>Hard difficulty:</strong> {summary.status.hard_difficulty}</li>
            <li><strong>SFT warmup:</strong> {summary.status.sft_warmup}</li>
            <li><strong>RLVR pipeline:</strong> {summary.status.rlvr_pipeline}</li>
            <li><strong>RLVR full run:</strong> {summary.status.rlvr_full_run}</li>
            <li><strong>Phase 2:</strong> {summary.status.phase2}</li>
          </ul>
        </aside>
      </section>

      <section className="section format-section" aria-labelledby="format-heading">
        <div className="section-header">
          <p className="eyebrow">Fixed output contract</p>
          <h2 id="format-heading">Evidence / Steps / Answer</h2>
          <p>The web demo keeps the same parser-compatible format used by the benchmark and RL reward.</p>
        </div>
        <pre className="output-box">{summary.output_format.join('\n')}</pre>
      </section>

      <TaskFamilyCards families={summary.task_families} />
      <SmokeSummaryCards summaries={summary.smoke_summaries} />
    </main>
  );
}
