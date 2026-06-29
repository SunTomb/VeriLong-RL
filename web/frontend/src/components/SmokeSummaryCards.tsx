import type { SmokeSummary } from '../api/client';

interface Props {
  summaries: SmokeSummary[];
}

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function SmokeSummaryCards({ summaries }: Props) {
  return (
    <section className="section" aria-labelledby="smoke-heading">
      <div className="section-header">
        <p className="eyebrow">Verified artifacts</p>
        <h2 id="smoke-heading">Smoke baselines, not a leaderboard</h2>
        <p>
          These cards come from tracked pilot smoke artifacts. They are for pipeline validation and UI
          demonstration, not fabricated model claims.
        </p>
      </div>
      <div className="card-grid two">
        {summaries.map((summary) => (
          <article className="card metric-card" key={summary.baseline}>
            <div className="card-title-row">
              <h3>{summary.label}</h3>
              <span className="pill">{summary.count} tasks</span>
            </div>
            <dl className="metric-list">
              <div>
                <dt>Reward</dt>
                <dd>{summary.reward_total_mean.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Answer EM</dt>
                <dd>{pct(summary.answer_exact_match_mean)}</dd>
              </div>
              <div>
                <dt>Citation F1</dt>
                <dd>{pct(summary.citation_f1_mean)}</dd>
              </div>
              <div>
                <dt>Over-citation</dt>
                <dd>{pct(summary.overcitation_rate_mean)}</dd>
              </div>
            </dl>
            <p className="note">{summary.note}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
