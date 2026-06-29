interface Props {
  metrics: Record<string, unknown>;
  errorType?: string | null;
}

const METRIC_KEYS = [
  ['reward_total', 'Reward'],
  ['answer_normalized_match', 'Answer'],
  ['citation_f1', 'Citation F1'],
  ['format_valid', 'Format'],
  ['step_count_valid', 'Steps'],
  ['distractor_citation_rate', 'Distractor rate'],
  ['stale_citation_rate', 'Stale rate'],
  ['overcitation_rate', 'Over-citation'],
];

function formatMetric(value: unknown): string {
  if (typeof value === 'number') {
    return value.toFixed(3);
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (value == null) {
    return '—';
  }
  return String(value);
}

export function ScoreBreakdown({ metrics, errorType }: Props) {
  return (
    <section className="panel" aria-labelledby="score-heading">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Programmatic reward</p>
          <h2 id="score-heading">Score breakdown</h2>
        </div>
        <span className={errorType ? 'pill warning' : 'pill success'}>{errorType ?? 'no error'}</span>
      </div>
      <dl className="metric-list wide">
        {METRIC_KEYS.map(([key, label]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>{formatMetric(metrics[key])}</dd>
          </div>
        ))}
      </dl>
      {typeof metrics.reward_components === 'object' && metrics.reward_components ? (
        <details className="prompt-preview">
          <summary>Reward components</summary>
          <pre>{JSON.stringify(metrics.reward_components, null, 2)}</pre>
        </details>
      ) : null}
    </section>
  );
}
