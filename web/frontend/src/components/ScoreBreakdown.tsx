interface Props {
  metrics: Record<string, unknown>;
  errorType?: string | null;
}

const METRIC_KEYS = [
  ['reward_total', '综合得分 (Reward)'],
  ['answer_normalized_match', '答案规范匹配'],
  ['citation_f1', '引用 F1 值'],
  ['format_valid', '格式规约率'],
  ['step_count_valid', '推理步数有效性'],
  ['distractor_citation_rate', '干扰项误引率'],
  ['stale_citation_rate', '过期记录误引率'],
  ['overcitation_rate', '冗余过度引用率'],
];

function formatMetric(value: unknown): string {
  if (typeof value === 'number') return value.toFixed(3);
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (value == null) return '—';
  return String(value);
}

export function ScoreBreakdown({ metrics, errorType }: Props) {
  return (
    <section className="panel" aria-labelledby="score-heading">
      <div className="panel-header">
        <div>
          <p className="eyebrow">程序化评测奖励</p>
          <h2 id="score-heading">奖励信号得分细则</h2>
        </div>
        <span className={errorType ? 'pill warning' : 'pill success'}>
          {errorType ?? '评估正常解析'}
        </span>
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
          <summary>奖励组件原始权重与得分明细 (Components)</summary>
          <pre>{JSON.stringify(metrics.reward_components, null, 2)}</pre>
        </details>
      ) : null}
    </section>
  );
}
