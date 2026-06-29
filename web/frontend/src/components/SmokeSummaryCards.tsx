import type { SmokeSummary } from '../api/client';

const BASELINE_ZH: Record<string, string> = {
  'Oracle smoke': 'Oracle 完美预期基线',
  'Corrupted smoke': 'Corrupted 噪声污染基线',
};

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

interface Props {
  summaries: SmokeSummary[];
}

export function SmokeSummaryCards({ summaries }: Props) {
  return (
    <section className="section" aria-labelledby="smoke-heading">
      <div className="section-header">
        <p className="eyebrow">已验证构件</p>
        <h2 id="smoke-heading">冒烟测试基线（非模型排行榜）</h2>
        <p>此数据来源于已归档的 pilot 阶段冒烟测试评测结果。主要用于评估流程验证和 UI 功能展示，并非真实的基座模型排名。</p>
      </div>
      <div className="card-grid two">
        {summaries.map((summary) => (
          <article className="card metric-card" key={summary.baseline}>
            <div className="card-title-row">
              <h3>{BASELINE_ZH[summary.label] ?? summary.label}</h3>
              <span className="pill">{summary.count} 个任务</span>
            </div>
            <dl className="metric-list">
              <div><dt>综合奖励值</dt><dd>{summary.reward_total_mean.toFixed(3)}</dd></div>
              <div><dt>答案完全匹配</dt><dd>{pct(summary.answer_exact_match_mean)}</dd></div>
              <div><dt>引用 F1 值</dt><dd>{pct(summary.citation_f1_mean)}</dd></div>
              <div><dt>过度引用率</dt><dd>{pct(summary.overcitation_rate_mean)}</dd></div>
            </dl>
            <p className="note">{summary.note}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
