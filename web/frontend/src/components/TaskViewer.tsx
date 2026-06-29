import type { DemoCase } from '../api/client';

const ROLE_ZH: Record<string, string> = {
  gold: '黄金证据 (Gold)',
  distractor: '相似干扰项 (Distractor)',
  stale: '历史过时证据 (Stale)',
  neutral: '中性填充 (Neutral)',
};

interface Props {
  demoCase: DemoCase;
}

export function TaskViewer({ demoCase }: Props) {
  const roleCounts = demoCase.documents.reduce<Record<string, number>>((counts, document) => {
    counts[document.role] = (counts[document.role] ?? 0) + 1;
    return counts;
  }, {});

  return (
    <section className="panel" aria-labelledby="task-heading">
      <div className="panel-header">
        <div>
          <p className="eyebrow">当前任务属性</p>
          <h2 id="task-heading">{demoCase.task_id}</h2>
        </div>
        <div className="tag-row">
          <span className="pill">{demoCase.task_family}</span>
          <span className="pill">{demoCase.difficulty}</span>
        </div>
      </div>
      <p className="question">{demoCase.question}</p>
      <div className="role-counts">
        {Object.entries(roleCounts).map(([role, count]) => (
          <span key={role}>{ROLE_ZH[role] ?? role}: {count}</span>
        ))}
      </div>
      <div className="documents">
        {demoCase.documents.slice(0, 18).map((document) => (
          <article className={`document role-${document.role}`} key={document.evidence_id}>
            <div className="document-meta">
              <strong>{document.evidence_id}</strong>
              <span>{document.doc_id}</span>
              <span>{ROLE_ZH[document.role] ?? document.role}</span>
            </div>
            <p>{document.text}</p>
          </article>
        ))}
      </div>
      {demoCase.documents.length > 18 ? (
        <p className="note">为保持页面阅读体验，仅展示前 18 / {demoCase.documents.length} 篇文档片段。</p>
      ) : null}
    </section>
  );
}
