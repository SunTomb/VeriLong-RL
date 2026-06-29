import type { DemoCase } from '../api/client';

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
          <p className="eyebrow">Selected task</p>
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
          <span key={role}>{role}: {count}</span>
        ))}
      </div>
      <div className="documents">
        {demoCase.documents.slice(0, 18).map((document) => (
          <article className={`document role-${document.role}`} key={document.evidence_id}>
            <div className="document-meta">
              <strong>{document.evidence_id}</strong>
              <span>{document.doc_id}</span>
              <span>{document.role}</span>
            </div>
            <p>{document.text}</p>
          </article>
        ))}
      </div>
      {demoCase.documents.length > 18 ? (
        <p className="note">Showing the first 18 of {demoCase.documents.length} documents for readability.</p>
      ) : null}
    </section>
  );
}
