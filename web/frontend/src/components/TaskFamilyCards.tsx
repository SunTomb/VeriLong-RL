import type { TaskFamilySummary } from '../api/client';

interface Props {
  families: TaskFamilySummary[];
}

export function TaskFamilyCards({ families }: Props) {
  return (
    <section className="section" aria-labelledby="task-families-heading">
      <div className="section-header">
        <p className="eyebrow">Benchmark families</p>
        <h2 id="task-families-heading">Three verifiable long-context skills</h2>
      </div>
      <div className="card-grid three">
        {families.map((family) => (
          <article className="card" key={family.id}>
            <h3>{family.label}</h3>
            <p>{family.description}</p>
            <p className="signal">Signal: {family.signal}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
