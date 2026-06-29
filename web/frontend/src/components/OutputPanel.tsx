interface Props {
  title: string;
  outputText: string;
  parsedOutput: Record<string, unknown>;
  promptPreview?: string;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

export function OutputPanel({ title, outputText, parsedOutput, promptPreview }: Props) {
  const evidenceIds = asStringArray(parsedOutput.pred_evidence_ids);
  const steps = asStringArray(parsedOutput.pred_steps);

  return (
    <section className="panel" aria-labelledby={`${title.replace(/\s+/g, '-')}-heading`}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">Model output</p>
          <h2 id={`${title.replace(/\s+/g, '-')}-heading`}>{title}</h2>
        </div>
      </div>
      <pre className="output-box">{outputText}</pre>
      <div className="parsed-grid">
        <div>
          <h3>Parsed evidence</h3>
          <p>{evidenceIds.length ? evidenceIds.join(', ') : 'None parsed'}</p>
        </div>
        <div>
          <h3>Parsed answer</h3>
          <p>{String(parsedOutput.pred_answer ?? '')}</p>
        </div>
      </div>
      <div>
        <h3>Parsed steps</h3>
        <ol>
          {steps.map((step, index) => (
            <li key={`${step}-${index}`}>{step}</li>
          ))}
        </ol>
      </div>
      {promptPreview ? (
        <details className="prompt-preview">
          <summary>Prompt preview</summary>
          <pre>{promptPreview}</pre>
        </details>
      ) : null}
    </section>
  );
}
