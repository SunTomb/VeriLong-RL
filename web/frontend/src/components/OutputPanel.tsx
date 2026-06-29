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
          <p className="eyebrow">模型评估输出</p>
          <h2 id={`${title.replace(/\s+/g, '-')}-heading`}>{title}</h2>
        </div>
      </div>
      <pre className="output-box">{outputText}</pre>
      <div className="parsed-grid">
        <div>
          <h3>已解析引用 (Evidence)</h3>
          <p>{evidenceIds.length ? evidenceIds.join(', ') : '未解析到任何有效引用'}</p>
        </div>
        <div>
          <h3>已解析答案 (Answer)</h3>
          <p>{String(parsedOutput.pred_answer ?? '')}</p>
        </div>
      </div>
      <div>
        <h3>已解析步骤 (Steps)</h3>
        <ol>
          {steps.map((step, index) => (
            <li key={`${step}-${index}`}>{step}</li>
          ))}
        </ol>
      </div>
      {promptPreview ? (
        <details className="prompt-preview">
          <summary>大模型输入 Prompt 预览</summary>
          <pre>{promptPreview}</pre>
        </details>
      ) : null}
    </section>
  );
}
