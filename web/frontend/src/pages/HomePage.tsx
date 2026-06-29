const STATUS_LABELS: Record<string, string> = {
  phase1_pilot: '阶段 1 综合能力',
  hard_difficulty: '困难难度系统',
  sft_warmup: 'SFT 热身训练',
  rlvr_pipeline: 'RLVR 对齐流程',
  rlvr_full_run: 'RLVR 完整训练',
  phase2: '阶段 2 真实文献',
};

const STATUS_VALUES: Record<string, string> = {
  completed: '✓ 已完成',
  validated: '✓ 管线已验证',
  deferred: '⋯ 待算力资源',
  design_only: '⋯ 仅设计文档',
};

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
          <p className="eyebrow">阶段 1 Pilot 测试已完成</p>
          <h1>{summary.project}</h1>
          <p className="hero-copy">{summary.tagline}</p>
          <div className="hero-actions">
            <button className="primary" type="button" onClick={onOpenDemo}>进入交互式在线演示</button>
            <a className="secondary" href="https://github.com/SunTomb/VeriLong-RL" target="_blank" rel="noreferrer">GitHub 开源仓库</a>
          </div>
        </div>
        <aside className="status-card" aria-label="项目当前进度">
          <h2>项目当前进度</h2>
          <ul>
            {Object.entries(summary.status).map(([key, value]) => (
              <li key={key}>
                <strong>{STATUS_LABELS[key] ?? key}：</strong>
                {STATUS_VALUES[value] ?? value}
              </li>
            ))}
          </ul>
        </aside>
      </section>

      <section className="section format-section" aria-labelledby="format-heading">
        <div className="section-header">
          <p className="eyebrow">固定输出格式规约</p>
          <h2 id="format-heading">证据引用 / 推理步骤 / 最终答案</h2>
          <p>Web 演示系统完全遵循与基准评估及 RL 奖励信号一致的结构化输出格式。</p>
        </div>
        <pre className="output-box">{summary.output_format.join('\n')}</pre>
      </section>

      <TaskFamilyCards families={summary.task_families} />
      <SmokeSummaryCards summaries={summary.smoke_summaries} />
    </main>
  );
}
