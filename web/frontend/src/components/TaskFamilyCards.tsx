import type { TaskFamilySummary } from '../api/client';

const FAMILY_ZH: Record<string, { label: string; description: string; signal: string }> = {
  anti_distractor_retrieval: {
    label: '抗干扰证据检索',
    description: '从大量字面或语义相似的干扰文本中，精准定位唯一的支持证据段落。',
    signal: '引用精确率与干扰项引用惩罚项共同衡量模型的"过度引用"行为。',
  },
  multi_hop_reasoning: {
    label: '多跳证据推理',
    description: '组合分布于长上下文不同位置的多个离散事实，完成多跳逻辑链推理。',
    signal: '黄金证据召回率与推理步骤有效性衡量模型是否遍历所有推理路径。',
  },
  temporal_update: {
    label: '时序证据更新',
    description: '在包含过期事实与历史副本的长文档中，识别并应用最新时序更新信息。',
    signal: '过期引用惩罚项严格区分模型引用的是最新有效数据还是过时历史数据。',
  },
};

interface Props {
  families: TaskFamilySummary[];
}

export function TaskFamilyCards({ families }: Props) {
  return (
    <section className="section" aria-labelledby="task-families-heading">
      <div className="section-header">
        <p className="eyebrow">核心评估任务族</p>
        <h2 id="task-families-heading">三大可验证长上下文推理能力</h2>
      </div>
      <div className="card-grid three">
        {families.map((family) => {
          const zh = FAMILY_ZH[family.id];
          return (
            <article className="card" key={family.id}>
              <h3>{zh?.label ?? family.label}</h3>
              <p>{zh?.description ?? family.description}</p>
              <p className="signal">评估信号：{zh?.signal ?? family.signal}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
