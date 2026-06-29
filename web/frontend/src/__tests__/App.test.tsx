import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../App';

const summary = {
  project: 'VeriLong-RL',
  tagline: 'A verifiable long-context benchmark for evidence-grounded reasoning and RLVR.',
  status: {
    phase1_pilot: 'completed',
    hard_difficulty: 'completed',
    sft_warmup: 'completed',
    rlvr_pipeline: 'validated',
    rlvr_full_run: 'deferred',
    phase2: 'design_only',
  },
  output_format: ['Evidence: E01, E02', 'Steps:', '1. Grounded reasoning step.', 'Answer: final answer only'],
  task_families: [
    {
      id: 'anti_distractor_retrieval',
      label: 'Anti-distractor retrieval',
      description: 'Find the one supporting record.',
      signal: 'Citation precision.',
    },
  ],
  smoke_summaries: [
    {
      label: 'Oracle smoke',
      baseline: 'oracle_format_baseline',
      count: 1200,
      reward_total_mean: 0.95,
      answer_exact_match_mean: 1,
      citation_f1_mean: 1,
      overcitation_rate_mean: 0,
      note: 'Smoke baseline only; not a live model leaderboard.',
    },
  ],
};

const caseSummary = {
  task_id: 'vlr_pilot_000001',
  task_family: 'anti_distractor_retrieval',
  difficulty: 'easy',
  question: 'Which access code is assigned to Project Nova?',
  model: 'oracle_format_baseline',
  reward_total: 0.95,
  error_type: null,
};

const detail = {
  ...caseSummary,
  documents: [
    { doc_id: 'D01', evidence_id: 'E01', text: 'Project Nova uses access code M63.', role: 'gold' },
  ],
  gold_answer: 'M63',
  gold_evidence_ids: ['E01'],
  distractor_evidence_ids: [],
  stale_evidence_ids: [],
  model_output: 'Evidence: E01\nSteps:\n1. Therefore the answer is M63.\nAnswer: M63',
  parsed_output: {
    pred_answer: 'M63',
    pred_evidence_ids: ['E01'],
    pred_steps: ['Therefore the answer is M63.'],
    format_valid: true,
  },
  metric_breakdown: {
    reward_total: 0.95,
    answer_normalized_match: 1,
    citation_f1: 1,
    format_valid: 1,
    step_count_valid: 1,
    distractor_citation_rate: 0,
    stale_citation_rate: 0,
    overcitation_rate: 0,
  },
  prompt_preview: 'Question: Which access code is assigned to Project Nova?\nDocuments:',
};

const dryRun = {
  task_id: 'vlr_pilot_000001',
  model: 'dry_run_oracle_stub',
  source: 'dry_run',
  output_text: 'Evidence: E01\nSteps:\n1. Therefore the answer is M63.\nAnswer: M63',
  parsed_output: detail.parsed_output,
  metric_breakdown: { ...detail.metric_breakdown, reward_components: { answer: 0.4 } },
  error_type: null,
  prompt_preview: detail.prompt_preview,
};

function mockFetch() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input);
    if (url === '/api/summary') return Promise.resolve(jsonResponse(summary));
    if (url === '/api/cases') return Promise.resolve(jsonResponse([caseSummary]));
    if (url === '/api/cases/vlr_pilot_000001') return Promise.resolve(jsonResponse(detail));
    if (url === '/api/demo/dry-run' && init?.method === 'POST') return Promise.resolve(jsonResponse(dryRun));
    return Promise.resolve(new Response('not found', { status: 404 }));
  });
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => { vi.restoreAllMocks(); });

describe('App', () => {
  it('renders Chinese homepage with project name and smoke label', async () => {
    mockFetch();
    render(<App />);

    expect(await screen.findByRole('heading', { name: 'VeriLong-RL' })).toBeInTheDocument();
    expect(screen.getByText(/项目当前进度/)).toBeInTheDocument();
    expect(screen.getByText(/冒烟测试基线（非模型排行榜）/)).toBeInTheDocument();
    expect(screen.getByText(/抗干扰证据检索/)).toBeInTheDocument();
  });

  it('runs the Chinese dry-run demo flow', async () => {
    mockFetch();
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: '交互式演示' }));
    expect(await screen.findByText(/Which access code is assigned to Project Nova/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /运行离线评分器/ }));

    expect(await screen.findByRole('heading', { name: /离线评分器输出/ })).toBeInTheDocument();
    expect(screen.getAllByText('0.950').length).toBeGreaterThan(0);
  });
});
