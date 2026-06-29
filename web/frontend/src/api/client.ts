export interface SmokeSummary {
  label: string;
  baseline: string;
  count: number;
  reward_total_mean: number;
  answer_exact_match_mean: number;
  citation_f1_mean: number;
  overcitation_rate_mean: number;
  note: string;
}

export interface TaskFamilySummary {
  id: string;
  label: string;
  description: string;
  signal: string;
}

export interface SummaryResponse {
  project: string;
  tagline: string;
  status: Record<string, string>;
  output_format: string[];
  task_families: TaskFamilySummary[];
  smoke_summaries: SmokeSummary[];
}

export interface DemoDocument {
  doc_id: string;
  evidence_id: string;
  text: string;
  role: string;
}

export interface DemoCaseSummary {
  task_id: string;
  task_family: string;
  difficulty: string;
  question: string;
  model: string;
  reward_total: number | null;
  error_type: string | null;
}

export interface DemoCase {
  task_id: string;
  task_family: string;
  difficulty: string;
  question: string;
  documents: DemoDocument[];
  gold_answer: string;
  gold_evidence_ids: string[];
  distractor_evidence_ids: string[];
  stale_evidence_ids: string[];
  model: string;
  model_output: string;
  parsed_output: Record<string, unknown>;
  metric_breakdown: Record<string, unknown>;
  error_type: string | null;
  prompt_preview: string;
}

export interface DemoRunResponse {
  task_id: string;
  model: string;
  source: 'dry_run';
  output_text: string;
  parsed_output: Record<string, unknown>;
  metric_breakdown: Record<string, unknown>;
  error_type: string | null;
  prompt_preview: string;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchSummary(): Promise<SummaryResponse> {
  return getJson<SummaryResponse>('/api/summary');
}

export function fetchCases(): Promise<DemoCaseSummary[]> {
  return getJson<DemoCaseSummary[]>('/api/cases');
}

export function fetchCase(taskId: string): Promise<DemoCase> {
  return getJson<DemoCase>(`/api/cases/${encodeURIComponent(taskId)}`);
}

export async function runDryDemo(taskId: string): Promise<DemoRunResponse> {
  const response = await fetch('/api/demo/dry-run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });
  if (!response.ok) {
    throw new Error(`Dry-run failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<DemoRunResponse>;
}
