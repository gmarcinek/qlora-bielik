export type Corpus = {
  id: string;
  name: string;
  description: string;
  example_count: number;
};

export type Example = {
  id: string;
  corpus_id?: string;
  corpus_name?: string;
  created_at: string;
  split: "train" | "validation" | "test";
  messages: Array<{ role: string; content: string }>;
  metadata: { flag?: ExampleFlag; import_id?: string };
};

export type MessageRole = "system" | "user" | "assistant";

export type Message = {
  role: MessageRole;
  content: string;
};

export type ExampleFlag = "positive" | "negative" | "unclassified";

type ExampleDraft = {
  split: string;
  messages: Message[];
  flag: ExampleFlag;
};

export type ImportedExample = {
  messages: Message[];
  split?: "train" | "validation" | "test";
  source?: string;
  flag?: ExampleFlag;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? "Nie udalo sie wykonac zadania.");
  }
  return response.json() as Promise<T>;
}

export type TrainingMetric = {
  step: number;
  max_steps: number;
  time: number;
  epoch?: number;
  num_train_epochs?: number;
  loss?: number;
  eval_loss?: number;
  eval_mean_token_accuracy?: number;
  learning_rate?: number;
  grad_norm?: number;
  mean_token_accuracy?: number;
  entropy?: number;
  train_loss?: number;
  train_runtime?: number;
  max_grad_norm?: number;
  perplexity?: number;
  eval_perplexity?: number;
  generalization_gap?: number;
  seconds_per_step?: number;
  tokens_per_second?: number;
  vram_peak_gb?: number;
  vram_reserved_gb?: number;
  gpu_util?: number;
  gpu_temp?: number;
  gpu_power?: number;
  gpu_memory_used_gb?: number;
  lora_a_norm?: number;
  lora_b_norm?: number;
  lora_grad_norm?: number;
  // group_loss/flag:positive, group_loss/type:XYZ, ...
  [key: string]: number | undefined;
};

export type DatasetSplitStats = {
  count: number;
  mean: number;
  p50: number;
  p90: number;
  p99: number;
  max: number;
  truncated: number;
  truncated_pct: number;
  bin_width: number;
  histogram: number[];
};

export type DatasetStats = {
  max_length: number;
  splits: Record<string, DatasetSplitStats>;
  groups: Record<string, number>;
};

export type LoraLayerSnapshot = {
  step: number;
  layers: Array<{ layer: number; b_norm: number; grad_norm: number | null }>;
};

export type TrainingHyperparameters = {
  learning_rate: number | null;
  epochs: number | null;
  batch_size: number | null;
  gradient_accumulation_steps: number | null;
  max_length: number | null;
  quantization: string | null;
  lora_rank: number | null;
  lora_alpha: number | null;
  lora_dropout: number | null;
};

export type TrainingStatus = {
  state: string;
  splits: Record<"train" | "validation" | "test", number>;
  profile: string;
  logs: string;
  container_id: string | null;
  exit_code?: number | null;
  started_at?: string;
  finished_at?: string;
  metrics?: TrainingMetric[];
  dataset_stats?: DatasetStats | null;
  lora_layers?: LoraLayerSnapshot | null;
  hyperparameters?: TrainingHyperparameters | null;
  error?: string;
  adapter_ready: boolean;
  job?: {
    corpus_id: string | null;
    base_model: string | null;
    adapter_name: string | null;
  };
};

export type TrainingRunSummary = {
  run_id: string;
  adapter_name: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  steps: number;
  best_eval_loss: number | null;
};

export type TrainingStart = {
  corpusId: string;
  baseModel: string;
  adapterName: string;
};

export type EvaluationScore = {
  tp: number;
  fp: number;
  fn: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

export type EvaluationSummary = {
  examples: number;
  json_valid: number | null;
  exact_match: number | null;
  strict: EvaluationScore;
  relaxed: EvaluationScore;
  per_type: Record<string, EvaluationScore>;
  per_type_relaxed?: Record<string, EvaluationScore>;
  negatives: { examples: number; correct: number };
};

export type EvaluationCurvePoint = {
  n: number;
  json_valid: number;
  strict_precision: number | null;
  strict_recall: number | null;
  strict_f1: number | null;
  relaxed_precision: number | null;
  relaxed_recall: number | null;
  relaxed_f1: number | null;
};

export type EvaluationCheckpointResult = {
  checkpoint: string;
  done: number;
  total: number | null;
  finished: boolean;
  summary: EvaluationSummary | null;
  curve: EvaluationCurvePoint[];
};

export type EvaluationStatus = {
  state: string;
  exit_code?: number | null;
  stopped?: boolean;
  progress: {
    checkpoint?: string;
    done: number;
    total: number;
    elapsed: number;
  } | null;
  summary: EvaluationSummary | null;
  comparison?: Array<EvaluationSummary & { checkpoint: string }>;
  checkpoints?: EvaluationCheckpointResult[];
  output?: string | null;
  adapter_name?: string | null;
  checkpoint?: string | null;
  splits?: string | null;
  logs: string;
  error?: string;
};

export type ServingStatus = {
  state: "idle" | "loading" | "ready" | "failed" | "unavailable";
  adapter_name?: string;
  checkpoint?: string;
  model?: string;
  logs?: string;
  error?: string | null;
  exit_code?: number | null;
};

export type ExportQuantization = "Q4_K_M" | "Q5_K_M" | "Q6_K" | "Q8_0";

export type ExportEntry = {
  id: string;
  adapter_name: string;
  checkpoint: string;
  quantization: ExportQuantization;
  model_name: string;
  gguf: string;
  state: "running" | "ready" | "failed";
  stage: string;
  stages: Record<string, { seconds: number }>;
  started_at: number;
  stage_started_at?: number;
  finished_at?: number;
  gguf_bytes?: number | null;
  error?: string;
  log_tail?: string;
};

export type ExportsStatus = {
  exports: ExportEntry[];
  stages: string[];
  logs: string;
};

export type BulkTransformResult = {
  matched: number;
  skipped: Record<string, number>;
  samples: Array<{ id: string; before: string; after: string }>;
  revision_id: string | null;
};

export type ExampleReview = {
  recommendation: "positive" | "negative" | "needs_review";
  reason: string;
  confidence: "high" | "medium" | "low";
};

export type ClassificationStatus = {
  state: "idle" | "running" | "completed" | "failed";
  total: number;
  processed: number;
  classified: number;
  needs_review: number;
  error: string | null;
};

export type ParaphraseProviderCatalog = {
  providers: Record<
    string,
    {
      label: string;
      models: Record<string, { label: string }>;
    }
  >;
};

export const api = {
  corpora: () => request<Corpus[]>("/api/corpora"),
  createCorpus: (name: string, description: string) =>
    request<Corpus>("/api/corpora", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),
  examples: (corpusId: string) =>
    request<Example[]>(`/api/corpora/${corpusId}/examples`),
  allExamples: () => request<Example[]>("/api/examples"),
  createExample: (corpusId: string, example: ExampleDraft) =>
    request<Example>(`/api/corpora/${corpusId}/examples`, {
      method: "POST",
      body: JSON.stringify(example),
    }),
  importExamples: (corpusId: string, examples: ImportedExample[]) =>
    request<{ imported: number; import_id: string }>(
      `/api/corpora/${corpusId}/examples/import`,
      {
        method: "POST",
        body: JSON.stringify({ examples }),
      },
    ),
  systemPromptValidator: () =>
    request<{ prompt: string }>("/api/system-prompts/validator"),
  systemPromptParaphraser: () =>
    request<{ prompt: string }>("/api/system-prompts/paraphraser"),
  paraphraseProviders: () =>
    request<ParaphraseProviderCatalog>("/api/paraphrase/providers"),
  compareSystemPrompts: (
    original: string,
    candidate: string,
    provider: string,
    model: string,
    validatorPrompt: string,
  ) =>
    request<{
      semantic_equivalent: boolean;
      instruction_plan_equivalent: boolean;
      reason: string;
    }>("/api/system-prompts/compare", {
      method: "POST",
      body: JSON.stringify({
        original,
        candidate,
        provider,
        model,
        validator_prompt: validatorPrompt,
      }),
    }),
  paraphraseSystemPrompt: (
    original: string,
    provider: string,
    model: string,
    paraphraserPrompt: string,
  ) =>
    request<{ candidate: string; missing_terms: string[] }>(
      "/api/system-prompts/paraphrase",
      {
        method: "POST",
        body: JSON.stringify({
          original,
          provider,
          model,
          paraphraser_prompt: paraphraserPrompt,
        }),
      },
    ),
  paraphraseSelectedText: (
    selectedText: string,
    provider: string,
    model: string,
  ) =>
    request<{ replacement: string }>("/api/text/paraphrase", {
      method: "POST",
      body: JSON.stringify({ selected_text: selectedText, provider, model }),
    }),
  updateExample: (exampleId: string, example: ExampleDraft) =>
    request<Example>(`/api/examples/${exampleId}`, {
      method: "PUT",
      body: JSON.stringify(example),
    }),
  deleteExample: (exampleId: string) =>
    request<{ deleted: number; trash_id: string }>(
      `/api/examples/${exampleId}`,
      { method: "DELETE" },
    ),
  restoreTrash: (trashId: string) =>
    request<{ restored: number }>(`/api/trash/${trashId}/restore`, {
      method: "POST",
    }),
  bulkTransform: (exampleIds: string[], transform: string, dryRun: boolean) =>
    request<BulkTransformResult>("/api/examples/bulk/transform", {
      method: "POST",
      body: JSON.stringify({
        example_ids: exampleIds,
        transform,
        dry_run: dryRun,
      }),
    }),
  revertRevision: (revisionId: string) =>
    request<{ reverted: number }>(`/api/revisions/${revisionId}/revert`, {
      method: "POST",
    }),
  bulkSetFlag: (exampleIds: string[], flag: ExampleFlag) =>
    request<{ updated: number }>("/api/examples/bulk/flag", {
      method: "POST",
      body: JSON.stringify({ example_ids: exampleIds, flag }),
    }),
  bulkSetSplit: (
    exampleIds: string[],
    split: "train" | "validation" | "test",
  ) =>
    request<{ updated: number }>("/api/examples/bulk/split", {
      method: "POST",
      body: JSON.stringify({ example_ids: exampleIds, split }),
    }),
  bulkSetSystemPrompt: (
    exampleIds: string[],
    prompt: string,
    every: number,
    original: string,
  ) =>
    request<{ updated: number; skipped: number }>(
      "/api/examples/bulk/system-prompt",
      {
        method: "POST",
        body: JSON.stringify({
          example_ids: exampleIds,
          prompt,
          every,
          original,
        }),
      },
    ),
  bulkDelete: (exampleIds: string[]) =>
    request<{ deleted: number; trash_id: string | null }>(
      "/api/examples/bulk/delete",
      {
        method: "POST",
        body: JSON.stringify({ example_ids: exampleIds }),
      },
    ),
  bulkClassify: (exampleIds: string[]) =>
    request<{ classified: number; needs_review: number; missing: number }>(
      "/api/examples/bulk/classify",
      {
        method: "POST",
        body: JSON.stringify({ example_ids: exampleIds }),
      },
    ),
  classificationStatus: () =>
    request<ClassificationStatus>("/api/examples/classification/status"),
  startAutomaticClassification: (exampleIds: string[]) =>
    request<ClassificationStatus>("/api/examples/classification/start", {
      method: "POST",
      body: JSON.stringify({ example_ids: exampleIds }),
    }),
  reviewExample: (exampleId: string) =>
    request<ExampleReview>(`/api/examples/${exampleId}/review`, {
      method: "POST",
    }),
  chat: (prompt: string) =>
    request<{ response: string }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),
  chatMessages: (messages: Message[]) =>
    request<{ response: string }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ messages }),
    }),
  models: () => request<{ models: string[] }>("/api/models"),
  trainingStatus: (corpusId?: string) =>
    request<TrainingStatus>(
      `/api/training/status${corpusId ? `?corpus_id=${encodeURIComponent(corpusId)}` : ""}`,
    ),
  trainingModels: () => request<{ models: string[] }>("/api/training/models"),
  trainingRuns: () => request<TrainingRunSummary[]>("/api/training/runs"),
  trainingRun: (runId: string) =>
    request<TrainingStatus>(`/api/training/runs/${encodeURIComponent(runId)}`),
  startTraining: ({ corpusId, baseModel, adapterName }: TrainingStart) =>
    request<TrainingStatus>("/api/training/start", {
      method: "POST",
      body: JSON.stringify({
        corpus_id: corpusId,
        base_model: baseModel,
        adapter_name: adapterName,
      }),
    }),
  stopTraining: () =>
    request<TrainingStatus>("/api/training/stop", { method: "POST" }),
  evaluationAdapters: () =>
    request<{ adapters: Record<string, string[]> }>("/api/evaluation/adapters"),
  evaluationStatus: () => request<EvaluationStatus>("/api/evaluation/status"),
  startEvaluation: (
    corpusId: string,
    adapterName: string,
    checkpoints: string[],
    splits: Array<"train" | "validation" | "test">,
  ) =>
    request<EvaluationStatus>("/api/evaluation/start", {
      method: "POST",
      body: JSON.stringify({
        corpus_id: corpusId,
        adapter_name: adapterName,
        checkpoints,
        splits,
      }),
    }),
  stopEvaluation: () =>
    request<EvaluationStatus>("/api/evaluation/stop", { method: "POST" }),
  servingStatus: () => request<ServingStatus>("/api/serving/status"),
  deployCheckpoint: (adapterName: string, checkpoint: string) =>
    request<ServingStatus>("/api/serving/deploy", {
      method: "POST",
      body: JSON.stringify({ adapter_name: adapterName, checkpoint }),
    }),
  stopServing: () =>
    request<ServingStatus>("/api/serving/stop", { method: "POST" }),
  exports: () => request<ExportsStatus>("/api/exports"),
  startExport: (
    adapterName: string,
    checkpoint: string,
    quantization: ExportQuantization,
    modelName: string,
  ) =>
    request<ExportsStatus>("/api/exports", {
      method: "POST",
      body: JSON.stringify({
        adapter_name: adapterName,
        checkpoint,
        quantization,
        model_name: modelName || null,
      }),
    }),
  deleteExport: (exportId: string) =>
    request<ExportsStatus>(`/api/exports/${encodeURIComponent(exportId)}`, {
      method: "DELETE",
    }),
  chatStream: async (
    messages: Message[],
    model: string,
    onContent: (content: string) => void,
  ) => {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, model, stream: true }),
    });
    if (!response.ok || !response.body) {
      throw new Error("Nie udało się uruchomić strumienia odpowiedzi.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let remainder = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      remainder += decoder.decode(value, { stream: true });
      const lines = remainder.split("\n");
      remainder = lines.pop() ?? "";
      lines
        .filter(Boolean)
        .forEach((line) => onContent(JSON.parse(line).content));
    }
  },
};
