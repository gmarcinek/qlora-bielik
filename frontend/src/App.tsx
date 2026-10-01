import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronDown,
  ChevronUp,
  Database,
  Download,
  FilePlus2,
  FileUp,
  GraduationCap,
  MessageSquareText,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
import {
  HashRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  api,
  Corpus,
  Example,
  ExampleFlag,
  ExampleReview,
  ImportedExample,
  Message,
  MessageRole,
  ParaphraseProviderCatalog,
  TrainingMetric,
  TrainingStatus,
} from "./api";
import { PageTemplate } from "./components/PageTemplate";
import { MediumPageTemplate } from "./components/MediumPageTemplate";

type Split = "train" | "validation" | "test";
type Draft = Message & { id: string };
type ImportTarget =
  | "messages"
  | "system"
  | "user"
  | "assistant"
  | "split"
  | "flag"
  | "source";
type ImportMapping = Record<ImportTarget, string>;
type ImportAdapter = "mapping" | "owu-annotations";
type ImportSession = {
  records: Record<string, unknown>[];
  keys: string[];
  mapping: ImportMapping;
  adapter: ImportAdapter;
};

const draft = (role: MessageRole = "user"): Draft => ({
  id: crypto.randomUUID(),
  role,
  content: "",
});

const importTargets: Array<{ key: ImportTarget; label: string }> = [
  { key: "messages", label: "messages" },
  { key: "system", label: "system" },
  { key: "user", label: "user" },
  { key: "assistant", label: "assistant" },
  { key: "split", label: "split" },
  { key: "flag", label: "flag" },
  { key: "source", label: "source" },
];

const importKeyCandidates: Record<ImportTarget, string[]> = {
  messages: ["messages"],
  system: ["system", "system_prompt", "instruction"],
  user: ["user", "prompt", "question", "input", "query"],
  assistant: ["assistant", "answer", "response", "output", "completion"],
  split: ["split", "dataset_split", "partition"],
  flag: ["flag", "label", "class"],
  source: ["source", "origin"],
};

function detectImportMapping(keys: string[]): ImportMapping {
  const normalizedKeys = new Map(keys.map((key) => [key.toLowerCase(), key]));
  return Object.fromEntries(
    importTargets.map(({ key }) => [
      key,
      importKeyCandidates[key]
        .map((candidate) => normalizedKeys.get(candidate))
        .find(Boolean) ?? "",
    ]),
  ) as ImportMapping;
}

function parseImportFile(content: string): Record<string, unknown>[] {
  const record = (value: unknown, index: number) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`Rekord ${index + 1} musi być obiektem JSON.`);
    }
    return value as Record<string, unknown>;
  };
  const trimmed = content.trim();
  if (!trimmed) throw new Error("Plik nie zawiera encji JSONL.");
  if (trimmed.startsWith("[")) {
    const records = JSON.parse(trimmed) as unknown;
    if (!Array.isArray(records))
      throw new Error("Plik JSON musi zawierać tablicę encji.");
    return records.map(record);
  }
  return trimmed.split(/\r?\n/).map((line, index) => {
    try {
      return record(JSON.parse(line), index);
    } catch {
      throw new Error(`Niepoprawny JSON w linii ${index + 1}.`);
    }
  });
}

function isOwuAnnotationDto(records: Record<string, unknown>[]): boolean {
  return records.every(
    (record) =>
      typeof record.task === "string" &&
      typeof record.text === "string" &&
      record.target !== undefined,
  );
}

function isCorpusDto(records: Record<string, unknown>[]): boolean {
  return records.every((record) => {
    const messages = record.messages;
    return (
      Array.isArray(messages) &&
      messages.length >= 2 &&
      messages.every(
        (message) =>
          message &&
          typeof message === "object" &&
          ["system", "user", "assistant"].includes(
            (message as Record<string, unknown>).role as string,
          ) &&
          typeof (message as Record<string, unknown>).content === "string" &&
          Boolean((message as Record<string, unknown>).content),
      ) &&
      (messages.at(-1) as Record<string, unknown>).role === "assistant"
    );
  });
}

function mapImportRecord(
  record: Record<string, unknown>,
  mapping: ImportMapping,
  index: number,
): ImportedExample {
  const value = (target: ImportTarget) =>
    mapping[target] ? record[mapping[target]] : undefined;
  const text = (target: ImportTarget) => {
    const mapped = value(target);
    return typeof mapped === "string" ? mapped.trim() : "";
  };
  const importedMessages = value("messages");
  const messages: Message[] = Array.isArray(importedMessages)
    ? (importedMessages as Message[])
    : [
        ...(text("system")
          ? [{ role: "system" as const, content: text("system") }]
          : []),
        ...(text("user")
          ? [{ role: "user" as const, content: text("user") }]
          : []),
        ...(text("assistant")
          ? [{ role: "assistant" as const, content: text("assistant") }]
          : []),
      ];
  if (!messages.length) {
    throw new Error(`Brak mapowania wiadomości w rekordzie ${index + 1}.`);
  }
  const split = text("split");
  const flag = text("flag");
  if (split && !["train", "validation", "test"].includes(split)) {
    throw new Error(`Niepoprawny split w rekordzie ${index + 1}.`);
  }
  if (flag && !["positive", "negative", "unclassified"].includes(flag)) {
    throw new Error(`Niepoprawna klasa w rekordzie ${index + 1}.`);
  }
  return {
    messages,
    ...(split ? { split: split as Split } : {}),
    ...(flag ? { flag: flag as ExampleFlag } : {}),
    ...(text("source") ? { source: text("source") } : {}),
  };
}

function mapOwuAnnotationRecord(
  record: Record<string, unknown>,
  index: number,
): ImportedExample {
  const task = typeof record.task === "string" ? record.task : "";
  const text = typeof record.text === "string" ? record.text.trim() : "";
  if (!text || record.target === undefined) {
    throw new Error(`Niepełny rekord OWU annotations: ${index + 1}.`);
  }
  const labels = Array.isArray(record.labels)
    ? record.labels.filter(
        (label): label is string => typeof label === "string",
      )
    : [];
  const origin = typeof record.origin === "string" ? record.origin : "";
  const documentId =
    typeof record.document_id === "string" ? record.document_id : "";
  const sourceKey =
    typeof record.source_key === "string" ? record.source_key : "";
  const system =
    task === "ner"
      ? `Rozpoznaj w podanym fragmencie wszystkie encje typów: ${labels.join(", ")}. Zwróć wyłącznie JSON z kluczem entities.`
      : "Wyodrębnij z podanego fragmentu wszystkie klauzule wyłączenia ochrony lub wypłaty. Zwróć wyłącznie JSON zgodny z zadaniem.";
  const target = record.target;
  const source = [documentId, sourceKey, origin].filter(Boolean).join(" / ");
  return {
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: `${origin ? `Źródło: ${origin}\n` : ""}<tekst>${text}</tekst>`,
      },
      { role: "assistant", content: JSON.stringify(target) },
    ],
    ...(source ? { source } : {}),
  };
}

function CorporaPage({
  corpora,
  onCreated,
  onCorpusUpdated,
  openCreate = false,
}: {
  corpora: Corpus[];
  onCreated: (corpus: Corpus) => void;
  onCorpusUpdated: () => void;
  openCreate?: boolean;
}) {
  const navigate = useNavigate();
  const { corpusId } = useParams();
  const selectedCorpus = corpora.find((corpus) => corpus.id === corpusId);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [examplesLoading, setExamplesLoading] = useState(false);
  const [selectedExample, setSelectedExample] = useState<Example | null>(null);
  const [drawerMessages, setDrawerMessages] = useState<Message[]>([]);
  const [drawerEditing, setDrawerEditing] = useState(false);
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(
    null,
  );
  const [drawerError, setDrawerError] = useState("");
  const [review, setReview] = useState<ExampleReview | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const [classification, setClassification] =
    useState<ClassificationStatus | null>(null);
  const [query, setQuery] = useState("");
  const [splitFilter, setSplitFilter] = useState<Split | "">("");
  const [flagFilter, setFlagFilter] = useState<ExampleFlag | "">("");
  const [importFilter, setImportFilter] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkSplit, setBulkSplit] = useState<Split>("train");
  const [exportSplit, setExportSplit] = useState<Split>("train");
  const [importSplit, setImportSplit] = useState<Split>("train");
  const [activeView, setActiveView] = useState<"list" | "duplicates">("list");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [importNotice, setImportNotice] = useState("");
  const [importError, setImportError] = useState("");
  const [importSession, setImportSession] = useState<ImportSession | null>(
    null,
  );
  const [manualPromptOpen, setManualPromptOpen] = useState(false);
  const [manualPromptSource, setManualPromptSource] = useState<
    "manual" | "randomized"
  >("manual");
  const [manualPrompt, setManualPrompt] = useState("");
  const [manualOriginalPrompt, setManualOriginalPrompt] = useState("");
  const [manualPromptIds, setManualPromptIds] = useState<string[]>([]);
  const [manualEvery, setManualEvery] = useState(2);
  const [manualComparison, setManualComparison] = useState<{
    semantic_equivalent: boolean;
    instruction_plan_equivalent: boolean;
    reason: string;
  } | null>(null);
  const [comparingManualPrompt, setComparingManualPrompt] = useState(false);
  const [paraphrasingPrompt, setParaphrasingPrompt] = useState(false);
  const [paraphraseProviders, setParaphraseProviders] = useState<
    ParaphraseProviderCatalog["providers"]
  >({});
  const [paraphraseProvider, setParaphraseProvider] = useState("openai");
  const [paraphraseModel, setParaphraseModel] = useState("gpt-4.1");
  const [promptSelection, setPromptSelection] = useState<{
    start: number;
    end: number;
  } | null>(null);
  const [paraphrasingSelection, setParaphrasingSelection] = useState(false);
  const [promptModalError, setPromptModalError] = useState("");
  const [missingPromptTerms, setMissingPromptTerms] = useState<string[]>([]);
  const [promptFlash, setPromptFlash] = useState(false);
  const manualPromptRef = useRef<HTMLTextAreaElement>(null);
  const [validatorPrompt, setValidatorPrompt] = useState("");
  const [paraphraserPrompt, setParaphraserPrompt] = useState("");
  const [promptTab, setPromptTab] = useState<"paraphraser" | "validator">(
    "validator",
  );
  const paraphraseModels =
    paraphraseProviders[paraphraseProvider]?.models ?? {};
  useEffect(() => {
    setOpen(openCreate);
  }, [openCreate]);
  useEffect(() => {
    setExamplesLoading(true);
    setSelectedExample(null);
    const request = corpusId ? api.examples(corpusId) : api.allExamples();
    request
      .then((items) => {
        setExamples(items);
        setSelectedExample(null);
        setSelectedIds(new Set());
      })
      .catch(() => {
        setExamples([]);
        setSelectedExample(null);
      })
      .finally(() => setExamplesLoading(false));
  }, [corpusId]);
  useEffect(() => {
    if (classification?.state !== "running") return;
    const interval = window.setInterval(() => {
      void api.classificationStatus().then((job) => {
        setClassification(job);
        if (job.state !== "running") {
          const request = corpusId ? api.examples(corpusId) : api.allExamples();
          void request.then(setExamples);
        }
      });
    }, 1500);
    return () => window.clearInterval(interval);
  }, [classification?.state, corpusId]);
  useEffect(() => {
    if (!promptSelection || !manualPromptRef.current) return;
    manualPromptRef.current.focus();
    manualPromptRef.current.setSelectionRange(
      promptSelection.start,
      promptSelection.end,
    );
  }, [manualPrompt, promptSelection]);
  useEffect(() => {
    if (!manualPromptOpen || validatorPrompt) return;
    void api
      .systemPromptValidator()
      .then(({ prompt }) => setValidatorPrompt(prompt));
  }, [manualPromptOpen, validatorPrompt]);
  useEffect(() => {
    if (!manualPromptOpen || paraphraserPrompt) return;
    void api
      .systemPromptParaphraser()
      .then(({ prompt }) => setParaphraserPrompt(prompt));
  }, [manualPromptOpen, paraphraserPrompt]);
  useEffect(() => {
    if (!manualPromptOpen || Object.keys(paraphraseProviders).length) return;
    void api.paraphraseProviders().then(({ providers }) => {
      setParaphraseProviders(providers);
      const provider = providers[paraphraseProvider]
        ? paraphraseProvider
        : Object.keys(providers)[0];
      const models = providers[provider]?.models ?? {};
      setParaphraseProvider(provider);
      if (!models[paraphraseModel]) {
        setParaphraseModel(Object.keys(models)[0] ?? "");
      }
    });
  }, [
    manualPromptOpen,
    paraphraseModel,
    paraphraseProvider,
    paraphraseProviders,
  ]);
  useEffect(() => {
    if (!manualPromptOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setManualPromptOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [manualPromptOpen]);
  async function submit() {
    setBusy(true);
    try {
      const corpus = await api.createCorpus(name.trim(), description.trim());
      onCreated(corpus);
      navigate(`/builder/${corpus.id}`);
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  }
  async function importEntities(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !selectedCorpus) return;
    setBusy(true);
    setImportNotice("");
    setImportError("");
    try {
      const records = parseImportFile(await file.text());
      if (!records.length) throw new Error("Plik nie zawiera encji JSONL.");
      if (isCorpusDto(records)) {
        await saveImportedExamples(records as ImportedExample[]);
        return;
      }
      const keys = [
        ...new Set(records.flatMap((record) => Object.keys(record))),
      ];
      setImportSession({
        records,
        keys,
        mapping: detectImportMapping(keys),
        adapter: isOwuAnnotationDto(records) ? "owu-annotations" : "mapping",
      });
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się odczytać pliku.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function saveImportedExamples(examples: ImportedExample[]) {
    if (!selectedCorpus) return;
    const result = await api.importExamples(
      selectedCorpus.id,
      examples.map((example) => ({
        ...example,
        split: example.split ?? importSplit,
      })),
    );
    setExamples(await api.examples(selectedCorpus.id));
    onCorpusUpdated();
    setFlagFilter("unclassified");
    setImportFilter(result.import_id);
    setImportNotice(
      `Zaimportowano encje: ${result.imported}. Domyślny split: ${importSplit}. Partia: ${result.import_id}. Następne zadanie: klasyfikacja positive/negative.`,
    );
  }
  function openManualPromptVariant(
    targetPrompt?: string,
    targetItems?: Example[],
  ) {
    const groups = new Map<string, Example[]>();
    examples.forEach((example) => {
      const prompt = example.messages.find(
        (message) => message.role === "system",
      )?.content;
      if (prompt) groups.set(prompt, [...(groups.get(prompt) ?? []), example]);
    });
    const group =
      targetPrompt && targetItems
        ? [targetPrompt, targetItems]
        : [...groups.entries()]
            .filter(([, items]) => items.length > 15)
            .sort(([, left], [, right]) => right.length - left.length)[0];
    if (!group) {
      setImportError(
        "Nie znaleziono grupy z więcej niż 15 identycznymi system promptami.",
      );
      return;
    }
    const [prompt, items] = group;
    setManualPromptSource("manual");
    setManualPrompt(prompt);
    setManualOriginalPrompt(prompt);
    setManualPromptIds(items.map((item) => item.id));
    setManualEvery(2);
    setManualComparison(null);
    setPromptModalError("");
    setMissingPromptTerms([]);
    setManualPromptOpen(true);
  }
  async function compareManualPrompt() {
    if (!manualPrompt.trim()) return;
    setComparingManualPrompt(true);
    setPromptModalError("");
    try {
      setManualComparison(
        await api.compareSystemPrompts(
          manualOriginalPrompt,
          manualPrompt.trim(),
          paraphraseProvider,
          paraphraseModel,
          validatorPrompt,
        ),
      );
    } catch (error) {
      setPromptModalError(
        error instanceof Error
          ? error.message
          : "Nie udało się porównać promptów.",
      );
    } finally {
      setComparingManualPrompt(false);
    }
  }
  async function paraphraseManualPrompt() {
    if (!manualPrompt.trim()) return;
    setParaphrasingPrompt(true);
    setPromptModalError("");
    setMissingPromptTerms([]);
    try {
      const { candidate, missing_terms } = await api.paraphraseSystemPrompt(
        manualOriginalPrompt,
        paraphraseProvider,
        paraphraseModel,
        paraphraserPrompt,
      );
      setManualPrompt(candidate);
      setMissingPromptTerms(missing_terms);
      setManualComparison(null);
      setPromptSelection(null);
    } catch (error) {
      setPromptModalError(
        error instanceof Error
          ? error.message
          : "Nie udało się sparafrazować skrócenia.",
      );
    } finally {
      setParaphrasingPrompt(false);
    }
  }
  async function paraphraseSelectedPrompt() {
    if (!promptSelection || promptSelection.start === promptSelection.end)
      return;
    const selectedText = manualPrompt.slice(
      promptSelection.start,
      promptSelection.end,
    );
    setParaphrasingSelection(true);
    setPromptModalError("");
    try {
      const { replacement } = await api.paraphraseSelectedText(
        selectedText,
        paraphraseProvider,
        paraphraseModel,
      );
      setManualPrompt(
        `${manualPrompt.slice(0, promptSelection.start)}${replacement}${manualPrompt.slice(promptSelection.end)}`,
      );
      setPromptSelection({
        start: promptSelection.start,
        end: promptSelection.start + replacement.length,
      });
      setManualComparison(null);
      setPromptFlash(true);
      window.setTimeout(() => setPromptFlash(false), 900);
    } catch (error) {
      setPromptModalError(
        error instanceof Error
          ? error.message
          : "Nie udało się sparafrazować zaznaczenia.",
      );
    } finally {
      setParaphrasingSelection(false);
    }
  }
  async function applyManualPromptVariant() {
    if (!manualPrompt.trim() || !manualPromptIds.length) return;
    setBusy(true);
    setPromptModalError("");
    try {
      const result = await api.bulkSetSystemPrompt(
        manualPromptIds,
        manualPrompt.trim(),
        manualEvery,
        manualOriginalPrompt,
      );
      setExamples(await api.examples(selectedCorpus?.id ?? ""));
      setManualPromptOpen(false);
      setImportNotice(
        `Zapisano ręcznie zmieniony system prompt w ${result.updated} encjach. Pominięto jako niezgodne: ${result.skipped}.`,
      );
    } catch (error) {
      setPromptModalError(
        error instanceof Error
          ? error.message
          : "Nie udało się zapisać promptu.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function confirmImport() {
    if (!importSession || !selectedCorpus) return;
    setBusy(true);
    setImportError("");
    try {
      const examples = importSession.records.map((record, index) =>
        importSession.adapter === "owu-annotations"
          ? mapOwuAnnotationRecord(record, index)
          : mapImportRecord(record, importSession.mapping, index),
      );
      await saveImportedExamples(examples);
      setImportSession(null);
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się zaimportować encji.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function removeExample(example: Example) {
    if (!window.confirm("Usunąć tę encję?")) return;
    setBusy(true);
    try {
      await api.deleteExample(example.id);
      setExamples((current) =>
        current.filter((item) => item.id !== example.id),
      );
      setSelectedExample((current) =>
        current?.id === example.id ? null : current,
      );
    } finally {
      setBusy(false);
    }
  }
  const formatCreatedAt = (createdAt: string) => {
    const date = new Date(createdAt);
    const pad = (value: number) => String(value).padStart(2, "0");
    return `${date.getFullYear()}.${pad(date.getMonth() + 1)}.${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  };
  function openDrawer(example: Example) {
    setSelectedExample(example);
    setDrawerMessages(example.messages.map((message) => ({ ...message })));
    setDrawerEditing(false);
    setEditingMessageIndex(null);
    setDrawerError("");
    setReview(null);
  }
  function closeDrawer() {
    setSelectedExample(null);
    setDrawerEditing(false);
    setEditingMessageIndex(null);
    setDrawerError("");
    setReview(null);
  }
  function updateDrawerMessage(index: number, change: Partial<Message>) {
    setDrawerMessages((current) =>
      current.map((message, messageIndex) =>
        messageIndex === index ? { ...message, ...change } : message,
      ),
    );
  }
  function moveDrawerMessage(index: number, direction: -1 | 1) {
    const targetIndex = index + direction;
    setDrawerEditing(true);
    setDrawerMessages((current) => {
      if (targetIndex < 0 || targetIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  }
  function addDrawerMessage() {
    setDrawerEditing(true);
    setDrawerMessages((current) => {
      const completion = current.at(-1);
      const message: Message = { role: "user", content: "" };
      return completion?.role === "assistant"
        ? [...current.slice(0, -1), message, completion]
        : [...current, message];
    });
  }
  function removeDrawerMessage(index: number) {
    setDrawerEditing(true);
    setEditingMessageIndex((current) => {
      if (current === null || current === index) return null;
      return current > index ? current - 1 : current;
    });
    setDrawerMessages((current) =>
      current.filter((_, messageIndex) => messageIndex !== index),
    );
  }
  async function saveDrawer(asCopy = false) {
    if (!selectedExample) return;
    const targetCorpusId = selectedExample.corpus_id ?? corpusId;
    if (asCopy && !targetCorpusId) {
      setDrawerError("Nie można ustalić korpusu dla kopii encji.");
      return;
    }
    if (
      drawerMessages.length < 2 ||
      drawerMessages.some((message) => !message.content.trim()) ||
      drawerMessages.at(-1)?.role !== "assistant"
    ) {
      setDrawerError("Wypełnij wiadomości, a ostatnią ustaw jako assistant.");
      return;
    }
    setBusy(true);
    setDrawerError("");
    try {
      const payload = {
        split: selectedExample.split,
        messages: drawerMessages,
        flag: selectedExample.metadata.flag ?? "unclassified",
      };
      const updated = asCopy
        ? await api.createExample(targetCorpusId, payload)
        : await api.updateExample(selectedExample.id, payload);
      const merged = {
        ...selectedExample,
        ...updated,
        corpus_name: selectedExample.corpus_name,
        metadata: { flag: payload.flag },
      };
      setExamples((current) =>
        asCopy
          ? [merged, ...current]
          : current.map((example) =>
              example.id === merged.id ? merged : example,
            ),
      );
      setSelectedExample(merged);
      setDrawerMessages(merged.messages);
      setDrawerEditing(false);
      setEditingMessageIndex(null);
    } catch (error) {
      setDrawerError(
        error instanceof Error ? error.message : "Nie udało się zapisać encji.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function reviewSelectedExample() {
    if (!selectedExample) return;
    setReviewing(true);
    setDrawerError("");
    try {
      setReview(await api.reviewExample(selectedExample.id));
    } catch (error) {
      setDrawerError(
        error instanceof Error
          ? error.message
          : "Nie udało się sprawdzić encji.",
      );
    } finally {
      setReviewing(false);
    }
  }
  async function applyReviewFlag(flag: ExampleFlag) {
    if (!selectedExample) return;
    setBusy(true);
    setDrawerError("");
    try {
      const updated = await api.updateExample(selectedExample.id, {
        split: selectedExample.split,
        messages: selectedExample.messages,
        flag,
      });
      const merged = { ...selectedExample, ...updated, metadata: { flag } };
      setExamples((current) =>
        current.map((example) => (example.id === merged.id ? merged : example)),
      );
      setSelectedExample(merged);
      setReview(null);
    } catch (error) {
      setDrawerError(
        error instanceof Error
          ? error.message
          : "Nie udało się zapisać etykiety.",
      );
    } finally {
      setBusy(false);
    }
  }
  function toggleSelection(exampleId: string) {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(exampleId)) next.delete(exampleId);
      else next.add(exampleId);
      return next;
    });
  }
  async function applyBulkFlag(flag: ExampleFlag) {
    if (!selectedIds.size) return;
    setBusy(true);
    try {
      const exampleIds = [...selectedIds];
      await api.bulkSetFlag(exampleIds, flag);
      setExamples((current) =>
        current.map((example) =>
          exampleIds.includes(example.id)
            ? { ...example, metadata: { ...example.metadata, flag } }
            : example,
        ),
      );
      setSelectedIds(new Set());
    } finally {
      setBusy(false);
    }
  }
  async function applyBulkSplit() {
    if (!selectedIds.size) return;
    setBusy(true);
    try {
      const exampleIds = [...selectedIds];
      await api.bulkSetSplit(exampleIds, bulkSplit);
      setExamples((current) =>
        current.map((example) =>
          exampleIds.includes(example.id)
            ? { ...example, split: bulkSplit }
            : example,
        ),
      );
      setSelectedIds(new Set());
    } finally {
      setBusy(false);
    }
  }
  async function classifySelected() {
    if (!selectedIds.size) return;
    setBusy(true);
    setImportError("");
    try {
      const result = await api.bulkClassify([...selectedIds]);
      const refreshed = corpusId
        ? await api.examples(corpusId)
        : await api.allExamples();
      setExamples(refreshed);
      setSelectedIds(new Set());
      setImportNotice(
        `Sklasyfikowano: ${result.classified}. Do ręcznej oceny: ${result.needs_review}.`,
      );
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się sklasyfikować encji.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function startAutomaticClassification() {
    const exampleIds = filteredExamples
      .filter((example) => example.metadata.flag === "unclassified")
      .map((example) => example.id);
    if (!exampleIds.length) return;
    setBusy(true);
    setImportError("");
    try {
      setClassification(await api.startAutomaticClassification(exampleIds));
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się uruchomić automatu.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function deleteSelected() {
    if (
      !selectedIds.size ||
      !window.confirm(`Usunąć zaznaczone encje: ${selectedIds.size}?`)
    )
      return;
    setBusy(true);
    try {
      const exampleIds = [...selectedIds];
      await api.bulkDelete(exampleIds);
      setExamples((current) =>
        current.filter((example) => !exampleIds.includes(example.id)),
      );
      setSelectedIds(new Set());
      setSelectedExample((current) =>
        current && exampleIds.includes(current.id) ? null : current,
      );
    } finally {
      setBusy(false);
    }
  }
  const importBatches = [
    ...new Set(
      examples
        .map((example) => example.metadata.import_id)
        .filter((value): value is string => Boolean(value)),
    ),
  ];
  const filteredExamples = examples.filter((example) => {
    const createdAt = new Date(example.created_at);
    const from = fromDate ? new Date(`${fromDate}T00:00:00`) : null;
    const to = toDate ? new Date(`${toDate}T23:59:59.999`) : null;
    const searchable = [
      example.corpus_name ?? selectedCorpus?.name ?? "",
      ...example.messages.map((message) => message.content),
    ]
      .join(" ")
      .toLocaleLowerCase("pl-PL");
    return (
      (!query.trim() ||
        searchable.includes(query.trim().toLocaleLowerCase("pl-PL"))) &&
      (!splitFilter || example.split === splitFilter) &&
      (!flagFilter || example.metadata.flag === flagFilter) &&
      (!importFilter ||
        (importFilter === "manual"
          ? !example.metadata.import_id
          : example.metadata.import_id === importFilter)) &&
      (!from || createdAt >= from) &&
      (!to || createdAt <= to)
    );
  });
  const automaticCandidates = filteredExamples.filter(
    (example) => example.metadata.flag === "unclassified",
  );
  const systemPromptGroups = (() => {
    const groups = new Map<string, Example[]>();
    examples.forEach((example) => {
      const prompt = example.messages.find(
        (message) => message.role === "system",
      )?.content;
      if (prompt) groups.set(prompt, [...(groups.get(prompt) ?? []), example]);
    });
    return [...groups.entries()].sort(
      ([, left], [, right]) => right.length - left.length,
    );
  })();
  const duplicateSystemPromptGroups = systemPromptGroups.filter(
    ([, items]) => items.length > 15,
  );
  const uniqueSystemPromptCount = systemPromptGroups.filter(
    ([, items]) => items.length === 1,
  ).length;
  return (
    <MediumPageTemplate
      eyebrow="PRZEGLĄD"
      title={`Encje JSONL${selectedCorpus ? `: ${selectedCorpus.name}` : ""}`}
      actions={
        <div className="d-flex flex-column gap-2 align-self-start">
          <div className="d-flex flex-wrap gap-2">
            <div className="input-group input-group-sm export-control">
              <select
                className="form-select"
                value={exportSplit}
                disabled={!selectedCorpus}
                onChange={(event) =>
                  setExportSplit(event.target.value as Split)
                }
                aria-label="Split eksportu"
              >
                <option value="train">train</option>
                <option value="validation">validation</option>
                <option value="test">test</option>
              </select>
              {selectedCorpus ? (
                <a
                  className="btn btn-outline-primary"
                  href={`/api/corpora/${selectedCorpus.id}/export?split=${exportSplit}`}
                  download={`corpus-${selectedCorpus.id}-${exportSplit}.jsonl`}
                >
                  <Download size={17} className="me-1" /> Eksportuj
                </a>
              ) : (
                <button
                  className="btn btn-outline-primary"
                  type="button"
                  disabled
                >
                  <Download size={17} className="me-1" /> Eksportuj
                </button>
              )}
            </div>
            <div className="input-group input-group-sm export-control">
              <select
                className="form-select"
                value={importSplit}
                disabled={busy || !selectedCorpus}
                onChange={(event) =>
                  setImportSplit(event.target.value as Split)
                }
                aria-label="Domyślny split importu"
              >
                <option value="train">train</option>
                <option value="validation">validation</option>
                <option value="test">test</option>
              </select>
              <label className="btn btn-outline-primary mb-0">
                <FileUp size={17} className="me-1" /> Importuj JSONL
                <input
                  className="visually-hidden"
                  type="file"
                  accept=".jsonl,.ndjson,application/json"
                  disabled={busy || !selectedCorpus}
                  onChange={(event) => void importEntities(event)}
                />
              </label>
            </div>
            <button
              className="btn btn-primary"
              type="button"
              disabled={!selectedCorpus}
              onClick={() => navigate(`/builder/${selectedCorpus?.id}`)}
            >
              <FilePlus2 size={17} className="me-1" /> Dodaj encję
            </button>
          </div>
        </div>
      }
    >
      {importNotice && (
        <div className="alert alert-success">{importNotice}</div>
      )}
      {importError && <div className="alert alert-danger">{importError}</div>}
      <nav className="nav nav-tabs mb-3" aria-label="Widok encji">
        <button
          className={`nav-link ${activeView === "list" ? "active" : ""}`}
          type="button"
          onClick={() => setActiveView("list")}
        >
          Lista
        </button>
        <button
          className={`nav-link ${activeView === "duplicates" ? "active" : ""}`}
          type="button"
          onClick={() => setActiveView("duplicates")}
        >
          Duplikaty{" "}
          {duplicateSystemPromptGroups.length
            ? `(${duplicateSystemPromptGroups.length})`
            : ""}
        </button>
      </nav>
      {activeView === "list" && (
        <>
          <div className="row g-2 mb-3">
            <div className="col-12 col-md">
              <input
                className="form-control"
                placeholder="Filtruj treść encji"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <div className="col-6 col-md-auto">
              <select
                className="form-select"
                value={splitFilter}
                onChange={(event) =>
                  setSplitFilter(event.target.value as Split | "")
                }
              >
                <option value="">Wszystkie splity</option>
                <option value="train">train</option>
                <option value="validation">validation</option>
                <option value="test">test</option>
              </select>
            </div>
            <div className="col-6 col-md-auto">
              <select
                className="form-select"
                value={flagFilter}
                onChange={(event) =>
                  setFlagFilter(event.target.value as ExampleFlag | "")
                }
              >
                <option value="">Wszystkie klasy</option>
                <option value="unclassified">do klasyfikacji</option>
                <option value="positive">pozytywny</option>
                <option value="negative">negatywny</option>
              </select>
            </div>
          </div>
          <details className="advanced-filters mb-3">
            <summary>Filtry zaawansowane</summary>
            <div className="row g-2 mt-1">
              <div className="col-12 col-md-auto">
                <select
                  className="form-select"
                  value={importFilter}
                  onChange={(event) => setImportFilter(event.target.value)}
                >
                  <option value="">Wszystkie importy</option>
                  <option value="manual">Dodane ręcznie</option>
                  {importBatches.map((batch) => (
                    <option key={batch} value={batch}>
                      {batch}
                    </option>
                  ))}
                </select>
              </div>
              <div className="col-6 col-md-auto">
                <input
                  className="form-control"
                  type="date"
                  aria-label="Data od"
                  value={fromDate}
                  onChange={(event) => setFromDate(event.target.value)}
                />
              </div>
              <div className="col-6 col-md-auto">
                <input
                  className="form-control"
                  type="date"
                  aria-label="Data do"
                  value={toDate}
                  onChange={(event) => setToDate(event.target.value)}
                />
              </div>
            </div>
          </details>
          <div className="automatic-classification mb-3">
            <div>
              <p className="panel-title mb-1">AUTOMATYCZNA KLASYFIKACJA</p>
              <small className="text-secondary">
                Bielik przechodzi kolejno przez nieoznaczone encje z bieżących
                filtrów.
              </small>
            </div>
            {classification?.state === "running" ? (
              <strong>
                {classification.processed} / {classification.total}
              </strong>
            ) : (
              <button
                className="btn btn-sm btn-primary"
                type="button"
                disabled={busy || !automaticCandidates.length}
                onClick={() => void startAutomaticClassification()}
              >
                <Sparkles size={15} className="me-1" /> Klasyfikuj automatycznie
                ({automaticCandidates.length})
              </button>
            )}
            {classification?.state === "completed" && (
              <small className="text-secondary">
                Gotowe: {classification.classified}; do ręcznej oceny:{" "}
                {classification.needs_review}.
              </small>
            )}
            {classification?.state === "failed" && (
              <small className="text-danger">
                {classification.error || "Klasyfikacja nie powiodła się."}
              </small>
            )}
          </div>
          {selectedIds.size > 0 && (
            <div className="bulk-actions mb-3">
              <strong>Zaznaczone: {selectedIds.size}</strong>
              <div className="input-group input-group-sm bulk-split-control">
                <select
                  className="form-select"
                  value={bulkSplit}
                  disabled={busy}
                  onChange={(event) =>
                    setBulkSplit(event.target.value as Split)
                  }
                  aria-label="Docelowy split zaznaczonych encji"
                >
                  <option value="train">train</option>
                  <option value="validation">validation</option>
                  <option value="test">test</option>
                </select>
                <button
                  className="btn btn-outline-primary"
                  type="button"
                  disabled={busy}
                  onClick={() => void applyBulkSplit()}
                >
                  Ustaw split
                </button>
              </div>
              <button
                className="btn btn-sm btn-primary"
                type="button"
                disabled={busy || selectedIds.size > 100}
                onClick={() => void classifySelected()}
              >
                <Sparkles size={15} className="me-1" /> Klasyfikuj Bielikiem
              </button>
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                disabled={busy}
                onClick={() => void applyBulkFlag("unclassified")}
              >
                Do klasyfikacji
              </button>
              <button
                className="btn btn-sm btn-outline-success"
                type="button"
                disabled={busy}
                onClick={() => void applyBulkFlag("positive")}
              >
                Oznacz positive
              </button>
              <button
                className="btn btn-sm btn-outline-danger"
                type="button"
                disabled={busy}
                onClick={() => void applyBulkFlag("negative")}
              >
                Oznacz negative
              </button>
              <button
                className="btn btn-sm btn-danger"
                type="button"
                disabled={busy}
                onClick={() => void deleteSelected()}
              >
                <Trash2 size={15} className="me-1" /> Usuń
              </button>
              {selectedIds.size > 100 && (
                <small className="text-danger">
                  Maksymalnie 100 encji w jednym zadaniu klasyfikacji.
                </small>
              )}
            </div>
          )}
          {examplesLoading ? (
            <div className="text-secondary">Wczytywanie encji...</div>
          ) : filteredExamples.length ? (
            <div className="list-group shadow-sm">
              <label className="list-group-item d-flex align-items-center gap-2 entity-select-all">
                <input
                  type="checkbox"
                  checked={filteredExamples.every((example) =>
                    selectedIds.has(example.id),
                  )}
                  onChange={() =>
                    setSelectedIds((current) => {
                      const next = new Set(current);
                      const allSelected = filteredExamples.every((example) =>
                        next.has(example.id),
                      );
                      filteredExamples.forEach((example) =>
                        allSelected
                          ? next.delete(example.id)
                          : next.add(example.id),
                      );
                      return next;
                    })
                  }
                />
                Zaznacz widoczne ({filteredExamples.length})
              </label>
              {filteredExamples.map((example) => (
                <div
                  className={`list-group-item entity-list-item d-flex align-items-start gap-2 ${selectedExample?.id === example.id ? "selected" : ""}`}
                  key={example.id}
                >
                  <input
                    className="form-check-input mt-1"
                    type="checkbox"
                    checked={selectedIds.has(example.id)}
                    onChange={() => toggleSelection(example.id)}
                    aria-label="Zaznacz encję"
                  />
                  <button
                    className="entity-select text-start"
                    type="button"
                    onClick={() => openDrawer(example)}
                  >
                    <span className="entity-list-content">
                      <small className="entity-created d-block">
                        {formatCreatedAt(example.created_at)}
                      </small>
                      <span className="d-flex flex-wrap gap-2 mb-1">
                        <span className="badge text-bg-secondary">
                          {example.corpus_name ?? selectedCorpus?.name}
                        </span>
                        <span className="badge text-bg-light border text-dark">
                          {example.split}
                        </span>
                        <span
                          className={`badge text-bg-${example.metadata.flag === "negative" ? "danger" : example.metadata.flag === "positive" ? "success" : "warning"}`}
                        >
                          {example.metadata.flag === "negative"
                            ? "negatywny"
                            : example.metadata.flag === "positive"
                              ? "pozytywny"
                              : "do klasyfikacji"}
                        </span>
                        {example.metadata.import_id && (
                          <span className="badge text-bg-info">
                            {example.metadata.import_id}
                          </span>
                        )}
                      </span>
                      <small className="d-block text-truncate">
                        {example.messages.find(
                          (message) => message.role === "user",
                        )?.content || "Bez wiadomości użytkownika"}
                      </small>
                    </span>
                  </button>
                  <span className="d-flex align-items-start gap-2 ms-auto">
                    <button
                      className="btn btn-sm btn-outline-primary"
                      type="button"
                      title="Edytuj"
                      aria-label="Edytuj"
                      onClick={() =>
                        navigate(
                          `/builder/${example.corpus_id ?? corpusId}?edit=${example.id}`,
                        )
                      }
                    >
                      <Pencil size={15} />
                    </button>
                    <button
                      className="btn btn-sm btn-outline-danger"
                      type="button"
                      title="Usuń"
                      aria-label="Usuń"
                      disabled={busy}
                      onClick={() => void removeExample(example)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-secondary">
              Brak encji dla wybranych filtrów.
            </div>
          )}
        </>
      )}
      {activeView === "duplicates" && (
        <section>
          <div className="d-flex flex-wrap gap-2 mb-3">
            <span className="badge text-bg-secondary">
              Unikalne prompty: {systemPromptGroups.length}
            </span>
            <span className="badge text-bg-light border text-dark">
              Encje z promptem występującym raz: {uniqueSystemPromptCount}
            </span>
          </div>
          <div className="d-flex flex-wrap gap-2 mb-3">
            <small className="text-secondary">
              Wybierz automatyczną albo ręczną zmianę promptu dla konkretnej
              grupy.
            </small>
          </div>
          {duplicateSystemPromptGroups.length ? (
            <div className="list-group shadow-sm">
              {duplicateSystemPromptGroups.map(([prompt, items]) => (
                <div className="list-group-item" key={prompt}>
                  <div className="d-flex justify-content-between gap-3">
                    <div className="text-truncate flex-grow-1">
                      <strong>{items.length} encji</strong>
                      <small className="d-block text-secondary text-truncate">
                        {prompt}
                      </small>
                    </div>
                    <div className="d-flex align-items-center gap-2 flex-shrink-0">
                      <span className="badge text-bg-warning">duplikat</span>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        type="button"
                        disabled={busy}
                        onClick={() => openManualPromptVariant(prompt, items)}
                      >
                        <Pencil size={15} className="me-1" /> Deduplikuj ręcznie
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-secondary">
              Brak grup z więcej niż 15 identycznymi system promptami.
            </div>
          )}
        </section>
      )}
      {importSession && (
        <div className="modal-backdrop show confirm-backdrop">
          <div className="modal d-block" role="dialog" aria-modal="true">
            <div className="modal-dialog modal-lg">
              <div className="modal-content">
                <div className="modal-header">
                  <h2 className="h5 modal-title">Mapowanie importu</h2>
                </div>
                <div className="modal-body">
                  <p className="mb-3">
                    Wykryto rekordy:{" "}
                    <strong>{importSession.records.length}</strong>
                  </p>
                  <p className="text-secondary small">
                    {importSession.adapter === "owu-annotations"
                      ? "Wykryto DTO OWU annotations. Importer zbuduje wiadomości z task, labels, text i target."
                      : "Wybierz klucze wejściowego DTO dla pól encji korpusu. Gdy mapujesz messages, pola ról są ignorowane."}
                  </p>
                  {importSession.adapter === "owu-annotations" ? (
                    <div className="alert alert-info mb-0">
                      <code>target</code> zostanie zapisany jako odpowiedź
                      asystenta, a pusty wynik jako klasa <code>negative</code>.
                    </div>
                  ) : (
                    <div className="row g-3">
                      {importTargets.map(({ key, label }) => (
                        <label className="col-12 col-md-6" key={key}>
                          <span className="form-label">{label}</span>
                          <select
                            className="form-select"
                            value={importSession.mapping[key]}
                            onChange={(event) =>
                              setImportSession((current) =>
                                current
                                  ? {
                                      ...current,
                                      mapping: {
                                        ...current.mapping,
                                        [key]: event.target.value,
                                      },
                                    }
                                  : null,
                              )
                            }
                          >
                            <option value="">Nie mapuj</option>
                            {importSession.keys.map((sourceKey) => (
                              <option key={sourceKey} value={sourceKey}>
                                {sourceKey}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                  )}
                  {importError && (
                    <div className="alert alert-danger mt-3 mb-0">
                      {importError}
                    </div>
                  )}
                </div>
                <div className="modal-footer">
                  <button
                    className="btn btn-outline-secondary"
                    type="button"
                    disabled={busy}
                    onClick={() => setImportSession(null)}
                  >
                    Anuluj
                  </button>
                  <button
                    className="btn btn-primary"
                    type="button"
                    disabled={busy}
                    onClick={() => void confirmImport()}
                  >
                    Zatwierdź import
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {manualPromptOpen && (
        <div
          className="modal-backdrop show confirm-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget)
              setManualPromptOpen(false);
          }}
        >
          <div className="modal d-block" role="dialog" aria-modal="true">
            <div className="modal-dialog modal-prompt-comparison">
              <div className="modal-content">
                <div className="modal-header">
                  <h2 className="h5 modal-title">
                    {manualPromptSource === "randomized"
                      ? "Propozycja skrócenia promptu"
                      : "Ręczna zmiana system promptu"}
                  </h2>
                </div>
                <div className="modal-body">
                  {manualPromptSource === "randomized" && (
                    <div className="alert alert-info">
                      Bielik wygenerował skróconą propozycję. Przejrzyj ją, a
                      następnie porównaj przed zapisem.
                    </div>
                  )}
                  <label className="form-label">Co którą encję zmienić?</label>
                  <input
                    className="form-control mb-3"
                    type="number"
                    min={2}
                    max={100}
                    value={manualEvery}
                    onChange={(event) =>
                      setManualEvery(
                        Math.min(
                          100,
                          Math.max(2, Number(event.target.value) || 2),
                        ),
                      )
                    }
                  />
                  <p className="text-secondary">
                    Zmieniony prompt zostanie użyty w około{" "}
                    {Math.floor(manualPromptIds.length / manualEvery)} z{" "}
                    {manualPromptIds.length} encji tej grupy.
                  </p>
                  <div className="row g-3 mb-3">
                    <div className="col-12 col-md-6">
                      <label className="form-label">Dostawca parafrazy</label>
                      <select
                        className="form-select"
                        value={paraphraseProvider}
                        disabled={
                          paraphrasingPrompt ||
                          paraphrasingSelection ||
                          !Object.keys(paraphraseProviders).length
                        }
                        onChange={(event) => {
                          const provider = event.target.value;
                          setParaphraseProvider(provider);
                          setParaphraseModel(
                            Object.keys(
                              paraphraseProviders[provider]?.models ?? {},
                            )[0] ?? "",
                          );
                        }}
                      >
                        {Object.entries(paraphraseProviders).map(
                          ([provider, config]) => (
                            <option key={provider} value={provider}>
                              {config.label}
                            </option>
                          ),
                        )}
                      </select>
                    </div>
                    {Object.keys(paraphraseModels).length > 0 && (
                      <div className="col-12 col-md-6">
                        <label className="form-label">Model</label>
                        <select
                          className="form-select"
                          value={paraphraseModel}
                          disabled={paraphrasingPrompt || paraphrasingSelection}
                          onChange={(event) =>
                            setParaphraseModel(event.target.value)
                          }
                        >
                          {Object.entries(paraphraseModels).map(
                            ([model, config]) => (
                              <option key={model} value={model}>
                                {config.label}
                              </option>
                            ),
                          )}
                        </select>
                      </div>
                    )}
                  </div>
                  <div className="row g-3">
                    <div className="col-12 col-lg-6">
                      <label className="form-label">Oryginalny prompt</label>
                      <textarea
                        className="form-control"
                        rows={16}
                        readOnly
                        value={manualOriginalPrompt}
                      />
                    </div>
                    <div className="col-12 col-lg-6">
                      <label className="form-label">
                        Skrócenie (edytowalne)
                      </label>
                      <textarea
                        ref={manualPromptRef}
                        className={`form-control ${promptFlash ? "prompt-flash" : ""}`}
                        rows={16}
                        value={manualPrompt}
                        onChange={(event) => {
                          setManualPrompt(event.target.value);
                          setManualComparison(null);
                          setPromptSelection(null);
                        }}
                        onSelect={(event) =>
                          setPromptSelection({
                            start: event.currentTarget.selectionStart,
                            end: event.currentTarget.selectionEnd,
                          })
                        }
                      />
                      {manualPrompt.includes("\0") && (
                        <div className="alert alert-warning mt-2 mb-0">
                          Kandydat zawiera niewidoczny znak NUL, którego baza
                          danych nie obsługuje.
                          <button
                            className="btn btn-sm btn-outline-danger ms-2"
                            type="button"
                            onClick={() => {
                              setManualPrompt((prompt) =>
                                prompt.replaceAll("\0", ""),
                              );
                              setManualComparison(null);
                              setPromptSelection(null);
                              setPromptModalError("");
                            }}
                          >
                            Usuń znaki NUL
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  {manualComparison && (
                    <div
                      className={`alert mt-3 mb-0 ${manualComparison.semantic_equivalent && manualComparison.instruction_plan_equivalent ? "alert-success" : "alert-danger"}`}
                    >
                      Zgodność semantyczna:{" "}
                      {manualComparison.semantic_equivalent ? "tak" : "nie"}.
                      Zgodność planu instrukcji:{" "}
                      {manualComparison.instruction_plan_equivalent
                        ? "tak"
                        : "nie"}
                      .
                      <br />
                      Uzasadnienie walidatora: {manualComparison.reason}
                    </div>
                  )}
                  {missingPromptTerms.length > 0 && (
                    <div className="alert alert-warning mt-3 mb-0">
                      W skróceniu brakuje terminów z oryginału:{" "}
                      <strong>{missingPromptTerms.join(", ")}</strong>. Sprawdź
                      je przed walidacją.
                    </div>
                  )}
                  {promptModalError && (
                    <div
                      className="alert alert-danger mt-3 mb-0"
                      style={{ whiteSpace: "pre-wrap" }}
                    >
                      {promptModalError}
                    </div>
                  )}
                  <nav
                    className="nav nav-tabs mt-3"
                    aria-label="Prompty systemowe"
                  >
                    <button
                      className={`nav-link ${promptTab === "paraphraser" ? "active" : ""}`}
                      type="button"
                      onClick={() => setPromptTab("paraphraser")}
                    >
                      Prompt parafrazera
                    </button>
                    <button
                      className={`nav-link ${promptTab === "validator" ? "active" : ""}`}
                      type="button"
                      onClick={() => setPromptTab("validator")}
                    >
                      Prompt walidatora
                    </button>
                  </nav>
                  <textarea
                    className="form-control validator-prompt rounded-top-0"
                    rows={5}
                    disabled={
                      promptTab === "paraphraser"
                        ? !paraphraserPrompt
                        : !validatorPrompt
                    }
                    placeholder="Wczytywanie..."
                    value={
                      promptTab === "paraphraser"
                        ? paraphraserPrompt
                        : validatorPrompt
                    }
                    onChange={(event) => {
                      if (promptTab === "paraphraser") {
                        setParaphraserPrompt(event.target.value);
                      } else {
                        setValidatorPrompt(event.target.value);
                        setManualComparison(null);
                      }
                    }}
                  />
                </div>
                <div className="modal-footer">
                  <button
                    className="btn btn-outline-secondary"
                    type="button"
                    disabled={busy}
                    onClick={() => setManualPromptOpen(false)}
                  >
                    Anuluj
                  </button>
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={
                      busy ||
                      comparingManualPrompt ||
                      paraphrasingPrompt ||
                      paraphrasingSelection ||
                      !manualPrompt.trim()
                    }
                    onClick={() => void compareManualPrompt()}
                  >
                    {comparingManualPrompt
                      ? "Walidowanie..."
                      : "Waliduj skrócenie"}
                  </button>
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={
                      busy ||
                      comparingManualPrompt ||
                      paraphrasingPrompt ||
                      paraphrasingSelection
                    }
                    onClick={() => void paraphraseManualPrompt()}
                  >
                    {paraphrasingPrompt ? (
                      <>
                        <span
                          className="spinner-border spinner-border-sm me-1"
                          aria-hidden="true"
                        />{" "}
                        Parafrazowanie...
                      </>
                    ) : (
                      <>
                        <Sparkles size={15} className="me-1" /> Parafrazuj cały
                        prompt
                      </>
                    )}
                  </button>
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={
                      busy ||
                      comparingManualPrompt ||
                      paraphrasingPrompt ||
                      paraphrasingSelection ||
                      !promptSelection ||
                      promptSelection.start === promptSelection.end
                    }
                    onClick={() => void paraphraseSelectedPrompt()}
                  >
                    {paraphrasingSelection ? (
                      <>
                        <span
                          className="spinner-border spinner-border-sm me-1"
                          aria-hidden="true"
                        />{" "}
                        Parafrazowanie...
                      </>
                    ) : (
                      <>
                        <Sparkles size={15} className="me-1" /> Parafrazuj
                        zaznaczone
                      </>
                    )}
                  </button>
                  <button
                    className="btn btn-primary"
                    type="button"
                    disabled={
                      busy ||
                      comparingManualPrompt ||
                      paraphrasingPrompt ||
                      paraphrasingSelection ||
                      !manualPrompt.trim()
                    }
                    onClick={() => void applyManualPromptVariant()}
                  >
                    Zatwierdź i zapisz prompt
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {selectedExample && (
        <>
          <button
            className="entity-drawer-backdrop"
            type="button"
            aria-label="Zamknij podgląd"
            onClick={closeDrawer}
          />
          <aside className="entity-drawer" aria-label="Podgląd encji">
            <div className="entity-drawer-header">
              <div>
                <h2 className="h5 mb-1">Szczegóły encji</h2>
                <small className="text-secondary">
                  {formatCreatedAt(selectedExample.created_at)}
                </small>
              </div>
              <div className="d-flex flex-wrap justify-content-end gap-2">
                <button
                  className="btn btn-outline-secondary"
                  type="button"
                  disabled={drawerEditing || filteredExamples.length < 2}
                  onClick={() => {
                    const index = filteredExamples.findIndex(
                      (example) => example.id === selectedExample.id,
                    );
                    openDrawer(
                      filteredExamples[(index + 1) % filteredExamples.length],
                    );
                  }}
                >
                  Następna
                </button>
                {drawerEditing ? (
                  <>
                    <button
                      className="btn btn-outline-primary"
                      type="button"
                      disabled={busy}
                      onClick={() => void saveDrawer(true)}
                    >
                      Zapisz jako kopię
                    </button>
                    <button
                      className="btn btn-primary"
                      type="button"
                      disabled={busy}
                      onClick={() => void saveDrawer()}
                    >
                      Zapisz
                    </button>
                  </>
                ) : (
                  <button
                    className="btn btn-primary"
                    type="button"
                    onClick={() =>
                      navigate(
                        `/builder/${selectedExample.corpus_id ?? corpusId}?edit=${selectedExample.id}`,
                      )
                    }
                  >
                    Edytuj
                  </button>
                )}
                <button
                  className="btn btn-outline-secondary"
                  type="button"
                  title="Zamknij"
                  aria-label="Zamknij"
                  onClick={closeDrawer}
                >
                  <X size={17} />
                </button>
              </div>
            </div>
            {drawerError && (
              <div className="alert alert-danger">{drawerError}</div>
            )}
            <section className="review-panel">
              <div className="d-flex flex-wrap align-items-center justify-content-between gap-2">
                <div>
                  <p className="panel-title mb-1">RECENZJA LOKALNEGO BIELIKA</p>
                  <small className="text-secondary">
                    Sugestia nie zmienia etykiety bez zatwierdzenia.
                  </small>
                </div>
                <button
                  className="btn btn-outline-primary"
                  type="button"
                  disabled={reviewing || drawerEditing}
                  onClick={() => void reviewSelectedExample()}
                >
                  <Sparkles size={16} className="me-1" />{" "}
                  {reviewing ? "Sprawdzanie..." : "Sprawdź"}
                </button>
              </div>
              <div className="d-flex flex-wrap gap-2 mt-3">
                <button
                  className="btn btn-sm btn-outline-success"
                  type="button"
                  disabled={busy || drawerEditing}
                  onClick={() => void applyReviewFlag("positive")}
                >
                  Oznacz ręcznie: positive
                </button>
                <button
                  className="btn btn-sm btn-outline-danger"
                  type="button"
                  disabled={busy || drawerEditing}
                  onClick={() => void applyReviewFlag("negative")}
                >
                  Oznacz ręcznie: negative
                </button>
              </div>
              {review && (
                <div className="review-result mt-3">
                  <span
                    className={`badge text-bg-${review.recommendation === "positive" ? "success" : review.recommendation === "negative" ? "danger" : "warning"}`}
                  >
                    {review.recommendation}
                  </span>
                  <span className="text-secondary small ms-2">
                    pewność: {review.confidence}
                  </span>
                  <p className="mb-2 mt-2">{review.reason}</p>
                  {review.recommendation !== "needs_review" && (
                    <button
                      className="btn btn-sm btn-primary"
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void applyReviewFlag(review.recommendation)
                      }
                    >
                      Przyjmij: {review.recommendation}
                    </button>
                  )}
                </div>
              )}
            </section>
            <div className="entity-thread">
              {drawerMessages.map((message, index) => (
                <article
                  className={`chat-message ${message.role} ${editingMessageIndex === index ? "editing" : ""}`}
                  key={`${message.role}-${index}`}
                  onClick={() => {
                    setDrawerEditing(true);
                    setEditingMessageIndex(index);
                  }}
                >
                  <button
                    className="chat-message-delete"
                    type="button"
                    title="Usuń wiadomość"
                    aria-label="Usuń wiadomość"
                    onClick={(event) => {
                      event.stopPropagation();
                      removeDrawerMessage(index);
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                  {editingMessageIndex === index ? (
                    <div className="entity-message-editor">
                      <div className="d-flex justify-content-between gap-2 mb-2">
                        <select
                          className="form-select form-select-sm role-select"
                          value={message.role}
                          onChange={(event) =>
                            updateDrawerMessage(index, {
                              role: event.target.value as MessageRole,
                            })
                          }
                        >
                          <option value="system">system</option>
                          <option value="user">user</option>
                          <option value="assistant">assistant</option>
                        </select>
                        {message.role !== "system" && (
                          <span className="d-flex gap-1">
                            <button
                              className="btn btn-sm message-move-button"
                              type="button"
                              title="Przenieś wyżej"
                              aria-label="Przenieś wyżej"
                              disabled={index === 0}
                              onClick={() => moveDrawerMessage(index, -1)}
                            >
                              <ChevronUp size={15} />
                            </button>
                            <button
                              className="btn btn-sm message-move-button"
                              type="button"
                              title="Przenieś niżej"
                              aria-label="Przenieś niżej"
                              disabled={index === drawerMessages.length - 1}
                              onClick={() => moveDrawerMessage(index, 1)}
                            >
                              <ChevronDown size={15} />
                            </button>
                          </span>
                        )}
                      </div>
                      <textarea
                        className="form-control"
                        spellCheck={false}
                        rows={Math.max(
                          4,
                          message.content.split(/\r?\n/).length + 1,
                        )}
                        value={message.content}
                        onFocus={(event) => {
                          event.currentTarget.style.height = "auto";
                          event.currentTarget.style.height = `${event.currentTarget.scrollHeight}px`;
                        }}
                        onInput={(event) => {
                          event.currentTarget.style.height = "auto";
                          event.currentTarget.style.height = `${event.currentTarget.scrollHeight}px`;
                        }}
                        onChange={(event) =>
                          updateDrawerMessage(index, {
                            content: event.target.value,
                          })
                        }
                      />
                    </div>
                  ) : (
                    <>
                      <strong>{message.role}</strong>
                      <p>{message.content}</p>
                    </>
                  )}
                </article>
              ))}
            </div>
            {drawerEditing && (
              <>
                <button
                  className="btn btn-outline-secondary mt-3"
                  type="button"
                  onClick={addDrawerMessage}
                >
                  <Plus size={17} className="me-1" /> Dodaj wiadomość
                </button>
                <div className="entity-drawer-footer">
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={busy}
                    onClick={() => void saveDrawer(true)}
                  >
                    Zapisz jako kopię
                  </button>
                  <button
                    className="btn btn-primary"
                    type="button"
                    disabled={busy}
                    onClick={() => void saveDrawer()}
                  >
                    Zapisz
                  </button>
                </div>
              </>
            )}
          </aside>
        </>
      )}
      {open && (
        <div className="modal-backdrop show confirm-backdrop">
          <div className="modal d-block" role="dialog">
            <div className="modal-dialog">
              <div className="modal-content">
                <div className="modal-header">
                  <h2 className="h5 modal-title">Nowy korpus</h2>
                </div>
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (name.trim()) setConfirm(true);
                  }}
                >
                  <div className="modal-body">
                    <label className="form-label" htmlFor="new-name">
                      Nazwa
                    </label>
                    <input
                      id="new-name"
                      className="form-control mb-3"
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      required
                    />
                    <label className="form-label" htmlFor="new-description">
                      Opis
                    </label>
                    <textarea
                      id="new-description"
                      className="form-control"
                      rows={3}
                      value={description}
                      onChange={(event) => setDescription(event.target.value)}
                    />
                  </div>
                  <div className="modal-footer">
                    <button
                      className="btn btn-outline-secondary"
                      type="button"
                      onClick={() => {
                        setOpen(false);
                        navigate("/corpora");
                      }}
                    >
                      Anuluj
                    </button>
                    <button className="btn btn-primary">Utwórz</button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
      {confirm && (
        <div className="modal-backdrop show confirm-backdrop">
          <div className="modal d-block" role="dialog">
            <div className="modal-dialog">
              <div className="modal-content">
                <div className="modal-header">
                  <h2 className="h5 modal-title">Potwierdź utworzenie</h2>
                </div>
                <div className="modal-body">
                  <strong>{name}</strong>
                </div>
                <div className="modal-footer">
                  <button
                    className="btn btn-outline-secondary"
                    type="button"
                    onClick={() => setConfirm(false)}
                  >
                    Wróć
                  </button>
                  <button
                    className="btn btn-primary"
                    type="button"
                    disabled={busy}
                    onClick={() => void submit()}
                  >
                    Potwierdź
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </MediumPageTemplate>
  );
}

function Builder({ corpora }: { corpora: Corpus[] }) {
  const { corpusId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const editingId = searchParams.get("edit");
  const selected = corpora.find((item) => item.id === corpusId);
  const [additionalMessages, setAdditionalMessages] = useState<Draft[]>(() => [
    draft(),
  ]);
  const [output, setOutput] = useState("");
  const [split, setSplit] = useState<Split>("train");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  useEffect(() => {
    api
      .models()
      .then(({ models }) => {
        setModels(models);
        setModel((current) => current || models[0] || "");
      })
      .catch(() => setModels([]));
  }, []);
  useEffect(() => {
    if (!editingId || !corpusId) return;
    api.examples(corpusId).then((examples) => {
      const example = examples.find((item) => item.id === editingId);
      if (!example) return;
      const completion = example.messages.at(-1);
      const input = example.messages.slice(0, -1);
      setAdditionalMessages(
        input.map((message) => ({
          id: crypto.randomUUID(),
          role: message.role as MessageRole,
          content: message.content,
        })),
      );
      setOutput(completion?.role === "assistant" ? completion.content : "");
      setSplit(example.split);
    });
  }, [corpusId, editingId]);
  const update = (id: string, change: Partial<Message>) =>
    setAdditionalMessages((current) =>
      current.map((item) => (item.id === id ? { ...item, ...change } : item)),
    );
  const messages = [
    ...additionalMessages
      .filter((item) => item.content.trim())
      .map(({ role, content }) => ({ role, content: content.trim() })),
    ...(output.trim()
      ? [{ role: "assistant" as const, content: output.trim() }]
      : []),
  ];
  async function generate() {
    const prompt = messages.filter((item) => item.role !== "assistant");
    if (!prompt.length || !model) return;
    setBusy(true);
    setOutput("");
    try {
      await api.chatStream(prompt, model, (content) =>
        setOutput((current) => current + content),
      );
    } finally {
      setBusy(false);
    }
  }
  async function save(flag: ExampleFlag) {
    if (
      !selected ||
      !additionalMessages.some((message) => message.content.trim()) ||
      !output.trim()
    )
      return;
    setBusy(true);
    try {
      const payload = {
        split,
        messages,
        flag,
      };
      if (editingId) {
        await api.updateExample(editingId, payload);
        navigate(`/corpora/${selected.id}`);
        return;
      }
      await api.createExample(selected.id, payload);
      setAdditionalMessages([draft()]);
      setOutput("");
      setNotice("Przykład zapisany.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <PageTemplate
      eyebrow={editingId ? "EDYCJA ENCJI" : "KONSTRUKCJA KORPUSU"}
      title={selected?.name ?? "Wybierz korpus po lewej"}
      actions={
        <div className="btn-group align-self-start">
          {(["train", "validation", "test"] as Split[]).map((item) => (
            <button
              className={`btn ${split === item ? "btn-primary" : "btn-outline-primary"}`}
              key={item}
              type="button"
              onClick={() => setSplit(item)}
            >
              {item}
            </button>
          ))}
        </div>
      }
    >
      {notice && <div className="alert alert-success mt-3">{notice}</div>}
      <form className="row g-4 mt-1">
        <section className="col-12 col-xl-6">
          <div className="card h-100 shadow-sm border-0">
            <div className="card-header panel-title">PRZYKŁAD INSTRUKCYJNY</div>
            <div className="card-body vstack gap-3">
              {additionalMessages.map((item) => (
                <div className="card border" key={item.id}>
                  <div className="card-header bg-light d-flex justify-content-between">
                    <select
                      className="form-select form-select-sm role-select"
                      value={item.role}
                      onChange={(event) =>
                        update(item.id, {
                          role: event.target.value as MessageRole,
                        })
                      }
                    >
                      <option value="system">system</option>
                      <option value="user">user</option>
                      <option value="assistant">assistant</option>
                    </select>
                    <button
                      className="btn btn-sm btn-outline-danger"
                      type="button"
                      onClick={() =>
                        setAdditionalMessages((current) =>
                          current.filter((message) => message.id !== item.id),
                        )
                      }
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                  <div className="card-body message-panel">
                    <textarea
                      className="form-control message-textarea"
                      rows={5}
                      value={item.content}
                      onChange={(event) =>
                        update(item.id, { content: event.target.value })
                      }
                    />
                  </div>
                </div>
              ))}
              <button
                className="btn btn-outline-secondary"
                type="button"
                onClick={() =>
                  setAdditionalMessages((current) => [...current, draft()])
                }
              >
                <Plus size={17} className="me-1" /> Dodaj wiadomość dodatkową
              </button>
            </div>
          </div>
        </section>
        <section className="col-12 col-xl-6">
          <div className="card h-100 shadow-sm border-0">
            <div className="card-header panel-title">WZORCOWA ODPOWIEDŹ</div>
            <div className="card-body d-flex flex-column">
              <label className="form-label" htmlFor="output">
                Odpowiedź asystenta
              </label>
              <textarea
                id="output"
                className="form-control flex-grow-1"
                rows={16}
                value={output}
                onChange={(event) => setOutput(event.target.value)}
              />
              <div className="row g-2 mt-2">
                <div className="col">
                  <select
                    className="form-select"
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    disabled={!models.length}
                  >
                    {models.map((item) => (
                      <option key={item}>{item}</option>
                    ))}
                  </select>
                </div>
                <div className="col-auto">
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={busy || !model}
                    onClick={() => void generate()}
                  >
                    <Sparkles size={17} className="me-1" /> Uzupełnij
                  </button>
                </div>
              </div>
              <span className="form-label mt-3">Klasa do balansowania</span>
              <div className="d-flex flex-wrap justify-content-between gap-2">
                {(
                  [
                    ["positive", "Zapisz pozytywny", "success"],
                    ["negative", "Zapisz negatywny", "danger"],
                  ] as const
                ).map(([value, label, variant]) => (
                  <button
                    className={`btn btn-${variant}`}
                    key={value}
                    type="button"
                    disabled={
                      busy ||
                      !selected ||
                      !additionalMessages.some((message) =>
                        message.content.trim(),
                      ) ||
                      !output.trim()
                    }
                    onClick={() => void save(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>
      </form>
    </PageTemplate>
  );
}

function Chat() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [history, setHistory] = useState<Message[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api.models().then(({ models }) => {
      setModels(models);
      setModel(models[0] || "");
    });
  }, []);
  async function send(event: FormEvent) {
    event.preventDefault();
    if (!text.trim() || !model) return;
    const next = [...history, { role: "user" as const, content: text.trim() }];
    setHistory([...next, { role: "assistant", content: "" }]);
    setText("");
    setBusy(true);
    try {
      await api.chatStream(next, model, (content) =>
        setHistory((current) =>
          current.map((message, index) =>
            index === current.length - 1
              ? { ...message, content: message.content + content }
              : message,
          ),
        ),
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="workspace-page chat-page">
      <p className="section-kicker">LOKALNA INFERENCJA</p>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h1>Lokalny Bielik</h1>
        <select
          className="form-select model-select"
          value={model}
          onChange={(event) => setModel(event.target.value)}
        >
          {models.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
      </div>
      <div className="card shadow-sm border-0">
        <div className="card-body chat-history">
          {history.map((message, index) => (
            <article className={`chat-message ${message.role}`} key={index}>
              <strong>{message.role}</strong>
              <p>{message.content}</p>
            </article>
          ))}
        </div>
        <form className="card-footer" onSubmit={send}>
          <textarea
            className="form-control"
            rows={4}
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
          <button className="btn btn-primary mt-2" disabled={busy || !model}>
            <Send size={17} className="me-1" /> Wyślij
          </button>
        </form>
      </div>
    </section>
  );
}

type ChartSeries = {
  label: string;
  color: string;
  points: Array<{ x: number; y: number }>;
};

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours ? `${hours} h ${minutes} min` : `${minutes} min ${total % 60} s`;
}

function formatMetric(value: number | undefined | null, digits = 4) {
  if (value === undefined || value === null) return "-";
  return Math.abs(value) < 0.001 && value !== 0
    ? value.toExponential(2)
    : value.toFixed(digits);
}

function LineChart({
  title,
  series,
  format,
}: {
  title: string;
  series: ChartSeries[];
  format: (value: number) => string;
}) {
  const width = 520;
  const height = 200;
  const padding = { top: 12, right: 12, bottom: 24, left: 64 };
  const points = series.flatMap((item) => item.points);
  if (points.length < 2) {
    return (
      <div className="training-chart">
        <h3 className="h6">{title}</h3>
        <p className="text-secondary small mb-0">
          Za mało punktów — wykres pojawi się po kilku krokach uczenia.
        </p>
      </div>
    );
  }
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const scaleX = (x: number) =>
    padding.left +
    ((x - minX) / (maxX - minX || 1)) * (width - padding.left - padding.right);
  const scaleY = (y: number) =>
    height -
    padding.bottom -
    ((y - minY) / (maxY - minY || 1)) * (height - padding.top - padding.bottom);
  return (
    <div className="training-chart">
      <div className="d-flex justify-content-between align-items-baseline">
        <h3 className="h6 mb-1">{title}</h3>
        <div className="d-flex gap-3 small">
          {series
            .filter((item) => item.points.length)
            .map((item) => (
              <span key={item.label}>
                <span
                  className="training-chart-swatch"
                  style={{ background: item.color }}
                />
                {item.label}
              </span>
            ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-100" role="img">
        {[minY, (minY + maxY) / 2, maxY].map((value) => (
          <g key={value}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={scaleY(value)}
              y2={scaleY(value)}
              stroke="#e3e8e6"
            />
            <text
              x={padding.left - 6}
              y={scaleY(value) + 4}
              textAnchor="end"
              fontSize="11"
              fill="#6c757d"
            >
              {format(value)}
            </text>
          </g>
        ))}
        <text x={padding.left} y={height - 6} fontSize="11" fill="#6c757d">
          krok {minX}
        </text>
        <text
          x={width - padding.right}
          y={height - 6}
          textAnchor="end"
          fontSize="11"
          fill="#6c757d"
        >
          krok {maxX}
        </text>
        {series.map((item) => (
          <g key={item.label}>
            <polyline
              fill="none"
              stroke={item.color}
              strokeWidth="2"
              points={item.points
                .map((point) => `${scaleX(point.x)},${scaleY(point.y)}`)
                .join(" ")}
            />
            {item.points.length < 30 &&
              item.points.map((point) => (
                <circle
                  key={point.x}
                  cx={scaleX(point.x)}
                  cy={scaleY(point.y)}
                  r="3"
                  fill={item.color}
                />
              ))}
          </g>
        ))}
      </svg>
    </div>
  );
}

function metricSeries(
  metrics: TrainingMetric[],
  key: "loss" | "eval_loss" | "learning_rate",
) {
  return metrics
    .filter((metric) => typeof metric[key] === "number")
    .map((metric) => ({ x: metric.step, y: metric[key] as number }));
}

function TrainingDashboard({ status }: { status: TrainingStatus | null }) {
  const metrics = status?.metrics ?? [];
  const hyperparameters = status?.hyperparameters;
  const last = metrics.at(-1);
  const lossPoints = metricSeries(metrics, "loss");
  const evalPoints = metricSeries(metrics, "eval_loss");
  const learningRatePoints = metricSeries(metrics, "learning_rate");
  const step = last?.step ?? 0;
  const maxSteps = last?.max_steps ?? 0;
  const progress = maxSteps ? Math.min(100, (step / maxSteps) * 100) : 0;
  const first = metrics.find((metric) => metric.step > 0) ?? metrics[0];
  const secondsPerStep =
    first && last && last.step > first.step
      ? (last.time - first.time) / (last.step - first.step)
      : NaN;
  const startedAt = status?.started_at ? Date.parse(status.started_at) : NaN;
  const elapsed =
    status?.state === "running"
      ? (Date.now() - startedAt) / 1000
      : last && metrics[0]
        ? last.time - metrics[0].time
        : NaN;
  const epoch = [...metrics].reverse().find((metric) => metric.epoch)?.epoch;
  const totalEpochs =
    metrics.find((metric) => metric.num_train_epochs)?.num_train_epochs ??
    hyperparameters?.epochs;
  const currentLearningRate = learningRatePoints.at(-1)?.y;
  const lastLoss = lossPoints.at(-1)?.y;
  const minLoss = lossPoints.length
    ? Math.min(...lossPoints.map((point) => point.y))
    : undefined;
  const lastEval = evalPoints.at(-1)?.y;
  const gradNorm = [...metrics]
    .reverse()
    .find((metric) => typeof metric.grad_norm === "number")?.grad_norm;
  const effectiveBatch =
    hyperparameters?.batch_size && hyperparameters.gradient_accumulation_steps
      ? hyperparameters.batch_size * hyperparameters.gradient_accumulation_steps
      : null;
  return (
    <>
      <div className="row g-3 mt-2">
        <div className="col-12 col-lg-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <p className="text-secondary mb-1">Postęp</p>
              <div className="display-6">
                {maxSteps ? `${progress.toFixed(1)}%` : "-"}
              </div>
              <div className="progress my-2" style={{ height: 8 }}>
                <div
                  className={`progress-bar ${status?.state === "running" ? "progress-bar-striped progress-bar-animated" : ""}`}
                  style={{ width: `${progress}%` }}
                />
              </div>
              <dl className="training-stats mb-0">
                <dt>Krok</dt>
                <dd>{maxSteps ? `${step} / ${maxSteps}` : "-"}</dd>
                <dt>Epoka</dt>
                <dd>
                  {epoch !== undefined
                    ? `${epoch.toFixed(2)} / ${totalEpochs ?? "-"}`
                    : "-"}
                </dd>
                <dt>Czas</dt>
                <dd>{formatDuration(elapsed)}</dd>
                <dt>Pozostało (ETA)</dt>
                <dd>
                  {status?.state === "running" && maxSteps
                    ? formatDuration(secondsPerStep * (maxSteps - step))
                    : "-"}
                </dd>
                <dt>Na krok</dt>
                <dd>
                  {Number.isFinite(secondsPerStep)
                    ? `${secondsPerStep.toFixed(1)} s`
                    : "-"}
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="col-12 col-lg-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <p className="text-secondary mb-1">Learning rate (aktualny)</p>
              <div className="display-6">
                {formatMetric(
                  currentLearningRate ?? hyperparameters?.learning_rate,
                )}
              </div>
              <dl className="training-stats mt-2 mb-0">
                <dt>LR bazowy</dt>
                <dd>{formatMetric(hyperparameters?.learning_rate)}</dd>
                <dt>Batch efektywny</dt>
                <dd>
                  {effectiveBatch
                    ? `${effectiveBatch} (${hyperparameters?.batch_size} × ${hyperparameters?.gradient_accumulation_steps})`
                    : "-"}
                </dd>
                <dt>Epoki</dt>
                <dd>{hyperparameters?.epochs ?? "-"}</dd>
                <dt>LoRA r / alpha</dt>
                <dd>
                  {hyperparameters
                    ? `${hyperparameters.lora_rank} / ${hyperparameters.lora_alpha} (dropout ${hyperparameters.lora_dropout})`
                    : "-"}
                </dd>
                <dt>Max długość</dt>
                <dd>
                  {hyperparameters?.max_length ?? "-"}{" "}
                  {hyperparameters?.quantization
                    ? `· ${hyperparameters.quantization}`
                    : ""}
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="col-12 col-lg-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <p className="text-secondary mb-1">Loss (train)</p>
              <div className="display-6">{formatMetric(lastLoss)}</div>
              <dl className="training-stats mt-2 mb-0">
                <dt>Minimalny loss</dt>
                <dd>{formatMetric(minLoss)}</dd>
                <dt>Eval loss</dt>
                <dd>{formatMetric(lastEval)}</dd>
                <dt>Grad norm</dt>
                <dd>{formatMetric(gradNorm)}</dd>
                <dt>Final train loss</dt>
                <dd>{formatMetric(last?.train_loss)}</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>
      <div className="card shadow-sm border-0 mt-4">
        <div className="card-body">
          <div className="row g-4">
            <div className="col-12 col-xl-6">
              <LineChart
                title="Loss"
                format={(value) => value.toFixed(3)}
                series={[
                  { label: "train", color: "#176b61", points: lossPoints },
                  { label: "eval", color: "#d9822b", points: evalPoints },
                ]}
              />
            </div>
            <div className="col-12 col-xl-6">
              <LineChart
                title="Learning rate"
                format={(value) => value.toExponential(1)}
                series={[
                  {
                    label: "lr",
                    color: "#3d6fb6",
                    points: learningRatePoints,
                  },
                ]}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Training() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [baseModels, setBaseModels] = useState<string[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [adapterName, setAdapterName] = useState("bielik-qlora-v1");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = () =>
    api
      .trainingStatus()
      .then(setStatus)
      .catch((requestError: Error) => setError(requestError.message));

  useEffect(() => {
    void refresh();
    void api.corpora().then((items) => {
      setCorpora(items);
      setCorpusId((current) => current || items[0]?.id || "");
    });
    void api.trainingModels().then(({ models }) => {
      setBaseModels(models);
      setBaseModel((current) => current || models[0] || "");
    });
    const interval = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(interval);
  }, []);

  const start = async () => {
    setBusy(true);
    setError("");
    try {
      setStatus(await api.startTraining({ corpusId, baseModel, adapterName }));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Nie udało się uruchomić treningu.",
      );
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setError("");
    try {
      setStatus(await api.stopTraining());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Nie udało się zatrzymać treningu.",
      );
    } finally {
      setBusy(false);
    }
  };

  const splits = status?.splits;
  const isRunning = status?.state === "running";
  return (
    <section className="workspace-page medium-page">
      <p className="section-kicker">QLORA</p>
      <h1>Uczenie i status</h1>
      <div className="row g-3 mt-2">
        {(["train", "validation", "test"] as Split[]).map((split) => (
          <div className="col-12 col-md-4" key={split}>
            <div className="card shadow-sm border-0">
              <div className="card-body">
                <p className="text-secondary mb-1">{split}</p>
                <div className="display-6">{splits?.[split] ?? "-"}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="card shadow-sm border-0 mt-4">
        <div className="card-body">
          <div className="row g-3">
            <div className="col-12 col-md-6">
              <label className="form-label" htmlFor="training-corpus">
                Korpus
              </label>
              <select
                id="training-corpus"
                className="form-select"
                value={corpusId}
                onChange={(event) => setCorpusId(event.target.value)}
                disabled={isRunning}
              >
                <option value="">Wybierz korpus</option>
                {corpora.map((corpus) => (
                  <option key={corpus.id} value={corpus.id}>
                    {corpus.name} ({corpus.example_count})
                  </option>
                ))}
              </select>
            </div>
            <div className="col-12 col-md-6">
              <label className="form-label" htmlFor="training-model">
                Model bazowy
              </label>
              <select
                id="training-model"
                className="form-select"
                value={baseModel}
                onChange={(event) => setBaseModel(event.target.value)}
                disabled={isRunning}
              >
                {baseModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
            <div className="col-12">
              <label className="form-label" htmlFor="adapter-name">
                Nazwa adaptera
              </label>
              <input
                id="adapter-name"
                className="form-control"
                value={adapterName}
                onChange={(event) =>
                  setAdapterName(
                    event.target.value
                      .toLowerCase()
                      .replace(/[^a-z0-9-]/g, "-"),
                  )
                }
                disabled={isRunning}
              />
            </div>
          </div>
          <span
            className={`badge ${isRunning ? "text-bg-success" : "text-bg-secondary"}`}
          >
            {status?.state ?? "loading"}
          </span>
          <div className="d-flex flex-wrap gap-2 mt-3">
            <button
              className="btn btn-primary"
              onClick={() => void start()}
              disabled={
                busy ||
                isRunning ||
                !corpusId ||
                !baseModel ||
                adapterName.length < 2
              }
            >
              <Play size={16} className="me-1" /> Eksportuj i rozpocznij
            </button>
            <button
              className="btn btn-outline-danger"
              onClick={() => void stop()}
              disabled={busy || !isRunning}
            >
              <Square size={16} className="me-1" /> Zatrzymaj
            </button>
            <button
              className="btn btn-outline-secondary"
              onClick={() => void refresh()}
              disabled={busy}
            >
              <RefreshCw size={16} className="me-1" /> Odśwież
            </button>
          </div>
          {error && <p className="text-danger small mt-3 mb-0">{error}</p>}
          <h2 className="h5 mt-4">Profil QLoRA</h2>
          <code>{status?.profile ?? "profil QLoRA"}</code>
          <p className="text-secondary small mt-3 mb-0">
            Uczony jest nowy adapter LoRA, nie model bazowy. Eksport obejmuje
            tylko wybrany korpus i jego splity <code>train</code> oraz{" "}
            <code>validation</code>.
          </p>
          {status?.job?.adapter_name && (
            <p className="text-secondary small mt-2 mb-0">
              Ostatnie zadanie: <code>{status.job.base_model}</code> do adaptera{" "}
              <code>{status.job.adapter_name}</code>.
            </p>
          )}
        </div>
      </div>
      <TrainingDashboard status={status} />
      <div className="card shadow-sm border-0 mt-4">
        <div className="card-body">
          <h2 className="h5">Logi zadania</h2>
          <pre className="training-logs mb-0">
            {status?.logs || "Brak logów zadania."}
          </pre>
        </div>
      </div>
    </section>
  );
}

function Workspace() {
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const refresh = () => api.corpora().then(setCorpora);
  useEffect(() => {
    void refresh();
  }, []);
  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <Bot size={21} /> Bielik LoRA Lab{" "}
        <nav className="workspace-header-nav">
          <NavLink to="/chat">Lokalny Bielik</NavLink>
          <NavLink to="/training">Uczenie</NavLink>
        </nav>
        <button
          className="btn btn-sm btn-outline-light"
          onClick={() => void refresh()}
        >
          <RefreshCw size={15} />
        </button>
      </header>
      <aside className="corpus-sidebar">
        <div className="sidebar-title">
          <Database size={16} /> KORPUSY
        </div>
        <NavLink className="btn btn-primary w-100 mb-3" to="/corpora/new">
          <FilePlus2 size={17} className="me-1" /> Nowy korpus
        </NavLink>
        <nav>
          {corpora.map((corpus) => (
            <NavLink
              className="corpus-link"
              key={corpus.id}
              to={`/corpora/${corpus.id}`}
            >
              <span>{corpus.name}</span>
              <small>{corpus.example_count}</small>
            </NavLink>
          ))}
        </nav>
      </aside>
      <section className="content-area">
        <Routes>
          <Route
            path="/corpora"
            element={
              <CorporaPage
                corpora={corpora}
                onCreated={(corpus) =>
                  setCorpora((current) => [corpus, ...current])
                }
                onCorpusUpdated={() => void refresh()}
              />
            }
          />
          <Route
            path="/corpora/:corpusId"
            element={
              <CorporaPage
                corpora={corpora}
                onCreated={(corpus) =>
                  setCorpora((current) => [corpus, ...current])
                }
                onCorpusUpdated={() => void refresh()}
              />
            }
          />
          <Route
            path="/corpora/new"
            element={
              <CorporaPage
                corpora={corpora}
                onCreated={(corpus) =>
                  setCorpora((current) => [corpus, ...current])
                }
                onCorpusUpdated={() => void refresh()}
                openCreate
              />
            }
          />
          <Route
            path="/builder/:corpusId?"
            element={<Builder corpora={corpora} />}
          />
          <Route path="/chat" element={<Chat />} />
          <Route path="/training" element={<Training />} />
          <Route path="*" element={<Navigate to="/corpora" replace />} />
        </Routes>
      </section>
    </main>
  );
}

export function App() {
  return (
    <HashRouter>
      <Workspace />
    </HashRouter>
  );
}
