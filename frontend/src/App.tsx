import {
  ChangeEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Bot,
  ChevronDown,
  ChevronUp,
  Check,
  Database,
  Download,
  File as FileIcon,
  FileCode,
  FileImage,
  FilePlus2,
  FileSpreadsheet,
  FileText,
  FileUp,
  Folder,
  FolderOpen,
  Copy,
  GraduationCap,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Paperclip,
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
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  api,
  Corpus,
  Example,
  CorpusAnalysis,
  CorpusAnalysisPart,
  CorpusSettings,
  AnalysisTypeRow,
  EntityTypeDefinition,
  ExampleIssue,
  ModelChoice,
  PreferenceBatchRequest,
  PreferenceBatchStatus,
  SplitRatio,
  ExampleFlag,
  BulkTransformResult,
  EvaluationSummary,
  EvaluationCheckpointResult,
  EvaluationCurvePoint,
  EvaluationStatus,
  ServingStatus,
  ExportsStatus,
  ExportQuantization,
  AgentModel,
  AgentSession,
  AgentAttachment,
  ImportedExample,
  Message,
  MessageRole,
  ParaphraseProviderCatalog,
  TrainingMetric,
  DatasetStats,
  LoraLayerSnapshot,
  TrainingStatus,
  TrainingRunSummary,
} from "./api";
import { PageTemplate } from "./components/PageTemplate";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MediumPageTemplate } from "./components/MediumPageTemplate";

type Split = "train" | "validation" | "test";
// "unassigned" = parked outside training ("bez splitu"); not a split you train or evaluate on.
type ExampleSplit = Split | "unassigned";
const splitLabel = (split: string) =>
  split === "unassigned" ? "bez splitu" : split;
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
  if (!trimmed) throw new Error("Plik nie zawiera przykładów JSONL.");
  if (trimmed.startsWith("[")) {
    const records = JSON.parse(trimmed) as unknown;
    if (!Array.isArray(records))
      throw new Error("Plik JSON musi zawierać tablicę przykładów.");
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

function corpusDtoProblem(records: Record<string, unknown>[]): string | null {
  for (const [index, record] of records.entries()) {
    const line = `Rekord ${index + 1}`;
    const messages = record.messages;
    if (!Array.isArray(messages)) return `${line}: brak tablicy messages.`;
    if (messages.length < 2)
      return `${line}: messages ma mniej niż 2 wiadomości.`;
    for (const [position, message] of messages.entries()) {
      const item = (message ?? {}) as Record<string, unknown>;
      if (!["system", "user", "assistant"].includes(item.role as string))
        return `${line}, wiadomość ${position + 1}: nieobsługiwana rola „${String(item.role)}”.`;
      if (typeof item.content !== "string" || !item.content)
        return `${line}, wiadomość ${position + 1} (${String(item.role)}): pusta lub nie-tekstowa treść.`;
    }
    if ((messages.at(-1) as Record<string, unknown>).role !== "assistant")
      return `${line}: ostatnia wiadomość nie jest od assistant.`;
  }
  return null;
}

function isCorpusDto(records: Record<string, unknown>[]): boolean {
  return corpusDtoProblem(records) === null;
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

function autoMapRecords(
  records: Record<string, unknown>[],
): { examples: ImportedExample[]; skipped: string[] } | null {
  const owu = isOwuAnnotationDto(records);
  const keys = [...new Set(records.flatMap((record) => Object.keys(record)))];
  const detected = detectImportMapping(keys);
  if (!owu && !detected.messages && !(detected.user && detected.assistant))
    return null;
  // split/flag are applied separately so a bad value doesn't drop the record
  const mapping = { ...detected, split: "", flag: "" };
  const examples: ImportedExample[] = [];
  const skipped: string[] = [];
  records.forEach((record, index) => {
    try {
      const mapped = owu
        ? mapOwuAnnotationRecord(record, index)
        : mapImportRecord(record, mapping, index);
      const messages = mapped.messages
        .filter(
          (message) =>
            message &&
            ["system", "user", "assistant"].includes(message.role) &&
            typeof message.content === "string" &&
            message.content.trim(),
        )
        .map((message) => ({ role: message.role, content: message.content }));
      if (messages.length < 2 || messages.at(-1)?.role !== "assistant") {
        skipped.push(`rekord ${index + 1}: brak pary pytanie–odpowiedź`);
        return;
      }
      const split = detected.split ? record[detected.split] : undefined;
      const flag = detected.flag ? record[detected.flag] : undefined;
      examples.push({
        ...mapped,
        messages,
        ...(typeof split === "string" &&
        ["train", "validation", "test"].includes(split)
          ? { split: split as Split }
          : {}),
        ...(typeof flag === "string" &&
        ["positive", "negative", "unclassified"].includes(flag)
          ? { flag: flag as ExampleFlag }
          : {}),
      });
    } catch (error) {
      skipped.push(
        `rekord ${index + 1}: ${error instanceof Error ? error.message : "błąd"}`,
      );
    }
  });
  return examples.length ? { examples, skipped } : null;
}

type AgentRequest = { id: number; text: string };

type AgentLogItem = {
  kind: "user" | "assistant" | "tool" | "error";
  text: string;
};

const SESSION_AREAS: Array<[string, string]> = [
  ["uploads", "Załączniki (uploads)"],
  ["work", "Robocze (work)"],
  ["exports", "Wyniki (exports)"],
  ["converted", "Po konwersji (converted)"],
  ["notes", "Notatki (notes)"],
  ["scripts", "Skrypty (scripts)"],
];

const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "webp", "bmp"];
// Office documents are converted to PDF by LibreOffice in the sandbox.
const PDF_EXTENSIONS = [
  "pdf",
  "doc",
  "docx",
  "odt",
  "rtf",
  "ppt",
  "pptx",
  "odp",
];
const SHEET_EXTENSIONS = ["xlsx", "xlsm", "xls", "ods", "csv", "tsv"];
const CODE_EXTENSIONS = [
  "py",
  "sh",
  "json",
  "jsonl",
  "yaml",
  "yml",
  "js",
  "ts",
];

const extensionOf = (path: string) =>
  path.includes(".") ? path.split(".").pop()!.toLowerCase() : "";

function SessionFileIcon({ path }: { path: string }) {
  const extension = extensionOf(path);
  if (IMAGE_EXTENSIONS.includes(extension)) return <FileImage size={16} />;
  if (SHEET_EXTENSIONS.includes(extension))
    return <FileSpreadsheet size={16} />;
  if (CODE_EXTENSIONS.includes(extension)) return <FileCode size={16} />;
  if ([...PDF_EXTENSIONS, "md", "txt"].includes(extension))
    return <FileText size={16} />;
  return <FileIcon size={16} />;
}

type SessionFile = { path: string; bytes: number };
type FileTree = { folders: Map<string, FileTree>; files: SessionFile[] };

function buildTree(files: SessionFile[], prefix: string): FileTree {
  const root: FileTree = { folders: new Map(), files: [] };
  for (const file of files) {
    const parts = file.path.slice(prefix.length).split("/");
    let node = root;
    for (const folder of parts.slice(0, -1)) {
      if (!node.folders.has(folder))
        node.folders.set(folder, { folders: new Map(), files: [] });
      node = node.folders.get(folder)!;
    }
    node.files.push(file);
  }
  return root;
}

function FilePreview({ sessionId, path }: { sessionId: string; path: string }) {
  const url = `/api/agent/sessions/${sessionId}/preview?path=${encodeURIComponent(path)}`;
  const extension = extensionOf(path);
  const binary =
    IMAGE_EXTENSIONS.includes(extension) || PDF_EXTENSIONS.includes(extension);
  const [result, setResult] = useState<{
    kind: string;
    content: string;
    truncated?: boolean;
  } | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setResult(null);
    setError("");
    if (binary) return;
    let current = true;
    void fetch(url)
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok)
          throw new Error(body.detail ?? "Podgląd niedostępny.");
        if (current) setResult(body);
      })
      .catch((fetchError: Error) => current && setError(fetchError.message));
    return () => {
      current = false;
    };
  }, [url, binary]);
  if (IMAGE_EXTENSIONS.includes(extension))
    return (
      <div className="file-preview-image">
        <img src={url} alt={path} />
      </div>
    );
  if (PDF_EXTENSIONS.includes(extension))
    return <iframe className="file-preview-frame" src={url} title={path} />;
  if (error) return <div className="alert alert-warning m-3">{error}</div>;
  if (!result)
    return <div className="text-secondary p-3">Wczytywanie podglądu…</div>;
  return (
    <div className="file-preview-text">
      {result.truncated && (
        <div className="alert alert-light border small py-1">
          Podgląd skrócony — pełny plik pobierzesz przyciskiem obok nazwy.
        </div>
      )}
      {result.kind === "markdown" ? (
        <div className="markdown-preview">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {result.content}
          </ReactMarkdown>
        </div>
      ) : result.kind === "text" ? (
        <pre>{result.content}</pre>
      ) : (
        <div className="text-secondary">{result.content}</div>
      )}
    </div>
  );
}

function SessionFilesDrawer({
  sessionId,
  files,
  attachmentsByPath,
  onRefresh,
  onClose,
}: {
  sessionId: string;
  files: SessionFile[];
  attachmentsByPath: Map<string, AgentAttachment>;
  onRefresh: () => void;
  onClose: () => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (selected) setSelected(null);
      else onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selected, onClose]);
  const toggle = (key: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const renderTree = (
    tree: FileTree,
    key: string,
    depth: number,
  ): ReactNode => (
    <>
      {[...tree.folders.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([name, child]) => {
          const childKey = `${key}/${name}`;
          const open = !collapsed.has(childKey);
          return (
            <div key={childKey}>
              <button
                className="files-drawer-row folder"
                type="button"
                style={{ paddingLeft: `${0.5 + depth}rem` }}
                onClick={() => toggle(childKey)}
              >
                {open ? <FolderOpen size={16} /> : <Folder size={16} />}
                <span>{name}</span>
              </button>
              {open && renderTree(child, childKey, depth + 1)}
            </div>
          );
        })}
      {[...tree.files]
        .sort((a, b) => a.path.localeCompare(b.path))
        .map((file) => {
          const attachment = attachmentsByPath.get(file.path);
          return (
            <button
              key={file.path}
              className={`files-drawer-row file ${selected === file.path ? "active" : ""}`}
              type="button"
              style={{ paddingLeft: `${0.5 + depth}rem` }}
              title={file.path}
              onClick={() => setSelected(file.path)}
            >
              <SessionFileIcon path={file.path} />
              <span className="text-truncate">
                {file.path.split("/").pop()}
              </span>
              {attachment && (
                <span className="badge text-bg-light border text-dark">
                  {attachment.id}
                </span>
              )}
              <small className="text-secondary ms-auto">
                {formatBytes(file.bytes)}
              </small>
            </button>
          );
        })}
    </>
  );
  return (
    <aside
      className={`files-drawer ${selected ? "expanded" : ""}`}
      aria-label="Pliki rozmowy"
    >
      <div className="files-drawer-list">
        <div className="files-drawer-header">
          <strong>Pliki rozmowy · {sessionId.slice(0, 8)}</strong>
          <span className="d-flex gap-1">
            <button
              className="btn btn-sm btn-outline-secondary"
              type="button"
              title="Odśwież"
              onClick={onRefresh}
            >
              <RefreshCw size={14} />
            </button>
            <button
              className="btn btn-sm btn-outline-secondary"
              type="button"
              title="Zamknij"
              onClick={onClose}
            >
              <X size={14} />
            </button>
          </span>
        </div>
        <div className="files-drawer-tree">
          {SESSION_AREAS.map(([area, label]) => {
            const areaFiles = files.filter((file) =>
              file.path.startsWith(`${area}/`),
            );
            const open = !collapsed.has(area) && areaFiles.length > 0;
            return (
              <div key={area}>
                <button
                  className="files-drawer-row folder area"
                  type="button"
                  disabled={!areaFiles.length}
                  onClick={() => toggle(area)}
                >
                  {open ? <FolderOpen size={16} /> : <Folder size={16} />}
                  <span>{label}</span>
                  <small className="text-secondary ms-auto">
                    {areaFiles.length}
                  </small>
                </button>
                {open && renderTree(buildTree(areaFiles, `${area}/`), area, 1)}
              </div>
            );
          })}
        </div>
      </div>
      {selected && (
        <div className="files-drawer-preview">
          <div className="files-drawer-header">
            <span className="d-flex align-items-center gap-2 text-truncate">
              <SessionFileIcon path={selected} />
              <strong className="text-truncate">{selected}</strong>
            </span>
            <span className="d-flex gap-1">
              <a
                className="btn btn-sm btn-outline-secondary"
                href={`/api/agent/sessions/${sessionId}/files?path=${encodeURIComponent(selected)}`}
                download={selected.split("/").pop()}
                title="Pobierz"
              >
                <Download size={14} />
              </a>
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                title="Zamknij podgląd"
                onClick={() => setSelected(null)}
              >
                <X size={14} />
              </button>
            </span>
          </div>
          <FilePreview sessionId={sessionId} path={selected} />
        </div>
      )}
    </aside>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024)
    return `${(bytes / 1024).toFixed(1).replace(".", ",")} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1).replace(".", ",")} MB`;
}

function prettyAnswer(answer: string) {
  try {
    return JSON.stringify(JSON.parse(answer), null, 2);
  } catch {
    return answer;
  }
}

function toolLabel(name: string, args: Record<string, unknown>) {
  const details = Object.entries(args)
    .filter(([key]) => key !== "examples")
    .map(([key, value]) => {
      const text = JSON.stringify(value);
      return `${key}=${text.length > 160 ? `${text.slice(0, 160)}…` : text}`;
    })
    .join(", ");
  const count = Array.isArray(args.examples)
    ? ` (${args.examples.length})`
    : "";
  return `${name}${count}${details ? ` · ${details}` : ""}`;
}

function EntityAgentPanel({
  corpus,
  onAdded,
  request,
}: {
  corpus: Corpus | undefined;
  onAdded: () => Promise<void>;
  request?: AgentRequest | null;
}) {
  const storageKey = corpus ? `entity-agent:${corpus.id}` : "";
  const [models, setModels] = useState<AgentModel[]>([]);
  const [model, setModel] = useState(
    () => localStorage.getItem("entity-agent-model") ?? "",
  );
  const [log, setLog] = useState<AgentLogItem[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [session, setSession] = useState<AgentSession | null>(null);
  const [filesOpen, setFilesOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(0);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const refreshFiles = (sessionId: string) =>
    api
      .agentSession(sessionId)
      .then(setSession)
      .catch(() => setSession(null));
  useEffect(() => {
    void api.agentModels().then((result) => {
      setModels(result.models);
      setModel((current) =>
        result.models.some((item) => item.id === current && item.available)
          ? current
          : (result.default ?? ""),
      );
    });
  }, []);
  const corpusModel = corpus?.settings?.default_model;
  useEffect(() => {
    if (
      corpusModel &&
      models.some((item) => item.id === corpusModel && item.available)
    )
      setModel(corpusModel);
  }, [corpus?.id, corpusModel, models]);
  useEffect(() => {
    if (!storageKey) return;
    const saved = JSON.parse(localStorage.getItem(storageKey) ?? "{}");
    const sessionId: string = saved.conversationId ?? crypto.randomUUID();
    setLog(saved.log ?? []);
    setConversationId(sessionId);
    setSession(null);
    void refreshFiles(sessionId);
  }, [storageKey]);
  useEffect(() => {
    if (storageKey && conversationId)
      localStorage.setItem(storageKey, JSON.stringify({ log, conversationId }));
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [storageKey, log, conversationId]);
  useEffect(() => {
    if (!request) return;
    // While the agent is busy the request waits in the composer instead of being lost.
    if (busy || !model) setInput(request.text);
    else void send(request.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request?.id]);
  const send = async (override?: string) => {
    const text = (override ?? input).trim();
    if (!corpus || !model || !text || busy) return;
    const history = [...log, { kind: "user" as const, text }];
    setLog(history);
    if (override === undefined) setInput("");
    setBusy(true);
    setError("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await api.agentChat(
        corpus.id,
        model,
        history
          .filter((item) => item.kind === "user" || item.kind === "assistant")
          .map((item) => ({
            role: item.kind as "user" | "assistant",
            content: item.text,
          })),
        (event) => {
          if (event.type === "text")
            setLog((current) => [
              ...current,
              { kind: "assistant", text: event.content },
            ]);
          else if (event.type === "tool_call")
            setLog((current) => [
              ...current,
              { kind: "tool", text: toolLabel(event.name, event.arguments) },
            ]);
          else if (event.type === "progress")
            setLog((current) => [
              ...current,
              { kind: "tool", text: `… ${event.message}` },
            ]);
          else if (event.type === "tool_result" && !event.ok)
            setLog((current) => [
              ...current,
              {
                kind: "error",
                text: `${event.name}: ${event.error ?? "błąd"}`,
              },
            ]);
          else if (event.type === "proposals_changed") {
            setLog((current) => [
              ...current,
              {
                kind: "tool",
                text: `${event.action === "update_proposals" ? "Poprawiono" : "Odrzucono"} propozycje: ${event.count}`,
              },
            ]);
            void onAdded();
          } else if (event.type === "examples_parked") {
            setLog((current) => [
              ...current,
              {
                kind: "tool",
                text: `Przeniesiono do „bez splitu” (poza trening): ${event.count}`,
              },
            ]);
            void onAdded();
          } else if (event.type === "proposals") {
            setLog((current) => [
              ...current,
              {
                kind: "tool",
                text: `Zapisano w zakładce Propozycje: ${event.saved} (partia ${event.batch})`,
              },
            ]);
            void onAdded();
          } else if (event.type === "error")
            setLog((current) => [
              ...current,
              { kind: "error", text: event.message },
            ]);
        },
        controller.signal,
        conversationId,
      );
    } catch (requestError) {
      if (!controller.signal.aborted)
        setError(
          requestError instanceof Error
            ? requestError.message
            : String(requestError),
        );
    } finally {
      abortRef.current = null;
      setBusy(false);
      void refreshFiles(conversationId);
    }
  };
  const uploadFiles = async (files: FileList | File[] | null) => {
    const list = files ? Array.from(files) : [];
    if (!list.length || !conversationId) return;
    setUploading((count) => count + list.length);
    setError("");
    try {
      for (const file of list) {
        try {
          const attachment = await api.uploadAgentFile(conversationId, file);
          const placeholder = `--- załącznik ${attachment.id}: ${attachment.name}, ${formatBytes(attachment.bytes)} ---`;
          setInput((current) =>
            current.trim()
              ? `${current.trimEnd()}\n${placeholder}`
              : placeholder,
          );
          setLog((current) => [
            ...current,
            {
              kind: "tool",
              text: `Załącznik ${attachment.id} → ${attachment.path} (${
                attachment.handling === "markitdown"
                  ? attachment.converted
                    ? `MarkItDown: ${attachment.converted}, ${attachment.lines} linii`
                    : `konwersja nieudana: ${attachment.error}`
                  : attachment.handling === "text"
                    ? `tekst, ${attachment.lines} linii`
                    : "plik binarny"
              })`,
            },
          ]);
        } finally {
          setUploading((count) => count - 1);
        }
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : String(requestError),
      );
    } finally {
      if (uploadRef.current) uploadRef.current.value = "";
      void refreshFiles(conversationId);
    }
  };
  const sessionFiles = session?.files ?? [];
  const visibleFiles = sessionFiles.filter(
    (file) =>
      file.path.includes("/") &&
      !file.path.endsWith(".md.json") &&
      file.path !== "notes/notes.jsonl",
  );
  const attachmentsByPath = new Map(
    (session?.attachments ?? []).map((item) => [item.path, item]),
  );
  const copyTranscript = () =>
    void navigator.clipboard.writeText(
      log
        .filter((item) => item.kind === "user" || item.kind === "assistant")
        .map(
          (item) =>
            `${item.kind === "user" ? "user" : "asystent"}:\n${item.text}`,
        )
        .join("\n\n"),
    );
  return (
    <aside
      className={`entity-agent${dragging ? " dragging" : ""}`}
      onDragOver={(event) => {
        if (!corpus || !event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null))
          setDragging(false);
      }}
      onDrop={(event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();
        setDragging(false);
        void uploadFiles(event.dataTransfer.files);
      }}
    >
      {dragging && (
        <div className="entity-agent-dropzone">
          Upuść pliki — dołączę je do wiadomości
        </div>
      )}
      <div className="entity-agent-header">
        <div className="d-flex justify-content-between align-items-center gap-2">
          <h2 className="h6 mb-0">
            <Sparkles size={16} className="me-1" />
            Asystent
          </h2>
          <div className="d-flex gap-1">
            <button
              className={`btn btn-sm ${filesOpen ? "btn-secondary" : "btn-outline-secondary"}`}
              type="button"
              title="Pliki rozmowy"
              disabled={!corpus}
              onClick={() => {
                setFilesOpen((open) => !open);
                if (conversationId) void refreshFiles(conversationId);
              }}
            >
              <FolderOpen size={16} />
              {visibleFiles.length > 0 && (
                <span className="ms-1">{visibleFiles.length}</span>
              )}
            </button>
            <button
              className="btn btn-sm btn-outline-secondary"
              type="button"
              title="Kopiuj rozmowę (user/asystent)"
              disabled={!log.length}
              onClick={copyTranscript}
            >
              <Copy size={16} />
            </button>
            <button
              className="btn btn-sm btn-outline-secondary"
              type="button"
              disabled={busy || (!log.length && !visibleFiles.length)}
              onClick={() => {
                if (conversationId)
                  void api.closeAgentSession(conversationId).catch(() => null);
                setLog([]);
                setSession(null);
                setConversationId(crypto.randomUUID());
              }}
            >
              Nowa rozmowa
            </button>
          </div>
        </div>
        <select
          className="form-select form-select-sm mt-2"
          value={model}
          onChange={(event) => {
            setModel(event.target.value);
            localStorage.setItem("entity-agent-model", event.target.value);
          }}
          aria-label="Model agenta"
        >
          {models.map((item) => (
            <option key={item.id} value={item.id} disabled={!item.available}>
              {item.label}
              {item.available ? "" : " (brak klucza API)"}
            </option>
          ))}
        </select>
      </div>
      {filesOpen && conversationId && (
        <SessionFilesDrawer
          sessionId={conversationId}
          files={visibleFiles}
          attachmentsByPath={attachmentsByPath}
          onRefresh={() => void refreshFiles(conversationId)}
          onClose={() => setFilesOpen(false)}
        />
      )}
      <div className="entity-agent-log" ref={logRef}>
        {!corpus && (
          <p className="text-secondary small">
            Wybierz korpus po lewej, aby rozpocząć rozmowę.
          </p>
        )}
        {corpus && !log.length && (
          <p className="text-secondary small">
            Asystent ogólnego przeznaczenia: rozmawia na dowolny temat, może
            przeglądać korpus „{corpus.name}”, czytać dołączone pliki i
            proponować nowe przykłady w formacie korpusu (trafiają do zakładki
            Propozycje do Twojej oceny, poza trening). Pliki przeciągnij do okna
            czatu.
          </p>
        )}
        {log.map((item, index) =>
          item.kind === "tool" ? (
            <div className="entity-agent-tool" key={index}>
              → {item.text}
            </div>
          ) : item.kind === "error" ? (
            <div className="text-danger small" key={index}>
              {item.text}
            </div>
          ) : (
            <article className={`chat-message ${item.kind}`} key={index}>
              <strong>{item.kind === "user" ? "Ty" : "Asystent"}</strong>
              <MessageContent
                content={item.text}
                markdown={item.kind === "assistant"}
              />
            </article>
          ),
        )}
        {busy && <div className="entity-agent-tool">Asystent pracuje…</div>}
        {uploading > 0 && (
          <div className="entity-agent-tool">
            Wgrywanie i konwersja załączników: {uploading}…
          </div>
        )}
        <form
          className="entity-agent-input"
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          {error && <p className="text-danger small mb-2">{error}</p>}
          <div className="agent-composer">
            <textarea
              className="form-control"
              rows={5}
              value={input}
              disabled={!corpus}
              placeholder="Napisz wiadomość albo przeciągnij pliki. Enter wysyła, Shift+Enter nowa linia"
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
            />
            <input
              ref={uploadRef}
              type="file"
              multiple
              hidden
              onChange={(event) => void uploadFiles(event.target.files)}
            />
            <button
              className="btn btn-sm btn-link agent-composer-attach"
              type="button"
              title="Dołącz pliki (możesz też przeciągnąć je do okna czatu)"
              aria-label="Dołącz pliki"
              disabled={!corpus || !conversationId}
              onClick={() => uploadRef.current?.click()}
            >
              <Paperclip size={16} />
            </button>
            <div className="agent-composer-actions">
              {busy && (
                <button
                  className="btn btn-sm btn-outline-danger"
                  type="button"
                  title="Przerwij"
                  aria-label="Przerwij"
                  onClick={() => abortRef.current?.abort()}
                >
                  <Square size={14} />
                </button>
              )}
              <button
                className="btn btn-sm btn-primary"
                type="submit"
                disabled={
                  busy || uploading > 0 || !corpus || !model || !input.trim()
                }
              >
                <Send size={14} className="me-1" /> Wyślij
              </button>
            </div>
          </div>
        </form>
      </div>
    </aside>
  );
}

type TransformName = "wrap_entities_summary" | "pretty_json" | "compact_json";

const TRANSFORM_LABELS: Record<
  TransformName,
  { title: string; description: string }
> = {
  wrap_entities_summary: {
    title: "Tekst + JSON → {entities, summary}",
    description:
      "Tekst przed pierwszym obiektem JSON trafia do summary, obiekty JSON do listy entities.",
  },
  pretty_json: {
    title: "JSON → pretty (wcięcia)",
    description:
      "Odpowiedzi będące czystym JSON-em są formatowane z wcięciem 2 spacji; treść i kolejność kluczy bez zmian.",
  },
  compact_json: {
    title: "JSON → kompaktowy (jedna linia)",
    description:
      "Odpowiedzi będące czystym JSON-em są zapisywane w jednej linii bez zbędnych spacji (mniej tokenów przy uczeniu).",
  },
};

const JSON_TOKEN =
  /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
const CLAIM_MARKER = /\[((?:R\d+\.)?[FIS]\d*)\]/g;
const CLAIM_MARKER_ONLY = /^\[((?:R\d+\.)?([FIS])\d*)\]$/;

function MarkdownContent({ content }: { content: string }) {
  // Turn [F3] / [R1.I2] markers into inline code outside fenced blocks so they can be rendered as badges.
  const marked = content
    .split(/(```[\s\S]*?```)/g)
    .map((part, index) =>
      index % 2 ? part : part.replace(CLAIM_MARKER, "`[$1]`"),
    )
    .join("");
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ className, children }) => {
            const marker = CLAIM_MARKER_ONLY.exec(String(children));
            if (marker && !className)
              return (
                <span
                  className={`claim-marker ${marker[2] === "F" ? "fact" : "interpretation"}`}
                  title={marker[2] === "F" ? "Fakt ze źródła" : "Interpretacja"}
                >
                  {marker[1]}
                </span>
              );
            return <code className={className}>{children}</code>;
          },
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {marked}
      </ReactMarkdown>
    </div>
  );
}

function MessageContent({
  content,
  markdown = false,
}: {
  content: string;
  markdown?: boolean;
}) {
  if (!/^\s*[{[]/.test(content))
    return markdown ? <MarkdownContent content={content} /> : <p>{content}</p>;
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of content.matchAll(JSON_TOKEN)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(content.slice(last, start));
    const [token, string, colon, literal, number] = match;
    const className = string
      ? colon
        ? "json-key"
        : "json-string"
      : literal
        ? "json-literal"
        : number
          ? "json-number"
          : "";
    parts.push(
      <span className={className} key={start}>
        {string ?? token}
      </span>,
    );
    if (colon) parts.push(colon);
    last = start + token.length;
  }
  parts.push(content.slice(last));
  return <p className="json-content">{parts}</p>;
}

type CorpusView =
  | "list"
  | "proposals"
  | "analysis"
  | "duplicates"
  | "vocabulary"
  | "dpo"
  | "settings";

const DEFAULT_SPLIT_RATIO: SplitRatio = { train: 80, validation: 10, test: 10 };

// One editor for both tabs: they save the same corpus settings object.
function CorpusSettingsView({
  corpusId,
  onSaved,
  section,
}: {
  corpusId: string;
  onSaved: () => void;
  section: "general" | "vocabulary";
}) {
  const [settings, setSettings] = useState<CorpusSettings | null>(null);
  const [models, setModels] = useState<AgentModel[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(
    null,
  );
  useEffect(() => {
    setSettings(null);
    setMessage(null);
    void api.corpusSettings(corpusId).then(setSettings);
    void api.agentModels().then((result) => setModels(result.models));
  }, [corpusId]);
  if (!settings) return <div className="text-secondary">Wczytywanie…</div>;
  const ratio = settings.split_ratio ?? DEFAULT_SPLIT_RATIO;
  const ratioSum = ratio.train + ratio.validation + ratio.test;
  const types = settings.entity_types ?? [];
  const setTypes = (next: EntityTypeDefinition[]) =>
    setSettings({ ...settings, entity_types: next });
  const updateType = (index: number, change: Partial<EntityTypeDefinition>) =>
    setTypes(
      types.map((item, i) => (i === index ? { ...item, ...change } : item)),
    );
  async function loadTypesFromCorpus() {
    const analysis = await api.corpusAnalysis(corpusId);
    const known = new Set(types.map((item) => item.name));
    setTypes([
      ...types,
      ...[...analysis.corpus.types, ...(analysis.corpus.labels ?? [])]
        .filter((row) => row.type !== "(bez typu)" && !known.has(row.type))
        .map((row) => ({ name: row.type, definition: "", boundary: "" })),
    ]);
  }
  const setRatio = (split: keyof SplitRatio, value: number) =>
    setSettings({
      ...settings,
      split_ratio: { ...ratio, [split]: Math.min(100, Math.max(0, value)) },
    });
  async function save() {
    if (!settings) return;
    setSaving(true);
    setMessage(null);
    try {
      setSettings(await api.updateCorpusSettings(corpusId, settings));
      setMessage({
        ok: true,
        text:
          section === "vocabulary"
            ? "Zapisano słownik."
            : "Zapisano ustawienia korpusu.",
      });
      onSaved();
    } catch (error) {
      setMessage({
        ok: false,
        text: error instanceof Error ? error.message : "Nie udało się zapisać.",
      });
    } finally {
      setSaving(false);
    }
  }
  return (
    <section className="corpus-settings">
      {section === "general" && (
        <>
          <div className="mb-4">
            <label
              className="form-label fw-semibold"
              htmlFor="corpus-agent-prompt"
            >
              Dodatkowy prompt systemowy korpusu
            </label>
            <p className="small text-secondary mb-2">
              Opisz, do czego służy ten korpus i na co asystent ma zwracać uwagę
              — tekst trafia do instrukcji asystenta przy każdej rozmowie w tym
              korpusie.
            </p>
            <textarea
              id="corpus-agent-prompt"
              className="form-control"
              rows={8}
              maxLength={20000}
              value={settings.agent_prompt}
              placeholder="Np. Korpus uczy Bielika ekstrakcji wyłączeń odpowiedzialności z OWU. Wiadomość użytkownika zawsze zawiera tytuł artykułu i zdanie wprowadzające…"
              onChange={(event) =>
                setSettings({ ...settings, agent_prompt: event.target.value })
              }
            />
          </div>
          <div className="mb-4">
            <label className="form-label fw-semibold" htmlFor="corpus-model">
              Domyślny model asystenta
            </label>
            <select
              id="corpus-model"
              className="form-select corpus-settings-model"
              value={settings.default_model ?? ""}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  default_model: event.target.value || null,
                })
              }
            >
              <option value="">— jak w panelu asystenta —</option>
              {models.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                  disabled={!item.available}
                >
                  {item.label}
                  {item.available ? "" : " (brak klucza API)"}
                </option>
              ))}
            </select>
          </div>
          <div className="mb-4">
            <label
              className="form-label fw-semibold"
              htmlFor="corpus-exchanges"
            >
              Wymiany w przykładach asystenta
            </label>
            <select
              id="corpus-exchanges"
              className="form-select corpus-settings-model"
              value={settings.max_exchanges ?? 1}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  max_exchanges: Number(event.target.value),
                })
              }
            >
              <option value={1}>1 — user + assistant</option>
              <option value={2}>do 2 — z dopytaniem (followup)</option>
            </select>
          </div>
        </>
      )}
      {section === "vocabulary" && (
        <div className="mb-4">
          <p className="small text-secondary mb-3">
            Pojęcia, których uczy korpus: typy elementów w ekstrakcji (pole{" "}
            <code>type</code>) i etykiety w klasyfikacji. Gdy słownik nie jest
            pusty, asystent używa tylko tych nazw, a analiza zgłasza pozycje
            spoza słownika. Definicje trafiają do instrukcji asystenta (też dla
            przykładów definicyjnych), a „NIE jest nim” opisuje przypadek
            graniczny — z niego powstają trudne negatywy i pary kontrastowe.
          </p>
          {types.length > 0 && (
            <div className="entity-type-row entity-type-head small text-secondary">
              <span>Nazwa</span>
              <span>Definicja</span>
              <span>NIE jest nim (przypadek graniczny)</span>
              <span />
            </div>
          )}
          {types.map((item, index) => (
            <div className="entity-type-row" key={index}>
              <input
                className="form-control form-control-sm font-monospace"
                value={item.name}
                placeholder="EXCLUSION"
                aria-label="Nazwa typu"
                onChange={(event) =>
                  updateType(index, { name: event.target.value })
                }
              />
              <input
                className="form-control form-control-sm"
                value={item.definition}
                placeholder="Definicja, np. sytuacja, w której ubezpieczyciel nie wypłaci świadczenia"
                aria-label={`Definicja ${item.name}`}
                onChange={(event) =>
                  updateType(index, { definition: event.target.value })
                }
              />
              <input
                className="form-control form-control-sm"
                value={item.boundary}
                placeholder="NIE jest nim, np. sama nazwa choroby bez wyłączenia"
                aria-label={`Granica ${item.name}`}
                onChange={(event) =>
                  updateType(index, { boundary: event.target.value })
                }
              />
              <button
                className="btn btn-sm btn-outline-danger"
                type="button"
                title="Usuń typ ze słownika"
                aria-label={`Usuń ${item.name}`}
                onClick={() => setTypes(types.filter((_, i) => i !== index))}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          <div className="d-flex gap-2 mt-2">
            <button
              className="btn btn-sm btn-outline-primary"
              type="button"
              onClick={() =>
                setTypes([...types, { name: "", definition: "", boundary: "" }])
              }
            >
              Dodaj typ
            </button>
            <button
              className="btn btn-sm btn-outline-secondary"
              type="button"
              title="Dopisuje typy występujące w odpowiedziach korpusu (od najczęstszych); usuń te, które są podkategoriami"
              onClick={() => void loadTypesFromCorpus()}
            >
              Wczytaj typy z korpusu
            </button>
          </div>
        </div>
      )}
      {section === "general" && (
        <div className="mb-4">
          <span className="form-label fw-semibold d-block">
            Docelowe proporcje splitów
          </span>
          <p className="small text-secondary mb-2">
            Asystent przypisuje split nowym propozycjom tak, by zbliżać korpus
            do tych proporcji; zakładka Analiza pokazuje odchylenie.
          </p>
          <div className="d-flex flex-wrap gap-2 align-items-center">
            {(Object.keys(DEFAULT_SPLIT_RATIO) as Array<keyof SplitRatio>).map(
              (split) => (
                <div
                  className="input-group input-group-sm split-ratio-input"
                  key={split}
                >
                  <span className="input-group-text">{split}</span>
                  <input
                    className="form-control"
                    type="number"
                    min={0}
                    max={100}
                    value={ratio[split]}
                    onChange={(event) =>
                      setRatio(split, Number(event.target.value) || 0)
                    }
                    aria-label={`Udział ${split}`}
                  />
                  <span className="input-group-text">%</span>
                </div>
              ),
            )}
            <span
              className={`small ${ratioSum === 100 ? "text-secondary" : "text-danger"}`}
            >
              Suma: {ratioSum}%
            </span>
          </div>
        </div>
      )}
      {message && (
        <div
          className={`alert py-2 ${message.ok ? "alert-success" : "alert-danger"}`}
        >
          {message.text}
        </div>
      )}
      <button
        className="btn btn-primary"
        type="button"
        disabled={saving || ratioSum !== 100}
        onClick={() => void save()}
      >
        {section === "vocabulary" ? "Zapisz słownik" : "Zapisz ustawienia"}
      </button>
    </section>
  );
}

function BalanceBar({
  positive,
  negative,
  unclassified,
}: {
  positive: number;
  negative: number;
  unclassified: number;
}) {
  const total = positive + negative + unclassified || 1;
  return (
    <div
      className="balance-bar"
      title={`pozytywne ${positive} · negatywne ${negative} · do klasyfikacji ${unclassified}`}
    >
      <span
        className="positive"
        style={{ width: `${(positive / total) * 100}%` }}
      />
      <span
        className="negative"
        style={{ width: `${(negative / total) * 100}%` }}
      />
      <span
        className="unclassified"
        style={{ width: `${(unclassified / total) * 100}%` }}
      />
    </div>
  );
}

function PreferenceBatchView({
  corpusId,
  onProgress,
}: {
  corpusId: string;
  onProgress: () => void;
}) {
  const [text, setText] = useState("");
  const [sourceName, setSourceName] = useState("dokument");
  const [chunkChars, setChunkChars] = useState(1500);
  const [maxChunks, setMaxChunks] = useState(30);
  const [hint, setHint] = useState("");
  const [system, setSystem] = useState("");
  const [models, setModels] = useState<Array<ModelChoice & { label: string }>>(
    [],
  );
  const [instructionModel, setInstructionModel] = useState("");
  const [rejectedModel, setRejectedModel] = useState("");
  const [preview, setPreview] = useState<{
    total: number;
    used: number;
    sample: string[];
  } | null>(null);
  const [status, setStatus] = useState<PreferenceBatchStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    void api.paraphraseProviders().then((catalog) => {
      const options = Object.entries(catalog.providers).flatMap(
        ([provider, entry]) =>
          Object.entries(entry.models).map(([model, info]) => ({
            provider,
            model,
            label: `${entry.label} · ${info.label}`,
          })),
      );
      setModels(options);
      const key = (item: ModelChoice) => `${item.provider}::${item.model}`;
      const api_ = options.find((item) => item.provider !== "ollama");
      const local = options.find((item) => item.provider === "ollama");
      setInstructionModel((current) => current || (api_ ? key(api_) : ""));
      setRejectedModel(
        (current) => current || (local ? key(local) : api_ ? key(api_) : ""),
      );
    });
    void api.preferenceBatchStatus().then(setStatus);
  }, []);
  const running = status?.state === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      void api.preferenceBatchStatus().then((next) => {
        setStatus(next);
        onProgress();
      });
    }, 3000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);
  const choice = (value: string): ModelChoice => {
    const [provider, ...rest] = value.split("::");
    return { provider, model: rest.join("::") };
  };
  const payload = (): PreferenceBatchRequest => ({
    text,
    source_name: sourceName,
    chunk_chars: chunkChars,
    max_chunks: maxChunks,
    hint,
    system,
    instruction_model: choice(instructionModel),
    rejected_model: choice(rejectedModel),
  });
  async function run(action: () => Promise<void>) {
    setError("");
    try {
      await action();
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : String(requestError),
      );
    }
  }
  const modelSelect = (
    value: string,
    onChange: (value: string) => void,
    label: string,
  ) => (
    <select
      className="form-select form-select-sm"
      value={value}
      aria-label={label}
      onChange={(event) => onChange(event.target.value)}
    >
      {models.map((item) => (
        <option
          key={`${item.provider}::${item.model}`}
          value={`${item.provider}::${item.model}`}
        >
          {item.label}
        </option>
      ))}
    </select>
  );
  return (
    <section className="corpus-settings">
      <p className="small text-secondary">
        Dokument (rozdziały książki, artykuły, korespondencja…) jest dzielony na
        fragmenty. Dla każdego fragmentu model poleceń dopisuje polecenie, na
        które fragment jest odpowiedzią; <strong>chosen</strong> = oryginalny
        fragment, <strong>rejected</strong> = odpowiedź wybranego modelu na to
        polecenie (najlepiej modelu, który będziesz dostrajać). Pary trafiają do
        Propozycji; po akceptacji eksportujesz je jako DPO JSONL.
      </p>
      <div className="mb-3">
        <div className="d-flex gap-2 align-items-center mb-2">
          <label className="btn btn-sm btn-outline-primary mb-0">
            <FileUp size={15} className="me-1" /> Wczytaj plik .txt / .md
            <input
              className="visually-hidden"
              type="file"
              accept=".txt,.md,.markdown,text/plain"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                setSourceName(file.name);
                void file.text().then(setText);
                setPreview(null);
              }}
            />
          </label>
          <input
            className="form-control form-control-sm"
            value={sourceName}
            onChange={(event) => setSourceName(event.target.value)}
            aria-label="Nazwa źródła"
            style={{ maxWidth: "18rem" }}
          />
          <small className="text-secondary">{text.length} znaków</small>
        </div>
        <textarea
          className="form-control"
          rows={8}
          value={text}
          placeholder="…albo wklej tekst. Rozdziały (Rozdział, #, numeracja) zaczynają nowy fragment."
          onChange={(event) => {
            setText(event.target.value);
            setPreview(null);
          }}
        />
      </div>
      <div className="row g-2 mb-3">
        <div className="col-6 col-md-3">
          <label className="form-label small mb-1">
            Rozmiar fragmentu (znaki)
          </label>
          <input
            className="form-control form-control-sm"
            type="number"
            min={300}
            max={6000}
            value={chunkChars}
            onChange={(event) =>
              setChunkChars(Number(event.target.value) || 1500)
            }
          />
        </div>
        <div className="col-6 col-md-3">
          <label className="form-label small mb-1">Maks. par w partii</label>
          <input
            className="form-control form-control-sm"
            type="number"
            min={1}
            max={500}
            value={maxChunks}
            onChange={(event) => setMaxChunks(Number(event.target.value) || 30)}
          />
        </div>
        <div className="col-12 col-md-3">
          <label className="form-label small mb-1">Model poleceń</label>
          {modelSelect(instructionModel, setInstructionModel, "Model poleceń")}
        </div>
        <div className="col-12 col-md-3">
          <label className="form-label small mb-1">
            Model odpowiedzi „rejected”
          </label>
          {modelSelect(
            rejectedModel,
            setRejectedModel,
            "Model odpowiedzi rejected",
          )}
        </div>
      </div>
      <div className="mb-2">
        <label className="form-label small mb-1">
          Wskazówki do poleceń (opcjonalnie)
        </label>
        <input
          className="form-control form-control-sm"
          value={hint}
          placeholder="Np. polecenie ma prosić o napisanie sceny prozy w stylu autora, z opisem nastroju i bohaterów"
          onChange={(event) => setHint(event.target.value)}
        />
      </div>
      <div className="mb-3">
        <label className="form-label small mb-1">
          Prompt systemowy par (opcjonalnie)
        </label>
        <input
          className="form-control form-control-sm"
          value={system}
          placeholder="Pusty = pary bez promptu systemowego"
          onChange={(event) => setSystem(event.target.value)}
        />
      </div>
      {preview && (
        <div className="alert alert-light border small">
          Fragmentów: {preview.total}; w tej partii: {preview.used}. Pierwszy:
          <pre className="mb-0 mt-1 small text-wrap">
            {preview.sample[0]?.slice(0, 600)}
          </pre>
        </div>
      )}
      {status && status.state !== "idle" && (
        <div
          className={`alert py-2 small ${status.state === "running" ? "alert-info" : "alert-light border"}`}
        >
          {status.state === "running"
            ? "Generowanie… "
            : status.state === "cancelled"
              ? "Przerwano. "
              : "Gotowe. "}
          {status.processed ?? 0} / {status.total ?? 0}; zapisano par:{" "}
          {status.saved ?? 0}
          {status.errors
            ? `; błędy: ${status.errors} (${status.last_error ?? ""})`
            : ""}
        </div>
      )}
      {error && <div className="alert alert-danger py-2">{error}</div>}
      <div className="d-flex flex-wrap gap-2">
        <button
          className="btn btn-outline-secondary"
          type="button"
          disabled={text.length < 50}
          onClick={() =>
            void run(async () =>
              setPreview(await api.previewPreferenceChunks(payload())),
            )
          }
        >
          Podgląd podziału
        </button>
        {running ? (
          <button
            className="btn btn-outline-danger"
            type="button"
            onClick={() =>
              void run(async () => setStatus(await api.cancelPreferenceBatch()))
            }
          >
            Przerwij
          </button>
        ) : (
          <button
            className="btn btn-primary"
            type="button"
            disabled={text.length < 50 || !instructionModel || !rejectedModel}
            onClick={() =>
              void run(async () =>
                setStatus(await api.startPreferenceBatch(corpusId, payload())),
              )
            }
          >
            Generuj pary DPO
          </button>
        )}
        <a
          className="btn btn-outline-primary"
          href={`/api/corpora/${corpusId}/export-dpo`}
          download
        >
          <Download size={15} className="me-1" /> Eksport DPO JSONL
          (zaakceptowane)
        </a>
      </div>
    </section>
  );
}

const TASK_LABELS: Record<string, string> = {
  extraction: "ekstrakcja",
  classification: "klasyfikacja",
  generation: "generowanie",
};

function BalanceTable({
  title,
  note,
  nameHeader,
  rows,
  pending,
}: {
  title: string;
  note: string;
  nameHeader: string;
  rows: AnalysisTypeRow[];
  pending: Map<string, AnalysisTypeRow>;
}) {
  if (!rows.length) return null;
  return (
    <>
      <h2 className="h6">{title}</h2>
      <p className="small text-secondary mb-2">{note}</p>
      <div className="table-responsive mb-3">
        <table className="table table-sm align-middle analysis-table">
          <thead>
            <tr>
              <th>{nameHeader}</th>
              <th>Razem</th>
              <th>Poz.</th>
              <th>Neg.</th>
              <th>Udział neg.</th>
              <th>Balans</th>
              <th>train / val / test</th>
              <th>Propozycje</th>
              <th>Ostrzeżenia</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const waiting = pending.get(row.type);
              return (
                <tr key={row.type}>
                  <td>
                    <code>{row.type}</code>
                  </td>
                  <td>{row.total}</td>
                  <td>{row.positive}</td>
                  <td>{row.negative}</td>
                  <td>
                    {row.negative_share === null
                      ? "—"
                      : `${Math.round(row.negative_share * 100)}%`}
                  </td>
                  <td>
                    <BalanceBar {...row} />
                  </td>
                  <td>
                    {row.train} / {row.validation} / {row.test}
                  </td>
                  <td>
                    {waiting
                      ? `+${waiting.positive} / +${waiting.negative}`
                      : "—"}
                  </td>
                  <td>
                    {row.warnings.map((warning) => (
                      <span
                        className="badge text-bg-warning me-1 mb-1"
                        key={warning}
                      >
                        {warning}
                      </span>
                    ))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function CorpusAnalysisView({
  analysis,
  loading,
  onRefresh,
  onOpenExample,
}: {
  analysis: CorpusAnalysis | null;
  loading: boolean;
  onRefresh: () => void;
  onOpenExample: (exampleId: string) => void;
}) {
  const [scope, setScope] = useState<"corpus" | "proposals">("corpus");
  if (!analysis)
    return (
      <div className="text-secondary">
        {loading ? "Analizuję korpus…" : "Brak analizy."}
      </div>
    );
  const { proposals } = analysis;
  const corpus = scope === "proposals" ? proposals : analysis.corpus;
  const pendingByType = new Map(
    scope === "corpus" ? proposals.types.map((row) => [row.type, row]) : [],
  );
  const pendingByLabel = new Map(
    scope === "corpus"
      ? (proposals.labels ?? []).map((row) => [row.type, row])
      : [],
  );
  const issues = Object.entries(corpus.issues).filter(([, item]) => item.count);
  const lengths = corpus.length_tokens;
  const summary = (part: CorpusAnalysisPart) =>
    `${part.examples} · poz. ${part.flags.positive} · neg. ${part.flags.negative} · do klas. ${part.flags.unclassified}`;
  return (
    <section className="corpus-analysis">
      <div
        className="btn-group btn-group-sm mb-2"
        role="group"
        aria-label="Zakres analizy"
      >
        <button
          className={`btn ${scope === "corpus" ? "btn-primary" : "btn-outline-primary"}`}
          type="button"
          onClick={() => setScope("corpus")}
        >
          Korpus ({analysis.corpus.examples})
        </button>
        <button
          className={`btn ${scope === "proposals" ? "btn-primary" : "btn-outline-primary"}`}
          type="button"
          disabled={!proposals.examples}
          onClick={() => setScope("proposals")}
        >
          Propozycje ({proposals.examples})
        </button>
        <span
          className="btn btn-outline-secondary disabled"
          title="Przykłady przeniesione poza trening; pokaż je w Liście filtrem „bez splitu”"
        >
          Bez splitu ({analysis.unassigned ?? 0})
        </span>
      </div>
      <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
        <span className="badge text-bg-secondary">
          Korpus: {summary(analysis.corpus)}
        </span>
        {proposals.examples > 0 && (
          <span className="badge text-bg-primary">
            Propozycje: {summary(proposals)}
          </span>
        )}
        <span className="badge text-bg-light border text-dark">
          system prompt: {corpus.system_prompt.with} z /{" "}
          {corpus.system_prompt.without} bez
        </span>
        <span className="badge text-bg-light border text-dark">
          wymiany:{" "}
          {Object.entries(corpus.exchanges)
            .map(([count, number]) => `${count}× ${number}`)
            .join(", ")}
        </span>
        {corpus.tasks && (
          <span
            className="badge text-bg-light border text-dark"
            title="Rodzaj zadania wykryty z odpowiedzi (lub zapisany w metadanych przykładu)"
          >
            zadania:{" "}
            {Object.entries(corpus.tasks)
              .filter(([, count]) => count)
              .map(([task, count]) => `${TASK_LABELS[task] ?? task} ${count}`)
              .join(" · ")}
          </span>
        )}
        <span
          className={`badge border ${lengths.max_length && lengths.max > lengths.max_length ? "text-bg-warning" : "text-bg-light text-dark"}`}
          title={`Szacunek: ${lengths.estimate_chars_per_token} znaku na token`}
        >
          tokeny ~ mediana {lengths.median} · p95 {lengths.p95} · max{" "}
          {lengths.max}
          {lengths.max_length ? ` / limit ${lengths.max_length}` : ""}
        </span>
        <button
          className="btn btn-sm btn-outline-secondary ms-auto"
          type="button"
          disabled={loading}
          onClick={onRefresh}
        >
          <RefreshCw size={14} className="me-1" /> Odśwież
        </button>
      </div>
      <h2 className="h6">Jakość przykładów</h2>
      <div className="mb-3">
        {issues.length ? (
          issues.map(([check, item]) => (
            <details className="analysis-issue mb-2" key={check}>
              <summary>
                <span className="badge text-bg-danger me-2">{item.count}</span>
                {item.label}
              </summary>
              <ul className="list-unstyled small mt-2 mb-0">
                {item.examples.map((entry, index) => (
                  <li key={`${entry.id}-${index}`}>
                    <code
                      className="user-select-all me-1"
                      title="Kliknij, aby zaznaczyć całe id"
                    >
                      {entry.id}
                    </code>
                    <button
                      className="btn btn-link btn-sm p-0 me-2"
                      type="button"
                      onClick={() => onOpenExample(entry.id)}
                    >
                      otwórz
                    </button>
                    <span className="text-secondary">{entry.detail}</span>
                  </li>
                ))}
                {item.count > item.examples.length && (
                  <li className="text-secondary">
                    … i {item.count - item.examples.length} kolejnych
                  </li>
                )}
              </ul>
            </details>
          ))
        ) : (
          <div className="text-secondary small">Brak wykrytych problemów.</div>
        )}
      </div>
      <h2 className="h6 mt-3">Split × flaga</h2>
      <div className="table-responsive mb-3">
        <table className="table table-sm align-middle analysis-table">
          <thead>
            <tr>
              <th>Split</th>
              <th>Pozytywne</th>
              <th>Negatywne</th>
              <th>Do klasyfikacji</th>
              <th>Balans</th>
              <th>Udział / cel</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(corpus.splits).map(([split, counts]) => {
              const share = corpus.examples
                ? Math.round(
                    ((counts.positive + counts.negative + counts.unclassified) /
                      corpus.examples) *
                      100,
                  )
                : 0;
              const target = analysis.split_target?.[split as keyof SplitRatio];
              return (
                <tr key={split}>
                  <td>{split}</td>
                  <td>{counts.positive}</td>
                  <td>{counts.negative}</td>
                  <td>{counts.unclassified}</td>
                  <td>
                    <BalanceBar {...counts} />
                  </td>
                  <td
                    className={
                      target !== undefined && Math.abs(share - target) > 5
                        ? "text-danger"
                        : ""
                    }
                  >
                    {share}%{target !== undefined ? ` / ${target}%` : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <BalanceTable
        title="Typy elementów (ekstrakcja)"
        note="Pozytywny liczy się do typów z odpowiedzi, negatywny — do typu, o który pytało polecenie. Kolumna „propozycje” to oczekujące propozycje asystenta (poz./neg.)."
        nameHeader="Typ"
        rows={corpus.types}
        pending={pendingByType}
      />
      <BalanceTable
        title="Etykiety (klasyfikacja)"
        note="Liczność każdej etykiety w odpowiedziach klasyfikacji."
        nameHeader="Etykieta"
        rows={corpus.labels ?? []}
        pending={pendingByLabel}
      />
      {corpus.warnings.length > 0 && (
        <div className="alert alert-warning py-2">
          <strong className="small d-block mb-1">
            Zbalansowanie — do uzupełnienia
          </strong>
          <ul className="small mb-0">
            {corpus.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function CorporaPage({
  corpora,
  onCreated,
  onCorpusUpdated,
  openCreate = false,
  initialSplit,
}: {
  corpora: Corpus[];
  onCreated: (corpus: Corpus) => void;
  onCorpusUpdated: () => void;
  openCreate?: boolean;
  initialSplit?: Split;
}) {
  const navigate = useNavigate();
  const { corpusId } = useParams();
  const selectedCorpus = corpora.find((corpus) => corpus.id === corpusId);
  const loadExamples = () =>
    corpusId ? api.examples(corpusId) : Promise.resolve<Example[]>([]);
  useEffect(() => {
    if (!corpusId && !openCreate && corpora.length)
      navigate(`/corpora/${corpora[0].id}`, { replace: true });
  }, [corpusId, openCreate, corpora, navigate]);
  const splitRef = useRef<HTMLDivElement>(null);
  const [agentWidth, setAgentWidth] = useState<number | null>(
    () => Number(localStorage.getItem("corpora-agent-width")) || null,
  );
  const resizeAgent = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = splitRef.current?.getBoundingClientRect();
    if (!bounds || !event.currentTarget.hasPointerCapture(event.pointerId))
      return;
    const width = Math.round(
      Math.min(Math.max(bounds.right - event.clientX, 320), bounds.width - 480),
    );
    setAgentWidth(width);
    localStorage.setItem("corpora-agent-width", String(width));
  };
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [examplesLoading, setExamplesLoading] = useState(false);
  const [selectedExample, setSelectedExample] = useState<Example | null>(null);
  const [exampleIssues, setExampleIssues] = useState<ExampleIssue[]>([]);
  const [agentRequest, setAgentRequest] = useState<AgentRequest | null>(null);
  const [fixNote, setFixNote] = useState("");
  useEffect(() => setFixNote(""), [selectedExample?.id]);
  function requestFix() {
    if (!selectedExample) return;
    const proposal = selectedExample.metadata.flag === "proposal";
    const text = [
      `Napraw, jeśli to możliwe, ${proposal ? "propozycję" : "przykład korpusu"} id ${selectedExample.id} (split ${selectedExample.split}).`,
      "Problemy z analizy:",
      ...exampleIssues.map(
        (issue) =>
          `- ${issue.label}${issue.detail ? ` — ${issue.detail}` : ""}`,
      ),
      ...(fixNote.trim()
        ? ["Wskazówki użytkownika, jak zmienić:", fixNote.trim()]
        : []),
      proposal
        ? "Popraw ją przez update_proposals."
        : `Nie edytuj korpusu bezpośrednio — wywołaj propose_examples z replaces: "${selectedExample.id}" (poprawiona wersja zastąpi oryginał po akceptacji, bez duplikatu).`,
      "Jeśli naprawa nie jest możliwa bez zgadywania treści, wyjaśnij dlaczego.",
      "",
      "Przykład:",
      "```json",
      JSON.stringify(selectedExample.messages, null, 2),
      "```",
    ].join("\n");
    setAgentRequest({ id: Date.now(), text });
    closeDrawer();
  }
  const [drawerMessages, setDrawerMessages] = useState<Message[]>([]);
  useEffect(() => setExampleIssues([]), [selectedExample?.id]);
  useEffect(() => {
    if (!selectedExample || !drawerMessages.length) return;
    let current = true;
    const timer = window.setTimeout(() => {
      void api
        .exampleIssues(selectedExample.id, drawerMessages)
        .then((issues) => current && setExampleIssues(issues))
        .catch(() => undefined);
    }, 400);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [selectedExample, drawerMessages]);
  const [drawerEditing, setDrawerEditing] = useState(false);
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(
    null,
  );
  useEffect(() => {
    if (editingMessageIndex === null) return;
    const onPointerDown = (event: PointerEvent) => {
      if (
        !(event.target as HTMLElement | null)?.closest(".chat-message.editing")
      )
        setEditingMessageIndex(null);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [editingMessageIndex]);
  const [drawerError, setDrawerError] = useState("");
  const [classification, setClassification] =
    useState<ClassificationStatus | null>(null);
  const [query, setQuery] = useState("");
  const [splitFilter, setSplitFilter] = useState<ExampleSplit | "">("");
  const [flagFilter, setFlagFilter] = useState<ExampleFlag | "">("");
  const [importFilter, setImportFilter] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkSplit, setBulkSplit] = useState<ExampleSplit>("train");
  const [validationPercent, setValidationPercent] = useState(10);
  const corpusValidation = selectedCorpus?.settings?.split_ratio?.validation;
  useEffect(() => {
    if (corpusValidation) setValidationPercent(corpusValidation);
  }, [corpusId, corpusValidation]);
  const [exportSplit, setExportSplit] = useState<ExampleSplit | "all">("all");
  const [importSplit, setImportSplit] = useState<Split>("train");
  const [activeView, setActiveView] = useState<CorpusView>("analysis");
  const [analysis, setAnalysis] = useState<CorpusAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const loadAnalysis = () => {
    if (!corpusId) return;
    setAnalysisLoading(true);
    api
      .corpusAnalysis(corpusId)
      .then(setAnalysis)
      .catch((error) =>
        setImportError(
          error instanceof Error ? error.message : "Analiza nie powiodła się.",
        ),
      )
      .finally(() => setAnalysisLoading(false));
  };
  useEffect(() => {
    setAnalysis(null);
    if (activeView === "analysis") loadAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeView, corpusId]);
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [importNotice, setImportNotice] = useState("");
  const [lastDeletion, setLastDeletion] = useState<{
    trashId: string;
    count: number;
  } | null>(null);
  async function undoDeletion() {
    if (!lastDeletion) return;
    setBusy(true);
    setImportError("");
    try {
      const result = await api.restoreTrash(lastDeletion.trashId);
      setExamples(await loadExamples());
      setLastDeletion(null);
      setImportNotice(`Przywrócono przykłady: ${result.restored}.`);
      onCorpusUpdated();
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się przywrócić przykładów.",
      );
    } finally {
      setBusy(false);
    }
  }
  const [importError, setImportError] = useState("");
  const [transformPreview, setTransformPreview] =
    useState<BulkTransformResult | null>(null);
  const [transformIds, setTransformIds] = useState<string[]>([]);
  const [transformName, setTransformName] = useState<TransformName>(
    "wrap_entities_summary",
  );
  const [lastRevision, setLastRevision] = useState<{
    revisionId: string;
    count: number;
  } | null>(null);
  const reloadExamples = async () => setExamples(await loadExamples());
  async function previewTransform(
    name: TransformName = transformName,
    exampleIds: string[] = [...selectedIds],
  ) {
    setBusy(true);
    setImportError("");
    try {
      setTransformName(name);
      setTransformPreview(await api.bulkTransform(exampleIds, name, true));
      setTransformIds(exampleIds);
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się przygotować podglądu.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function applyTransform() {
    setBusy(true);
    setImportError("");
    try {
      const result = await api.bulkTransform(
        transformIds,
        transformName,
        false,
      );
      setTransformPreview(null);
      setSelectedIds(new Set());
      await reloadExamples();
      if (result.revision_id)
        setLastRevision({
          revisionId: result.revision_id,
          count: result.matched,
        });
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się przekształcić odpowiedzi.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function revertTransform() {
    if (!lastRevision) return;
    setBusy(true);
    try {
      const result = await api.revertRevision(lastRevision.revisionId);
      await reloadExamples();
      setLastRevision(null);
      setImportNotice(
        `Przywrócono poprzednią treść odpowiedzi: ${result.reverted}.`,
      );
    } catch (error) {
      setImportError(
        error instanceof Error ? error.message : "Nie udało się cofnąć zmian.",
      );
    } finally {
      setBusy(false);
    }
  }
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
    setSplitFilter(initialSplit ?? "");
  }, [initialSplit]);
  useEffect(() => {
    setExamplesLoading(true);
    setSelectedExample(null);
    const request = loadExamples();
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
          const request = loadExamples();
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
      if (!records.length)
        throw new Error("Plik nie zawiera przykładów JSONL.");
      if (isCorpusDto(records)) {
        await saveImportedExamples(records as ImportedExample[]);
        return;
      }
      const automatic = autoMapRecords(records);
      if (automatic) {
        await saveImportedExamples(automatic.examples, automatic.skipped);
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
  async function saveImportedExamples(
    examples: ImportedExample[],
    skipped: string[] = [],
  ) {
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
    setFlagFilter("");
    setImportFilter(result.import_id);
    setImportNotice(
      `Zaimportowano przykłady: ${result.imported}. Domyślny split: ${importSplit}. Partia: ${result.import_id}.` +
        (skipped.length
          ? ` Pominięto ${skipped.length}: ${skipped.slice(0, 3).join("; ")}${skipped.length > 3 ? "; …" : ""}.`
          : ""),
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
        `Zapisano ręcznie zmieniony system prompt w ${result.updated} przykładach. Pominięto jako niezgodne: ${result.skipped}.`,
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
          : "Nie udało się zaimportować przykładów.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function removeExample(example: Example) {
    if (!window.confirm("Usunąć ten przykład?")) return;
    setBusy(true);
    try {
      const result = await api.deleteExample(example.id);
      setLastDeletion({ trashId: result.trash_id, count: result.deleted });
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
  }
  function closeDrawer() {
    setSelectedExample(null);
    setDrawerEditing(false);
    setEditingMessageIndex(null);
    setDrawerError("");
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
      setDrawerError("Nie można ustalić korpusu dla kopii przykładu.");
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
        metadata: { ...selectedExample.metadata, flag: payload.flag },
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
        error instanceof Error
          ? error.message
          : "Nie udało się zapisać przykładu.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function applyFlag(flag: ExampleFlag) {
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
  async function randomSplitSelected() {
    // Stratified by flag so validation keeps the positive/negative ratio.
    const groups = new Map<string, string[]>();
    examples
      .filter((example) => selectedIds.has(example.id))
      .forEach((example) => {
        const flag = example.metadata.flag ?? "unclassified";
        groups.set(flag, [...(groups.get(flag) ?? []), example.id]);
      });
    const validationIds: string[] = [];
    const trainIds: string[] = [];
    const breakdown: string[] = [];
    groups.forEach((ids, flag) => {
      for (let index = ids.length - 1; index > 0; index -= 1) {
        const other = Math.floor(Math.random() * (index + 1));
        [ids[index], ids[other]] = [ids[other], ids[index]];
      }
      const count = Math.round((ids.length * validationPercent) / 100);
      validationIds.push(...ids.slice(0, count));
      trainIds.push(...ids.slice(count));
      breakdown.push(
        `${flag}: ${ids.length - count} train / ${count} validation`,
      );
    });
    const total = validationIds.length + trainIds.length;
    if (
      !validationIds.length ||
      !window.confirm(
        `Losowo podzielić ${total} zaznaczonych (${validationPercent}% do walidacji z każdej flagi):\n\n${breakdown.join("\n")}\n\nRazem: ${trainIds.length} train, ${validationIds.length} validation.`,
      )
    )
      return;
    setBusy(true);
    try {
      await api.bulkSetSplit(validationIds, "validation");
      if (trainIds.length) await api.bulkSetSplit(trainIds, "train");
      const validationSet = new Set(validationIds);
      setExamples((current) =>
        current.map((example) =>
          selectedIds.has(example.id)
            ? {
                ...example,
                split: validationSet.has(example.id) ? "validation" : "train",
              }
            : example,
        ),
      );
      setSelectedIds(new Set());
      setImportNotice(
        `Podzielono losowo: ${trainIds.length} train, ${validationIds.length} validation.`,
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
    const corpusNames = [
      ...new Set(
        examples
          .filter((example) => selectedIds.has(example.id))
          .map((example) => example.corpus_name ?? selectedCorpus?.name ?? "?"),
      ),
    ];
    if (
      !selectedIds.size ||
      !window.confirm(
        `Usunąć zaznaczone przykłady: ${selectedIds.size}?\nKorpus: ${corpusNames.join(", ")}\n\nUsunięcie można cofnąć przyciskiem „Cofnij”.`,
      )
    )
      return;
    setBusy(true);
    try {
      const exampleIds = [...selectedIds];
      const result = await api.bulkDelete(exampleIds);
      if (result.trash_id)
        setLastDeletion({ trashId: result.trash_id, count: result.deleted });
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
  const corpusExamples = examples.filter(
    (example) => example.metadata.flag !== "proposal",
  );
  const proposals = examples.filter(
    (example) => example.metadata.flag === "proposal",
  );
  const filteredExamples = corpusExamples.filter((example) => {
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
  function stepDrawer(offset: number) {
    const list = activeView === "proposals" ? proposals : filteredExamples;
    if (!selectedExample || drawerEditing || list.length < 2) return;
    const index = list.findIndex(
      (example) => example.id === selectedExample.id,
    );
    const count = list.length;
    openDrawer(list[(index + offset + count) % count]);
  }
  async function acceptProposals(exampleIds: string[]) {
    if (!exampleIds.length) return;
    setBusy(true);
    setImportError("");
    try {
      const result = await api.acceptProposals(exampleIds);
      await reloadExamples();
      setSelectedIds(new Set());
      setImportNotice(
        `Zaakceptowano propozycje: ${result.accepted}.${result.replaced ? ` Zastąpione oryginały przeniesione do kosza: ${result.replaced}.` : ""}`,
      );
      onCorpusUpdated();
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "Nie udało się zaakceptować propozycji.",
      );
    } finally {
      setBusy(false);
    }
  }
  function switchView(view: CorpusView) {
    setActiveView(view);
    setSelectedIds(new Set());
  }
  useEffect(() => {
    if (!selectedExample) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable]")) return;
      if (event.key === "ArrowLeft") stepDrawer(-1);
      else if (event.key === "ArrowRight") stepDrawer(1);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });
  const systemPromptGroups = (() => {
    const groups = new Map<string, Example[]>();
    corpusExamples.forEach((example) => {
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
    <div
      className="corpora-split"
      ref={splitRef}
      style={{
        gridTemplateColumns: agentWidth
          ? `minmax(0, 1fr) 6px ${agentWidth}px`
          : "minmax(0, 2fr) 6px 1fr",
      }}
    >
      <MediumPageTemplate
        eyebrow="PRZEGLĄD"
        title={`Przykłady SFT${selectedCorpus ? `: ${selectedCorpus.name}` : ""}`}
        actions={
          <div className="d-flex flex-column gap-2 align-self-start">
            <div className="d-flex flex-wrap gap-2">
              <div className="input-group input-group-sm export-control">
                <select
                  className="form-select"
                  value={exportSplit}
                  disabled={!selectedCorpus}
                  onChange={(event) =>
                    setExportSplit(event.target.value as ExampleSplit | "all")
                  }
                  aria-label="Split eksportu"
                >
                  <option value="all">wszystkie</option>
                  <option value="train">train</option>
                  <option value="validation">validation</option>
                  <option value="test">test</option>
                  <option value="unassigned">bez splitu</option>
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
                <FilePlus2 size={17} className="me-1" /> Dodaj przykład
              </button>
            </div>
          </div>
        }
      >
        {lastRevision && (
          <div className="alert alert-info d-flex justify-content-between align-items-center gap-2">
            <span>Przekształcono odpowiedzi: {lastRevision.count}.</span>
            <span className="d-flex gap-2">
              <button
                className="btn btn-sm btn-info"
                type="button"
                disabled={busy}
                onClick={() => void revertTransform()}
              >
                Cofnij
              </button>
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                onClick={() => setLastRevision(null)}
              >
                <X size={14} />
              </button>
            </span>
          </div>
        )}
        {lastDeletion && (
          <div className="alert alert-warning d-flex justify-content-between align-items-center gap-2">
            <span>Usunięto przykłady: {lastDeletion.count}.</span>
            <span className="d-flex gap-2">
              <button
                className="btn btn-sm btn-warning"
                type="button"
                disabled={busy}
                onClick={() => void undoDeletion()}
              >
                Cofnij
              </button>
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                onClick={() => setLastDeletion(null)}
              >
                <X size={14} />
              </button>
            </span>
          </div>
        )}
        {importNotice && (
          <div className="alert alert-success">{importNotice}</div>
        )}
        {importError && <div className="alert alert-danger">{importError}</div>}
        <nav className="nav nav-tabs mb-3 mt-3" aria-label="Widok korpusu">
          <button
            className={`nav-link ${activeView === "analysis" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("analysis")}
          >
            Analiza
          </button>
          <button
            className={`nav-link ${activeView === "list" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("list")}
          >
            Lista
          </button>
          <button
            className={`nav-link ${activeView === "proposals" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("proposals")}
          >
            Propozycje {proposals.length ? `(${proposals.length})` : ""}
          </button>
          <button
            className={`nav-link ${activeView === "duplicates" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("duplicates")}
          >
            Duplikaty{" "}
            {duplicateSystemPromptGroups.length
              ? `(${duplicateSystemPromptGroups.length})`
              : ""}
          </button>
          <button
            className={`nav-link ${activeView === "vocabulary" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("vocabulary")}
          >
            Słownik
          </button>
          <button
            className={`nav-link ${activeView === "dpo" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("dpo")}
          >
            Pary DPO
          </button>
          <button
            className={`nav-link ${activeView === "settings" ? "active" : ""}`}
            type="button"
            onClick={() => switchView("settings")}
          >
            Ustawienia
          </button>
        </nav>
        {corpusId && activeView === "list" && (
          <>
            <div className="row g-2 mb-3">
              <div className="col-12 col-md">
                <input
                  className="form-control"
                  placeholder="Filtruj treść przykładów"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </div>
              <div className="col-6 col-md-auto">
                <select
                  className="form-select"
                  value={splitFilter}
                  onChange={(event) =>
                    setSplitFilter(event.target.value as ExampleSplit | "")
                  }
                >
                  <option value="">Wszystkie splity</option>
                  <option value="train">train</option>
                  <option value="validation">validation</option>
                  <option value="test">test</option>
                  <option value="unassigned">bez splitu</option>
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
                  Bielik przechodzi kolejno przez nieoznaczone przykłady z
                  bieżących filtrów.
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
                  <Sparkles size={15} className="me-1" /> Klasyfikuj
                  automatycznie ({automaticCandidates.length})
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
                      setBulkSplit(event.target.value as ExampleSplit)
                    }
                    aria-label="Docelowy split zaznaczonych przykładów"
                  >
                    <option value="train">train</option>
                    <option value="validation">validation</option>
                    <option value="test">test</option>
                    <option value="unassigned">bez splitu</option>
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
                <div className="input-group input-group-sm random-split-control">
                  <span className="input-group-text">Walidacja</span>
                  <input
                    className="form-control"
                    type="number"
                    min={1}
                    max={99}
                    value={validationPercent}
                    disabled={busy}
                    onChange={(event) =>
                      setValidationPercent(
                        Math.min(
                          99,
                          Math.max(1, Number(event.target.value) || 1),
                        ),
                      )
                    }
                    aria-label="Procent do walidacji"
                  />
                  <span className="input-group-text">%</span>
                  <button
                    className="btn btn-outline-primary"
                    type="button"
                    disabled={busy || selectedIds.size < 2}
                    onClick={() => void randomSplitSelected()}
                    title="Losowo przypisz zaznaczone do train/validation"
                  >
                    Losowy podział
                  </button>
                </div>
                <button
                  className="btn btn-sm btn-outline-primary"
                  type="button"
                  disabled={busy}
                  onClick={() => void previewTransform()}
                >
                  Transformacje
                </button>
                <button
                  className="btn btn-sm btn-danger ms-auto"
                  type="button"
                  disabled={busy}
                  onClick={() => void deleteSelected()}
                >
                  <Trash2 size={15} className="me-1" /> Usuń
                </button>
              </div>
            )}
            {examplesLoading ? (
              <div className="text-secondary">Wczytywanie przykładów...</div>
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
                      aria-label="Zaznacz przykład"
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
                          <span
                            className={`badge border ${example.split === "unassigned" ? "text-bg-warning" : "text-bg-light text-dark"}`}
                            title={
                              example.split === "unassigned"
                                ? "Poza treningiem — przywróć splitem albo usuń na stałe"
                                : undefined
                            }
                          >
                            {splitLabel(example.split)}
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
                Brak przykładów dla wybranych filtrów.
              </div>
            )}
          </>
        )}
        {!corpusId && activeView !== "duplicates" && (
          <div className="text-secondary">
            {corpora.length
              ? "Wybierz korpus z listy po lewej."
              : "Brak korpusów — utwórz pierwszy przyciskiem „Nowy korpus”."}
          </div>
        )}
        {corpusId && activeView === "proposals" && (
          <section>
            <p className="text-secondary small">
              Przykłady zaproponowane przez asystenta. Nie trafiają do treningu,
              ewaluacji ani eksportu, dopóki ich nie zaakceptujesz — akceptacja
              nadaje flagę zaproponowaną przez asystenta.
            </p>
            {proposals.length ? (
              <>
                <div className="bulk-actions mb-3">
                  <label className="d-flex align-items-center gap-2 mb-0">
                    <input
                      type="checkbox"
                      checked={proposals.every((example) =>
                        selectedIds.has(example.id),
                      )}
                      onChange={() =>
                        setSelectedIds(
                          proposals.every((example) =>
                            selectedIds.has(example.id),
                          )
                            ? new Set()
                            : new Set(proposals.map((example) => example.id)),
                        )
                      }
                    />
                    Zaznacz wszystkie ({proposals.length})
                  </label>
                  <button
                    className="btn btn-sm btn-success ms-auto"
                    type="button"
                    disabled={busy || !selectedIds.size}
                    onClick={() => void acceptProposals([...selectedIds])}
                  >
                    <Check size={15} className="me-1" /> Akceptuj (
                    {selectedIds.size})
                  </button>
                  <button
                    className="btn btn-sm btn-outline-danger"
                    type="button"
                    disabled={busy || !selectedIds.size}
                    onClick={() => void deleteSelected()}
                  >
                    <Trash2 size={15} className="me-1" /> Odrzuć (
                    {selectedIds.size})
                  </button>
                </div>
                <div className="list-group shadow-sm">
                  {proposals.map((example) => {
                    const exchanges = example.messages.filter(
                      (message) => message.role === "user",
                    ).length;
                    return (
                      <div
                        className={`list-group-item entity-list-item d-flex align-items-start gap-2 ${selectedExample?.id === example.id ? "selected" : ""}`}
                        key={example.id}
                      >
                        <input
                          className="form-check-input mt-1"
                          type="checkbox"
                          checked={selectedIds.has(example.id)}
                          onChange={() => toggleSelection(example.id)}
                          aria-label="Zaznacz propozycję"
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
                              <span className="badge text-bg-primary">
                                propozycja
                              </span>
                              <span
                                className={`badge text-bg-${example.metadata.proposed_flag === "negative" ? "danger" : "success"}`}
                              >
                                {example.metadata.proposed_flag === "negative"
                                  ? "negatywny"
                                  : "pozytywny"}
                              </span>
                              <span className="badge text-bg-light border text-dark">
                                {example.split}
                              </span>
                              {example.metadata.replaces && (
                                <span
                                  className="badge text-bg-warning"
                                  title="Po akceptacji oryginał trafi do kosza"
                                >
                                  naprawa · zastępuje{" "}
                                  {example.metadata.replaces.slice(0, 8)}
                                </span>
                              )}
                              {exchanges > 1 && (
                                <span className="badge text-bg-light border text-dark">
                                  {exchanges} wymiany
                                </span>
                              )}
                              {example.metadata.rejected && (
                                <span
                                  className="badge text-bg-info"
                                  title={`rejected (${example.metadata.rejected_model ?? "model"}): ${example.metadata.rejected.slice(0, 300)}`}
                                >
                                  para DPO
                                </span>
                              )}
                              {!example.messages.some(
                                (message) => message.role === "system",
                              ) && (
                                <span className="badge text-bg-light border text-dark">
                                  bez system promptu
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
                            className="btn btn-sm btn-outline-success"
                            type="button"
                            title="Akceptuj"
                            aria-label="Akceptuj"
                            disabled={busy}
                            onClick={() => void acceptProposals([example.id])}
                          >
                            <Check size={15} />
                          </button>
                          <button
                            className="btn btn-sm btn-outline-danger"
                            type="button"
                            title="Odrzuć"
                            aria-label="Odrzuć"
                            disabled={busy}
                            onClick={() => void removeExample(example)}
                          >
                            <Trash2 size={15} />
                          </button>
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="text-secondary">
                Brak propozycji — poproś asystenta o nowe przykłady.
              </div>
            )}
          </section>
        )}
        {corpusId && activeView === "settings" && (
          <CorpusSettingsView
            corpusId={corpusId}
            onSaved={onCorpusUpdated}
            section="general"
          />
        )}
        {corpusId && activeView === "dpo" && (
          <PreferenceBatchView
            corpusId={corpusId}
            onProgress={() => void reloadExamples()}
          />
        )}
        {corpusId && activeView === "vocabulary" && (
          <CorpusSettingsView
            corpusId={corpusId}
            onSaved={onCorpusUpdated}
            section="vocabulary"
          />
        )}
        {corpusId && activeView === "analysis" && (
          <CorpusAnalysisView
            analysis={analysis}
            loading={analysisLoading}
            onRefresh={loadAnalysis}
            onOpenExample={(exampleId) => {
              const example = examples.find((item) => item.id === exampleId);
              if (example) openDrawer(example);
            }}
          />
        )}
        {activeView === "duplicates" && (
          <section>
            <div className="d-flex flex-wrap gap-2 mb-3">
              <span className="badge text-bg-secondary">
                Unikalne prompty: {systemPromptGroups.length}
              </span>
              <span className="badge text-bg-light border text-dark">
                Przykłady z promptem występującym raz: {uniqueSystemPromptCount}
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
                        <strong>{items.length} przykładów</strong>
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
                          <Pencil size={15} className="me-1" /> Deduplikuj
                          ręcznie
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
        {transformPreview && (
          <div className="modal-backdrop show confirm-backdrop">
            <div className="modal d-block" role="dialog" aria-modal="true">
              <div className="modal-dialog modal-xl">
                <div className="modal-content">
                  <div className="modal-header">
                    <h2 className="h5 modal-title">Transformacje</h2>
                  </div>
                  <div className="modal-body">
                    <select
                      className="form-select mb-2"
                      value={transformName}
                      disabled={busy}
                      onChange={(event) =>
                        void previewTransform(
                          event.target.value as TransformName,
                          transformIds,
                        )
                      }
                      aria-label="Rodzaj przekształcenia"
                    >
                      {Object.entries(TRANSFORM_LABELS).map(([name, label]) => (
                        <option key={name} value={name}>
                          {label.title}
                        </option>
                      ))}
                    </select>
                    <p className="text-secondary small">
                      {TRANSFORM_LABELS[transformName].description} Dotyczy
                      tylko odpowiedzi asystenta; zmiana jest zapisywana z kopią
                      oryginałów i można ją cofnąć.
                    </p>
                    <p className="mb-1">
                      Do zmiany: <strong>{transformPreview.matched}</strong> z{" "}
                      {transformIds.length}
                    </p>
                    {Object.entries(transformPreview.skipped).map(
                      ([reason, count]) => (
                        <div className="small text-secondary" key={reason}>
                          Pominięte ({reason}): {count}
                        </div>
                      ),
                    )}
                    {transformPreview.samples.map((sample) => (
                      <div className="row g-2 mt-2" key={sample.id}>
                        <div className="col-12 col-lg-6">
                          <div className="small text-secondary">Przed</div>
                          <pre className="transform-sample">
                            {sample.before}
                          </pre>
                        </div>
                        <div className="col-12 col-lg-6">
                          <div className="small text-secondary">Po</div>
                          <pre className="transform-sample">{sample.after}</pre>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="modal-footer">
                    <button
                      className="btn btn-outline-secondary"
                      type="button"
                      onClick={() => setTransformPreview(null)}
                    >
                      Anuluj
                    </button>
                    <button
                      className="btn btn-primary"
                      type="button"
                      disabled={busy || !transformPreview.matched}
                      onClick={() => void applyTransform()}
                    >
                      Zastosuj do {transformPreview.matched}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
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
                    {importSession.adapter === "mapping" &&
                      corpusDtoProblem(importSession.records) && (
                        <div className="alert alert-warning small">
                          Plik nie został rozpoznany jako gotowy format korpusu:{" "}
                          {corpusDtoProblem(importSession.records)}
                        </div>
                      )}
                    <p className="text-secondary small">
                      {importSession.adapter === "owu-annotations"
                        ? "Wykryto DTO OWU annotations. Importer zbuduje wiadomości z task, labels, text i target."
                        : "Wybierz klucze wejściowego DTO dla pól przykładów korpusu. Gdy mapujesz messages, pola ról są ignorowane."}
                    </p>
                    {importSession.adapter === "owu-annotations" ? (
                      <div className="alert alert-info mb-0">
                        <code>target</code> zostanie zapisany jako odpowiedź
                        asystenta, a pusty wynik jako klasa{" "}
                        <code>negative</code>.
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
                    <label className="form-label">
                      Co który przykład zmienić?
                    </label>
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
                      {manualPromptIds.length} przykładów tej grupy.
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
                            disabled={
                              paraphrasingPrompt || paraphrasingSelection
                            }
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
                        <strong>{missingPromptTerms.join(", ")}</strong>.
                        Sprawdź je przed walidacją.
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
                          <Sparkles size={15} className="me-1" /> Parafrazuj
                          cały prompt
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
            <aside className="entity-drawer" aria-label="Podgląd przykładu">
              <div className="entity-drawer-header">
                <div>
                  <h2 className="h5 mb-1">Szczegóły przykładu</h2>
                  <small className="text-secondary">
                    {formatCreatedAt(selectedExample.created_at)}
                  </small>
                </div>
                <div className="d-flex flex-wrap justify-content-end gap-2">
                  <button
                    className="btn btn-outline-secondary"
                    type="button"
                    title="Poprzednia (←)"
                    disabled={drawerEditing || filteredExamples.length < 2}
                    onClick={() => stepDrawer(-1)}
                  >
                    ← Poprzednia
                  </button>
                  <button
                    className="btn btn-outline-secondary"
                    type="button"
                    title="Następna (→)"
                    disabled={drawerEditing || filteredExamples.length < 2}
                    onClick={() => stepDrawer(1)}
                  >
                    Następna →
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
              {exampleIssues.length > 0 && (
                <section className="example-issues">
                  <div className="d-flex align-items-center justify-content-between gap-2 mb-2">
                    <p className="panel-title mb-0">PROBLEMY Z ANALIZY</p>
                    <div className="d-flex gap-2">
                      <button
                        className="btn btn-sm btn-outline-primary"
                        type="button"
                        disabled={drawerEditing}
                        onClick={requestFix}
                      >
                        <Sparkles size={15} className="me-1" /> Napraw
                      </button>
                      <button
                        className="btn btn-sm btn-outline-danger"
                        type="button"
                        title="Usuń przykład (do kosza)"
                        aria-label="Usuń przykład"
                        disabled={busy}
                        onClick={() => void removeExample(selectedExample)}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                  <ul className="mb-0">
                    {exampleIssues.map((issue, index) => (
                      <li key={`${issue.check}-${index}`}>
                        <strong>{issue.label}</strong>
                        {issue.detail && (
                          <span className="text-secondary">
                            {" "}
                            — {issue.detail}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                  <textarea
                    className="form-control form-control-sm mt-2"
                    rows={2}
                    value={fixNote}
                    placeholder="Opcjonalnie: jak zmienić (np. skróć fragment do jednego punktu, dodaj klucz summary)…"
                    aria-label="Wskazówki do naprawy"
                    onChange={(event) => setFixNote(event.target.value)}
                  />
                </section>
              )}
              {selectedExample.metadata.rejected && (
                <details className="example-issues dpo-rejected" open>
                  <summary className="panel-title">
                    ODPOWIEDŹ ODRZUCONA (DPO REJECTED ·{" "}
                    {selectedExample.metadata.rejected_model ?? "model"})
                  </summary>
                  <p className="mb-0 mt-2" style={{ whiteSpace: "pre-wrap" }}>
                    {selectedExample.metadata.rejected}
                  </p>
                </details>
              )}
              <div className="d-flex flex-wrap gap-2 mb-3">
                <button
                  className="btn btn-sm btn-outline-success"
                  type="button"
                  disabled={busy || drawerEditing}
                  onClick={() => void applyFlag("positive")}
                >
                  Oznacz ręcznie: positive
                </button>
                <button
                  className="btn btn-sm btn-outline-danger"
                  type="button"
                  disabled={busy || drawerEditing}
                  onClick={() => void applyFlag("negative")}
                >
                  Oznacz ręcznie: negative
                </button>
              </div>
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
                          ref={(element) => {
                            // Long single-line JSON wraps, so line count alone underestimates the height.
                            if (!element) return;
                            element.style.height = "auto";
                            element.style.height = `${element.scrollHeight}px`;
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
                        <MessageContent content={message.content} />
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
      <div
        className="training-splitter corpora-splitter"
        role="separator"
        aria-orientation="vertical"
        aria-label="Zmień szerokość panelu agenta"
        title="Przeciągnij, aby zmienić szerokość; dwuklik przywraca domyślną"
        onPointerDown={(event) =>
          event.currentTarget.setPointerCapture(event.pointerId)
        }
        onPointerMove={resizeAgent}
        onDoubleClick={() => {
          setAgentWidth(null);
          localStorage.removeItem("corpora-agent-width");
        }}
      />
      <EntityAgentPanel
        corpus={selectedCorpus}
        onAdded={reloadExamples}
        request={agentRequest}
      />
    </div>
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
  const [split, setSplit] = useState<ExampleSplit>("train");
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
      eyebrow={editingId ? "EDYCJA PRZYKŁADU" : "KONSTRUKCJA KORPUSU"}
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
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem("chat-system-prompt") ?? "",
  );
  const [systemOpen, setSystemOpen] = useState(false);
  const [systemNotice, setSystemNotice] = useState("");
  useEffect(() => {
    localStorage.setItem("chat-system-prompt", systemPrompt);
  }, [systemPrompt]);
  async function loadTrainingSystemPrompt() {
    setSystemNotice("");
    const counts = new Map<string, number>();
    for (const example of await api.allExamples()) {
      const system = example.messages.find(
        (message) => message.role === "system",
      );
      if (system?.content.trim())
        counts.set(system.content, (counts.get(system.content) ?? 0) + 1);
    }
    const [mostCommon] = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    if (!mostCommon) {
      setSystemNotice("W ostatnich przykładach nie ma promptu systemowego.");
      return;
    }
    setSystemPrompt(mostCommon[0]);
    setSystemNotice(
      `Wstawiono najczęstszy prompt systemowy (${mostCommon[1]} z ostatnich przykładów).`,
    );
  }
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
      const request = systemPrompt.trim()
        ? [{ role: "system" as const, content: systemPrompt.trim() }, ...next]
        : next;
      await api.chatStream(request, model, (content) =>
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
      <div className="card shadow-sm border-0 mb-3">
        <div className="card-body py-2">
          <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
            <button
              className="btn btn-link p-0 text-decoration-none d-flex align-items-center gap-1"
              type="button"
              onClick={() => setSystemOpen((open) => !open)}
            >
              {systemOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              Prompt systemowy
              {systemPrompt.trim() ? (
                <span className="badge text-bg-success ms-1">aktywny</span>
              ) : (
                <span className="badge text-bg-secondary ms-1">brak</span>
              )}
            </button>
            <div className="d-flex gap-2">
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                onClick={() => {
                  setSystemOpen(true);
                  void loadTrainingSystemPrompt();
                }}
              >
                Wstaw z danych treningowych
              </button>
              <button
                className="btn btn-sm btn-outline-secondary"
                type="button"
                disabled={busy || !history.length}
                onClick={() => setHistory([])}
              >
                <Trash2 size={14} className="me-1" /> Wyczyść rozmowę
              </button>
            </div>
          </div>
          {systemOpen && (
            <>
              <textarea
                className="form-control mt-2"
                rows={6}
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                placeholder="Instrukcja systemowa wysyłana na początku każdej rozmowy"
              />
              {systemNotice && (
                <p className="text-secondary small mt-1 mb-0">{systemNotice}</p>
              )}
            </>
          )}
        </div>
      </div>
      <div className="card shadow-sm border-0">
        <div className="card-body chat-history">
          {history.map((message, index) => (
            <article className={`chat-message ${message.role}`} key={index}>
              <strong>{message.role}</strong>
              <MessageContent content={message.content} />
            </article>
          ))}
        </div>
        <form className="card-footer" onSubmit={send}>
          <textarea
            className="form-control"
            rows={4}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                if (!busy && model) event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Enter wysyła, Shift+Enter nowa linia"
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

function logScaleTicks(min: number, max: number) {
  const ticks: number[] = [];
  for (
    let exponent = Math.floor(Math.log10(min));
    exponent <= Math.ceil(Math.log10(max));
    exponent++
  ) {
    for (const base of [1, 2, 3, 5]) {
      const value = base * 10 ** exponent;
      if (value > min * 1.08 && value < max / 1.08) ticks.push(value);
    }
  }
  return [min, ...ticks, max];
}

function LineChart({
  title,
  series,
  format,
  height = 220,
  width = 960,
  scaleToggle = false,
  xLabel = (value) => `krok ${value}`,
}: {
  title: string;
  series: ChartSeries[];
  format: (value: number) => string;
  height?: number;
  width?: number;
  scaleToggle?: boolean;
  xLabel?: (value: number) => string;
}) {
  const [scale, setScale] = useState<"linear" | "progressive">("linear");
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
  const progressive = scale === "progressive" && minY > 0;
  const transform = (y: number) => (progressive ? Math.log10(y) : y);
  const lowY = transform(minY);
  const highY = transform(maxY);
  const ticks = progressive
    ? logScaleTicks(minY, maxY)
    : [minY, (minY + maxY) / 2, maxY];
  const scaleX = (x: number) =>
    padding.left +
    ((x - minX) / (maxX - minX || 1)) * (width - padding.left - padding.right);
  const scaleY = (y: number) =>
    height -
    padding.bottom -
    ((transform(y) - lowY) / (highY - lowY || 1)) *
      (height - padding.top - padding.bottom);
  return (
    <div className="training-chart">
      <div className="d-flex justify-content-between align-items-baseline">
        <h3 className="h6 mb-1">{title}</h3>
        <div className="d-flex align-items-center gap-3 small">
          {series
            .filter((item) => series.length > 1 && item.points.length)
            .map((item) => (
              <span key={item.label}>
                <span
                  className="training-chart-swatch"
                  style={{ background: item.color }}
                />
                {item.label}
              </span>
            ))}
          {scaleToggle && (
            <div className="btn-group btn-group-sm" role="group">
              <button
                type="button"
                className={`btn ${scale === "linear" ? "btn-secondary" : "btn-outline-secondary"}`}
                onClick={() => setScale("linear")}
              >
                Liniowa
              </button>
              <button
                type="button"
                className={`btn ${scale === "progressive" ? "btn-secondary" : "btn-outline-secondary"}`}
                onClick={() => setScale("progressive")}
              >
                Progresywna
              </button>
            </div>
          )}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-100" role="img">
        {ticks.map((value) => (
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
          {xLabel(minX)}
        </text>
        <text
          x={width - padding.right}
          y={height - 6}
          textAnchor="end"
          fontSize="11"
          fill="#6c757d"
        >
          {xLabel(maxX)}
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

function metricSeries(metrics: TrainingMetric[], key: string) {
  return metrics
    .filter((metric) => typeof metric[key] === "number")
    .map((metric) => ({ x: metric.step, y: metric[key] as number }));
}

function lastMetric(metrics: TrainingMetric[], key: string) {
  return [...metrics]
    .reverse()
    .find((metric) => typeof metric[key] === "number")?.[key];
}

function maxMetric(metrics: TrainingMetric[], key: string) {
  const values = metricSeries(metrics, key).map((point) => point.y);
  return values.length ? Math.max(...values) : undefined;
}

const GROUP_COLORS = ["#176b61", "#c0503a", "#3d6fb6", "#d9822b", "#8e5bb5"];

function DatasetStatsPanel({ stats }: { stats: DatasetStats }) {
  return (
    <div className="card shadow-sm border-0 mt-3">
      <div className="card-body">
        <h3 className="h6">
          Jakość danych · długość przykładów w tokenach (max_length{" "}
          {stats.max_length})
        </h3>
        <div className="row g-4">
          {Object.entries(stats.splits).map(([split, item]) => (
            <div className="col-12 col-lg-6" key={split}>
              <div className="d-flex justify-content-between align-items-baseline">
                <strong>{split}</strong>
                <span
                  className={
                    item.truncated ? "text-danger small" : "text-success small"
                  }
                >
                  Ucięte: {item.truncated} ({item.truncated_pct}%)
                </span>
              </div>
              <div className="small text-secondary mb-2">
                {item.count} przykładów · średnio {item.mean} · p50 {item.p50} ·
                p90 {item.p90} · p99 {item.p99} · max {item.max}
              </div>
              <div className="histogram">
                {item.histogram.map((count, index) => {
                  const from = index * item.bin_width;
                  const peak = Math.max(...item.histogram, 1);
                  return (
                    <div
                      key={index}
                      className={`histogram-bar ${from >= stats.max_length ? "over" : ""}`}
                      style={{ height: `${(count / peak) * 100}%` }}
                      title={`${from}–${from + item.bin_width} tokenów: ${count}`}
                    />
                  );
                })}
              </div>
              <div className="d-flex justify-content-between small text-secondary">
                <span>0</span>
                <span>{item.histogram.length * item.bin_width}</span>
              </div>
            </div>
          ))}
        </div>
        {Object.values(stats.splits).some((item) => item.truncated) && (
          <p className="small text-danger mb-0 mt-2">
            Czerwone słupki przekraczają max_length — końcówki tych odpowiedzi
            są ucinane, więc model uczy się niedokończonego JSON-a. Rozważ
            większy max_length albo kompaktowy JSON.
          </p>
        )}
      </div>
    </div>
  );
}

function GroupLossPanel({
  metrics,
  counts,
}: {
  metrics: TrainingMetric[];
  counts: Record<string, number>;
}) {
  const keys = [
    ...new Set(
      metrics.flatMap((metric) =>
        Object.keys(metric).filter((key) => key.startsWith("group_loss/")),
      ),
    ),
  ];
  if (!keys.length) return null;
  const flagKeys = keys.filter((key) => key.startsWith("group_loss/flag:"));
  const typeRows = keys
    .filter((key) => key.startsWith("group_loss/type:"))
    .map((key) => {
      const points = metricSeries(metrics, key);
      const group = key.slice("group_loss/".length);
      return {
        name: group.slice("type:".length),
        count: counts[group],
        first: points[0]?.y,
        last: points.at(-1)?.y ?? 0,
      };
    })
    .sort((left, right) => right.last - left.last);
  const worst = Math.max(...typeRows.map((row) => row.last), 1e-9);
  return (
    <div className="card shadow-sm border-0 mt-3">
      <div className="card-body">
        <div className="row g-4">
          <div className="col-12 col-lg-6">
            <LineChart
              title="Eval loss wg flagi"
              width={480}
              height={200}
              format={(value) => value.toFixed(3)}
              series={flagKeys.map((key, index) => ({
                label: `${key.slice("group_loss/flag:".length)} (${counts[key.slice("group_loss/".length)] ?? "?"})`,
                color: GROUP_COLORS[index % GROUP_COLORS.length],
                points: metricSeries(metrics, key),
              }))}
            />
          </div>
          <div className="col-12 col-lg-6">
            <h3 className="h6 mb-1">
              Eval loss wg typu encji (ostatnia ewaluacja)
            </h3>
            <div className="group-loss-table">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr>
                    <th>Typ</th>
                    <th className="text-end">Przykłady</th>
                    <th className="text-end">Loss</th>
                    <th className="text-end">Zmiana</th>
                    <th style={{ width: "30%" }} />
                  </tr>
                </thead>
                <tbody>
                  {typeRows.map((row) => (
                    <tr key={row.name}>
                      <td className="text-break">{row.name}</td>
                      <td className="text-end">{row.count ?? "-"}</td>
                      <td className="text-end">{row.last.toFixed(3)}</td>
                      <td
                        className={`text-end ${row.first !== undefined && row.last > row.first ? "text-danger" : "text-success"}`}
                      >
                        {row.first !== undefined
                          ? (row.last - row.first).toFixed(3)
                          : "-"}
                      </td>
                      <td>
                        <div
                          className="group-loss-bar"
                          style={{ width: `${(row.last / worst) * 100}%` }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LoraLayersPanel({
  snapshot,
  clipped,
  steps,
  maxGradNorm,
}: {
  snapshot: LoraLayerSnapshot | null | undefined;
  clipped: number;
  steps: number;
  maxGradNorm: number;
}) {
  const columns: Array<{
    title: string;
    key: "b_norm" | "grad_norm";
    color: string;
  }> = [
    { title: "Norma wag LoRA B per warstwa", key: "b_norm", color: "#3d6fb6" },
    {
      title: "Norma gradientu LoRA per warstwa",
      key: "grad_norm",
      color: "#c0503a",
    },
  ];
  return (
    <div className="card shadow-sm border-0 mt-3">
      <div className="card-body">
        <div className="d-flex justify-content-between align-items-baseline">
          <h3 className="h6">
            Diagnostyka LoRA{snapshot ? ` · krok ${snapshot.step}` : ""}
          </h3>
          <span
            className={`small ${clipped ? "text-warning" : "text-secondary"}`}
          >
            Przycinanie gradientu (grad_norm &gt; {maxGradNorm}): {clipped} /{" "}
            {steps} kroków
            {steps ? ` (${((clipped / steps) * 100).toFixed(0)}%)` : ""}
          </span>
        </div>
        {snapshot?.layers.length ? (
          <div className="row g-4">
            {columns.map((column) => {
              const values = snapshot.layers.map(
                (layer) => layer[column.key] ?? 0,
              );
              const peak = Math.max(...values, 1e-12);
              return (
                <div className="col-12 col-lg-6" key={column.key}>
                  <div className="small text-secondary mb-1">
                    {column.title}
                  </div>
                  <div className="histogram">
                    {snapshot.layers.map((layer, index) => (
                      <div
                        key={layer.layer}
                        className="histogram-bar"
                        style={{
                          height: `${(values[index] / peak) * 100}%`,
                          background: column.color,
                        }}
                        title={`warstwa ${layer.layer}: ${values[index].toExponential(2)}`}
                      />
                    ))}
                  </div>
                  <div className="d-flex justify-content-between small text-secondary">
                    <span>warstwa {snapshot.layers[0].layer}</span>
                    <span>warstwa {snapshot.layers.at(-1)?.layer}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-secondary small mb-0">
            Brak danych per warstwa — pojawią się w kolejnym treningu.
          </p>
        )}
      </div>
    </div>
  );
}

function TrainingDashboard({ status }: { status: TrainingStatus | null }) {
  const metrics = status?.metrics ?? [];
  const hyperparameters = status?.hyperparameters;
  const last = metrics.at(-1);
  const lossPoints = metricSeries(metrics, "loss");
  const evalPoints = metricSeries(metrics, "eval_loss");
  const learningRatePoints = metricSeries(metrics, "learning_rate");
  const smallCharts: Array<{
    title: string;
    format: (value: number) => string;
    series: ChartSeries[];
  }> = [
    {
      title: "Learning rate",
      format: (value) => value.toExponential(1),
      series: [{ label: "lr", color: "#3d6fb6", points: learningRatePoints }],
    },
    {
      title: "Mean token accuracy",
      format: (value) => `${(value * 100).toFixed(1)}%`,
      series: [
        {
          label: "train",
          color: "#176b61",
          points: metricSeries(metrics, "mean_token_accuracy"),
        },
        {
          label: "eval",
          color: "#d9822b",
          points: metricSeries(metrics, "eval_mean_token_accuracy"),
        },
      ],
    },
    {
      title: "Perplexity = exp(loss)",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "train",
          color: "#176b61",
          points: metricSeries(metrics, "perplexity"),
        },
        {
          label: "eval",
          color: "#d9822b",
          points: metricSeries(metrics, "eval_perplexity"),
        },
      ],
    },
    {
      title: "Przeuczenie: eval_loss − średni train loss",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "różnica",
          color: "#c0503a",
          points: metricSeries(metrics, "generalization_gap"),
        },
      ],
    },
    {
      title: "Entropy",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "entropy",
          color: "#8e5bb5",
          points: metricSeries(metrics, "entropy"),
        },
      ],
    },
    {
      title: "Grad norm (przed przycięciem)",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "grad_norm",
          color: "#c0503a",
          points: metricSeries(metrics, "grad_norm"),
        },
      ],
    },
    {
      title: "Tokeny / s",
      format: (value) => value.toFixed(0),
      series: [
        {
          label: "tokeny/s",
          color: "#176b61",
          points: metricSeries(metrics, "tokens_per_second"),
        },
      ],
    },
    {
      title: "Sekundy / krok",
      format: (value) => value.toFixed(1),
      series: [
        {
          label: "s/krok",
          color: "#3d6fb6",
          points: metricSeries(metrics, "seconds_per_step"),
        },
      ],
    },
    {
      title: "VRAM (GiB)",
      format: (value) => value.toFixed(2),
      series: [
        {
          label: "szczyt alokacji",
          color: "#c0503a",
          points: metricSeries(metrics, "vram_peak_gb"),
        },
        {
          label: "zarezerwowane",
          color: "#8e5bb5",
          points: metricSeries(metrics, "vram_reserved_gb"),
        },
        {
          label: "nvidia-smi",
          color: "#6c757d",
          points: metricSeries(metrics, "gpu_memory_used_gb"),
        },
      ],
    },
    {
      title: "GPU obciążenie (%)",
      format: (value) => value.toFixed(0),
      series: [
        {
          label: "util",
          color: "#176b61",
          points: metricSeries(metrics, "gpu_util"),
        },
      ],
    },
    {
      title: "GPU temperatura (°C)",
      format: (value) => value.toFixed(0),
      series: [
        {
          label: "temp",
          color: "#d9822b",
          points: metricSeries(metrics, "gpu_temp"),
        },
      ],
    },
    {
      title: "GPU pobór mocy (W)",
      format: (value) => value.toFixed(0),
      series: [
        {
          label: "moc",
          color: "#c0503a",
          points: metricSeries(metrics, "gpu_power"),
        },
      ],
    },
    {
      title: "LoRA: norma wag adapterów",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "A",
          color: "#8e5bb5",
          points: metricSeries(metrics, "lora_a_norm"),
        },
        {
          label: "B",
          color: "#3d6fb6",
          points: metricSeries(metrics, "lora_b_norm"),
        },
      ],
    },
    {
      title: "LoRA: norma gradientu (po przycięciu)",
      format: (value) => value.toFixed(3),
      series: [
        {
          label: "grad",
          color: "#c0503a",
          points: metricSeries(metrics, "lora_grad_norm"),
        },
      ],
    },
  ];
  const maxGradNorm =
    metrics.find((metric) => typeof metric.max_grad_norm === "number")
      ?.max_grad_norm ?? 1;
  const gradNormPoints = metricSeries(metrics, "grad_norm");
  const clippedSteps = gradNormPoints.filter(
    (point) => point.y > maxGradNorm,
  ).length;
  const gapPoints = metricSeries(metrics, "generalization_gap");
  const lastGap = gapPoints.at(-1)?.y;
  const gapRising =
    lastGap !== undefined &&
    gapPoints.length > 1 &&
    lastGap > 0 &&
    lastGap > gapPoints[gapPoints.length - 2].y + 0.01;
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
        <div className="col-12 col-md-6 col-xxl-4">
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
        <div className="col-12 col-md-6 col-xxl-4">
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
        <div className="col-12 col-md-6 col-xxl-4">
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
        <div className="col-12 col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <p className="text-secondary mb-1">Perplexity (train / eval)</p>
              <div className="display-6">
                {formatMetric(lastMetric(metrics, "perplexity"), 3)}
                <span className="fs-4 text-secondary">
                  {" / "}
                  {formatMetric(lastMetric(metrics, "eval_perplexity"), 3)}
                </span>
              </div>
              <dl className="training-stats mt-2 mb-0">
                <dt>Eval − train loss</dt>
                <dd className={gapRising ? "text-danger" : ""}>
                  {formatMetric(lastGap)}
                  {gapRising && " ↑ rośnie — możliwe przeuczenie"}
                </dd>
                <dt>Eval accuracy</dt>
                <dd>
                  {lastMetric(metrics, "eval_mean_token_accuracy") !== undefined
                    ? `${((lastMetric(metrics, "eval_mean_token_accuracy") ?? 0) * 100).toFixed(1)}%`
                    : "-"}
                </dd>
                <dt>Przycięte gradienty</dt>
                <dd>
                  {gradNormPoints.length
                    ? `${clippedSteps} / ${gradNormPoints.length} (limit ${maxGradNorm})`
                    : "-"}
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <p className="text-secondary mb-1">Wydajność i GPU</p>
              <div className="display-6">
                {formatMetric(lastMetric(metrics, "tokens_per_second"), 0)}
                <span className="fs-4 text-secondary"> tok/s</span>
              </div>
              <dl className="training-stats mt-2 mb-0">
                <dt>VRAM szczyt (ostatni / max)</dt>
                <dd>
                  {formatMetric(lastMetric(metrics, "vram_peak_gb"), 2)} /{" "}
                  {formatMetric(maxMetric(metrics, "vram_peak_gb"), 2)} GiB
                </dd>
                <dt>GPU obciążenie</dt>
                <dd>{formatMetric(lastMetric(metrics, "gpu_util"), 0)} %</dd>
                <dt>Temperatura (max)</dt>
                <dd>
                  {formatMetric(lastMetric(metrics, "gpu_temp"), 0)} °C (
                  {formatMetric(maxMetric(metrics, "gpu_temp"), 0)} °C)
                </dd>
                <dt>Pobór mocy</dt>
                <dd>{formatMetric(lastMetric(metrics, "gpu_power"), 0)} W</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>
      <div className="card shadow-sm border-0 mt-4">
        <div className="card-body">
          <LineChart
            title="Loss"
            height={270}
            scaleToggle
            format={(value) => value.toFixed(3)}
            series={[
              { label: "train", color: "#176b61", points: lossPoints },
              { label: "eval", color: "#d9822b", points: evalPoints },
            ]}
          />
        </div>
      </div>
      <div className="card shadow-sm border-0 mt-3">
        <div className="card-body">
          <div className="row g-4">
            {smallCharts.map((chart) => (
              <div className="col-12 col-md-6" key={chart.title}>
                <LineChart
                  title={chart.title}
                  width={480}
                  height={160}
                  format={chart.format}
                  series={chart.series}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
      {status?.dataset_stats && (
        <DatasetStatsPanel stats={status.dataset_stats} />
      )}
      <GroupLossPanel
        metrics={metrics}
        counts={status?.dataset_stats?.groups ?? {}}
      />
      {metrics.length > 0 && (
        <LoraLayersPanel
          snapshot={status?.lora_layers}
          clipped={clippedSteps}
          steps={gradNormPoints.length}
          maxGradNorm={maxGradNorm}
        />
      )}
    </>
  );
}

const LOG_TOKEN_PATTERN =
  /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')(\s*:)?|(?<![\w.])(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?![\w.])|\b(True|False|None|null|true|false)\b/gi;

function logLineClass(line: string) {
  if (line.startsWith("BIELIK_METRIC")) return "log-metric";
  if (/Traceback|Error|Exception|FAILED/.test(line)) return "log-error";
  if (/warn|deprecated/i.test(line)) return "log-warning";
  if (/\d+%\|/.test(line)) return "log-progress";
  return "";
}

function highlightLogLine(line: string) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of line.matchAll(LOG_TOKEN_PATTERN)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(line.slice(last, index));
    const [token, text, colon, number, literal] = match;
    if (text) {
      parts.push(
        <span key={index} className={colon ? "log-key" : "log-string"}>
          {text}
        </span>,
      );
      if (colon) parts.push(colon);
    } else if (number) {
      parts.push(
        <span key={index} className="log-number">
          {number}
        </span>,
      );
    } else if (literal) {
      parts.push(
        <span key={index} className="log-literal">
          {literal}
        </span>,
      );
    }
    last = index + token.length;
  }
  if (last < line.length) parts.push(line.slice(last));
  return parts;
}

function TrainingLogView({ logs }: { logs: string }) {
  const lines = logs
    .replace(/\x1b\[[0-9;]*[A-Za-z]/g, "")
    .split("\n")
    .map((line) => {
      const segments = line.split("\r").filter((segment) => segment.trim());
      return segments.at(-1) ?? "";
    });
  return (
    <pre className="training-logs mb-0">
      {lines.map((line, index) => {
        const lineClass = logLineClass(line);
        return (
          <div key={index} className={lineClass}>
            {lineClass === "log-metric" ? (
              <>
                <span className="log-tag">BIELIK_METRIC</span>
                {highlightLogLine(line.slice("BIELIK_METRIC".length))}
              </>
            ) : lineClass === "log-progress" || lineClass === "log-error" ? (
              line
            ) : (
              highlightLogLine(line)
            )}
            {!line && "\u00a0"}
          </div>
        );
      })}
    </pre>
  );
}

function checkpointLabel(checkpoint: string) {
  if (checkpoint.startsWith("merged-"))
    return `${checkpoint.slice("merged-".length)} · zmergowany (Ollama)`;
  return checkpoint === "base" ? "base (bez adaptera)" : checkpoint;
}

const EXPORT_STAGE_LABELS: Record<string, string> = {
  merge: "wtapianie adaptera w model bazowy",
  convert: "konwersja do GGUF (f16)",
  quantize: "kwantyzacja",
  cleanup: "usuwanie plików pośrednich",
  register: "rejestracja w Ollamie",
};

function ExportSection({
  adapter,
  checkpoint,
  exportsStatus,
  trainingRunning,
  onExports,
}: {
  adapter: string;
  checkpoint: string;
  exportsStatus: ExportsStatus | null;
  trainingRunning: boolean;
  onExports: (status: ExportsStatus) => void;
}) {
  const [quantization, setQuantization] =
    useState<ExportQuantization>("Q4_K_M");
  const [modelName, setModelName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const allExports = exportsStatus?.exports ?? [];
  const exports = allExports.filter((entry) => entry.adapter_name === adapter);
  const stages = exportsStatus?.stages ?? [];
  const running = allExports.some((entry) => entry.state === "running");
  const runningElsewhere = allExports.find(
    (entry) => entry.state === "running" && entry.adapter_name !== adapter,
  );
  const canExport = checkpoint && checkpoint !== "base";
  const run = async (action: () => Promise<ExportsStatus>) => {
    setBusy(true);
    setError("");
    try {
      onExports(await action());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : String(requestError),
      );
    } finally {
      setBusy(false);
    }
  };
  const now = Date.now() / 1000;
  return (
    <div className="border-top mt-3 pt-3">
      <h3 className="h6 mb-1">Zmergowany model w Ollamie (szybki)</h3>
      <p className="text-secondary small mb-2">
        Adapter jest wtapiany w model bazowy, konwertowany do GGUF, kwantyzowany
        i rejestrowany w Ollamie. Taki model od razu jest w czacie i można go
        ewaluować jako „zmergowany” — generuje wielokrotnie szybciej niż
        adapter. Eksport działa na CPU (~25 GB RAM) i trwa
        kilkanaście–kilkadziesiąt minut.
      </p>
      <div className="row g-2 align-items-end">
        <div className="col-12 col-md-3">
          <label className="form-label small mb-1" htmlFor="export-quant">
            Kwantyzacja
          </label>
          <select
            id="export-quant"
            className="form-select"
            value={quantization}
            onChange={(event) =>
              setQuantization(event.target.value as ExportQuantization)
            }
          >
            <option value="Q4_K_M">Q4_K_M (~6.7 GB, zalecana)</option>
            <option value="Q5_K_M">Q5_K_M (~7.9 GB)</option>
            <option value="Q6_K">Q6_K (~9.1 GB)</option>
            <option value="Q8_0">Q8_0 (~11.8 GB)</option>
          </select>
        </div>
        <div className="col-12 col-md-5">
          <label className="form-label small mb-1" htmlFor="export-name">
            Nazwa w Ollamie
          </label>
          <input
            id="export-name"
            className="form-control"
            value={modelName}
            placeholder={`${adapter}-${checkpoint}`}
            onChange={(event) =>
              setModelName(event.target.value.replace(/[^A-Za-z0-9._-]/g, "-"))
            }
          />
        </div>
        <div className="col-12 col-md-4">
          <button
            className="btn btn-outline-primary w-100"
            type="button"
            disabled={busy || running || trainingRunning || !canExport}
            onClick={() =>
              void run(() =>
                api.startExport(adapter, checkpoint, quantization, modelName),
              )
            }
          >
            Zmerguj i wyślij do Ollamy
          </button>
        </div>
      </div>
      {!canExport && (
        <p className="text-secondary small mt-2 mb-0">
          Wybierz checkpoint (nie „base”), aby go zmergować.
        </p>
      )}
      {trainingRunning && (
        <p className="text-secondary small mt-2 mb-0">
          Eksport będzie możliwy po zakończeniu treningu.
        </p>
      )}
      {error && <p className="text-danger small mt-2 mb-0">{error}</p>}
      {runningElsewhere && (
        <p className="text-secondary small mt-2 mb-0">
          Trwa eksport innego adaptera ({runningElsewhere.adapter_name}/
          {runningElsewhere.checkpoint}) — kolejny po jego zakończeniu.
        </p>
      )}
      {exports.length === 0 && (
        <p className="text-secondary small mt-2 mb-0">
          Brak zmergowanych modeli dla adaptera {adapter}.
        </p>
      )}
      {exports.length > 0 && (
        <ul className="list-group list-group-flush mt-3">
          {exports.map((entry) => {
            const stageIndex = stages.indexOf(entry.stage);
            return (
              <li className="list-group-item px-0" key={entry.id}>
                <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
                  <div>
                    <code>{entry.model_name}</code>{" "}
                    <span className="text-secondary small">
                      {entry.adapter_name}/{entry.checkpoint} ·{" "}
                      {entry.quantization}
                      {entry.gguf_bytes
                        ? ` · ${(entry.gguf_bytes / 2 ** 30).toFixed(2)} GiB`
                        : ""}
                    </span>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    {entry.state === "running" && (
                      <span className="badge text-bg-info">
                        Etap {stageIndex + 1}/{stages.length}
                      </span>
                    )}
                    {entry.state === "ready" && (
                      <>
                        <span className="badge text-bg-success">W Ollamie</span>
                        <NavLink className="small" to="/chat">
                          czat
                        </NavLink>
                      </>
                    )}
                    {entry.state === "failed" && (
                      <>
                        <span className="badge text-bg-danger">Błąd</span>
                        <button
                          className="btn btn-sm btn-outline-primary"
                          type="button"
                          title="Wznów od etapu, który się nie udał"
                          disabled={busy || running}
                          onClick={() =>
                            void run(() => api.retryExport(entry.id))
                          }
                        >
                          Ponów od „{entry.stage}”
                        </button>
                      </>
                    )}
                    {entry.state !== "running" && (
                      <button
                        className="btn btn-sm btn-outline-danger"
                        type="button"
                        title="Usuń z Ollamy i z dysku"
                        disabled={busy}
                        onClick={() => {
                          if (
                            window.confirm(
                              `Usunąć ${entry.model_name} z Ollamy i plik GGUF z dysku?`,
                            )
                          )
                            void run(() => api.deleteExport(entry.id));
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                </div>
                {entry.state === "running" && (
                  <>
                    <div className="progress my-2" style={{ height: 6 }}>
                      <div
                        className="progress-bar progress-bar-striped progress-bar-animated"
                        style={{
                          width: `${((stageIndex + 0.5) / Math.max(stages.length, 1)) * 100}%`,
                        }}
                      />
                    </div>
                    <div className="small text-secondary">
                      {EXPORT_STAGE_LABELS[entry.stage] ?? entry.stage}
                      {entry.stage_started_at
                        ? ` · ${formatDuration(now - entry.stage_started_at)}`
                        : ""}
                      {" · łącznie "}
                      {formatDuration(now - entry.started_at)}
                    </div>
                  </>
                )}
                {entry.state === "ready" &&
                  Object.keys(entry.stages).length > 0 && (
                    <div className="small text-secondary">
                      {stages
                        .filter((stage) => entry.stages[stage])
                        .map(
                          (stage) =>
                            `${stage} ${formatDuration(entry.stages[stage].seconds)}`,
                        )
                        .join(" · ")}
                    </div>
                  )}
                {entry.state === "failed" && (
                  <details className="small mt-1">
                    <summary className="text-danger">{entry.error}</summary>
                    <pre className="transform-sample mt-1">
                      {entry.log_tail}
                    </pre>
                  </details>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function DeploymentPanel({
  adapters,
  best,
  status,
  gpuBusy,
  onStatus,
  exportsStatus,
  trainingRunning,
  onExports,
}: {
  adapters: Record<string, string[]>;
  best: Record<string, { checkpoint: string; eval_loss: number }>;
  status: ServingStatus | null;
  gpuBusy: boolean;
  onStatus: (status: ServingStatus) => void;
  exportsStatus: ExportsStatus | null;
  trainingRunning: boolean;
  onExports: (status: ExportsStatus) => void;
}) {
  const [adapter, setAdapter] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const adapterNames = Object.keys(adapters);
  const selectedAdapter = adapter || adapterNames[0] || "";
  const checkpoints = (adapters[selectedAdapter] ?? []).filter(
    (item) => !item.startsWith("merged-"),
  );
  const bestCheckpoint = best[selectedAdapter];
  const selectedCheckpoint =
    checkpoint && checkpoints.includes(checkpoint)
      ? checkpoint
      : bestCheckpoint && checkpoints.includes(bestCheckpoint.checkpoint)
        ? bestCheckpoint.checkpoint
        : (checkpoints.find((item) => item !== "base") ?? "");
  const active = status?.state === "ready" || status?.state === "loading";
  const isSelectedDeployed =
    active &&
    status?.adapter_name === selectedAdapter &&
    status?.checkpoint === selectedCheckpoint;
  const run = async (action: () => Promise<ServingStatus>) => {
    setBusy(true);
    setError("");
    try {
      onStatus(await action());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : String(requestError),
      );
    } finally {
      setBusy(false);
    }
  };
  const deploy = () => {
    if (
      active &&
      !window.confirm(
        `Zastąpić wdrożony ${status?.model} checkpointem ${selectedAdapter}/${selectedCheckpoint}?`,
      )
    )
      return;
    void run(() => api.deployCheckpoint(selectedAdapter, selectedCheckpoint));
  };
  const badge = {
    ready: ["text-bg-success", "Gotowy w czacie"],
    loading: ["text-bg-info", "Ładowanie modelu…"],
    failed: ["text-bg-danger", "Błąd wdrożenia"],
    unavailable: ["text-bg-secondary", "Docker niedostępny"],
    idle: ["text-bg-secondary", "Nic nie jest wdrożone"],
  }[status?.state ?? "idle"];
  return (
    <div className="card shadow-sm border-0 mt-3">
      <div className="card-body">
        <div className="d-flex flex-wrap justify-content-between align-items-start gap-2">
          <div>
            <h2 className="h5 mb-1">Wdrożenie do czatu</h2>
            <p className="text-secondary small mb-0">
              Checkpoint z adapterem LoRA (4-bit NF4, jak w ewaluacji) pojawi
              się w czacie Lokalny Bielik. Zajmuje GPU do czasu zatrzymania.
            </p>
          </div>
          <span className={`badge ${badge[0]}`}>{badge[1]}</span>
        </div>
        {active && status?.model && (
          <p className="small mt-2 mb-0">
            Wdrożony: <code>{status.model}</code>
            {status.state === "ready" && (
              <>
                {" · "}
                <NavLink to="/chat">otwórz czat</NavLink>
              </>
            )}
          </p>
        )}
        <div className="row g-2 align-items-end mt-1">
          <div className="col-12 col-md-5">
            <label className="form-label small mb-1" htmlFor="deploy-adapter">
              Adapter
            </label>
            <select
              id="deploy-adapter"
              className="form-select"
              value={selectedAdapter}
              onChange={(event) => {
                setAdapter(event.target.value);
                setCheckpoint("");
              }}
            >
              {adapterNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div className="col-12 col-md-4">
            <label
              className="form-label small mb-1"
              htmlFor="deploy-checkpoint"
            >
              Checkpoint
            </label>
            <select
              id="deploy-checkpoint"
              className="form-select"
              value={selectedCheckpoint}
              onChange={(event) => setCheckpoint(event.target.value)}
            >
              {checkpoints.map((item) => (
                <option key={item} value={item}>
                  {checkpointLabel(item)}
                  {bestCheckpoint?.checkpoint === item
                    ? ` ★ najlepszy (eval_loss ${bestCheckpoint.eval_loss.toFixed(4)})`
                    : ""}
                  {active &&
                  status?.adapter_name === selectedAdapter &&
                  status?.checkpoint === item
                    ? " — wdrożony"
                    : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="col-12 col-md-3 d-flex gap-2">
            <button
              className="btn btn-primary flex-grow-1"
              type="button"
              onClick={deploy}
              disabled={
                busy || gpuBusy || !selectedCheckpoint || isSelectedDeployed
              }
            >
              <Play size={16} className="me-1" /> Deploy checkpoint
            </button>
            <button
              className="btn btn-outline-danger"
              type="button"
              title="Zatrzymaj wdrożenie"
              onClick={() => void run(api.stopServing)}
              disabled={busy || !status || status.state === "idle"}
            >
              <Square size={16} />
            </button>
          </div>
        </div>
        {gpuBusy && (
          <p className="text-secondary small mt-2 mb-0">
            GPU zajęte przez trening lub ewaluację — wdrożenie będzie możliwe po
            ich zakończeniu.
          </p>
        )}
        {status?.state === "failed" && (
          <p className="text-danger small mt-2 mb-0">
            Kontener wdrożenia zakończył się (kod {status.exit_code}). Szczegóły
            w logach po prawej.
          </p>
        )}
        {error && <p className="text-danger small mt-2 mb-0">{error}</p>}
        <ExportSection
          adapter={selectedAdapter}
          checkpoint={selectedCheckpoint}
          exportsStatus={exportsStatus}
          trainingRunning={trainingRunning}
          onExports={onExports}
        />
      </div>
    </div>
  );
}

function MetricBars({
  title,
  items,
}: {
  title: string;
  items: Array<{ label: string; value: number | null; color: string }>;
}) {
  return (
    <div className="training-chart">
      <h3 className="h6 mb-2">{title}</h3>
      {items.length === 0 && (
        <p className="text-secondary small mb-0">Brak danych.</p>
      )}
      {items.map((item) => (
        <div className="metric-bar" key={item.label}>
          <span className="metric-bar-label" title={item.label}>
            {item.label}
          </span>
          <span className="metric-bar-track">
            <span
              className="metric-bar-fill"
              style={{
                width: `${(item.value ?? 0) * 100}%`,
                background: item.color,
              }}
            />
          </span>
          <span className="metric-bar-value">
            {item.value == null ? "-" : `${(item.value * 100).toFixed(1)}%`}
          </span>
        </div>
      ))}
    </div>
  );
}

function CheckpointCharts({ result }: { result: EvaluationCheckpointResult }) {
  const { summary, curve } = result;
  const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
  const curveSeries = (
    key: keyof EvaluationCurvePoint,
    label: string,
    color: string,
  ) => ({
    label,
    color,
    points: curve
      .filter((point) => typeof point[key] === "number")
      .map((point) => ({ x: point.n, y: point[key] as number })),
  });
  const perType = Object.entries(
    summary?.per_type_relaxed ?? summary?.per_type ?? {},
  );
  const progress =
    result.total && result.total > 0 ? (result.done / result.total) * 100 : 0;
  return (
    <div className="card shadow-sm border-0 mt-3">
      <div className="card-body">
        <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
          <h2 className="h5 mb-0">{checkpointLabel(result.checkpoint)}</h2>
          <span
            className={`badge ${result.finished ? "text-bg-success" : result.done ? "text-bg-info" : "text-bg-secondary"}`}
          >
            {result.finished
              ? "Zakończony"
              : result.done
                ? "W toku"
                : "Oczekuje"}{" "}
            · {result.done}/{result.total ?? "?"}
          </span>
        </div>
        <div className="progress my-2" style={{ height: 6 }}>
          <div
            className={`progress-bar ${result.finished ? "" : "progress-bar-striped progress-bar-animated"}`}
            style={{ width: `${progress}%` }}
          />
        </div>
        {!summary ? (
          <p className="text-secondary small mb-0">
            Wykresy pojawią się po pierwszych przykładach.
          </p>
        ) : (
          <>
            <dl className="training-stats mb-2">
              <dt>Tolerancyjne F1</dt>
              <dd>
                {summary.relaxed.f1 == null ? "-" : percent(summary.relaxed.f1)}
              </dd>
              <dt>Ścisłe F1 (z pozycjami)</dt>
              <dd>
                {summary.strict.f1 == null ? "-" : percent(summary.strict.f1)}
              </dd>
              <dt>TP / FP / FN</dt>
              <dd>
                {summary.relaxed.tp} / {summary.relaxed.fp} /{" "}
                {summary.relaxed.fn}
              </dd>
              <dt>Negatywne poprawnie</dt>
              <dd>
                {summary.negatives.correct}/{summary.negatives.examples}
              </dd>
            </dl>
            <div className="row g-3">
              <div className="col-12 col-xxl-6">
                <LineChart
                  title="Narastająco po przykładach"
                  series={[
                    curveSeries("relaxed_f1", "F1 tolerancyjne", "#176b61"),
                    curveSeries("strict_f1", "F1 ścisłe", "#c0503a"),
                    curveSeries("json_valid", "Poprawny JSON", "#8e5bb5"),
                  ]}
                  format={(value) => `${(value * 100).toFixed(0)}%`}
                  xLabel={(value) => `przykład ${value}`}
                  height={200}
                  width={560}
                />
              </div>
              <div className="col-12 col-xxl-6">
                <MetricBars
                  title="Precision / Recall / F1"
                  items={[
                    {
                      label: "Tol. precision",
                      value: summary.relaxed.precision,
                      color: "#3d6fb6",
                    },
                    {
                      label: "Tol. recall",
                      value: summary.relaxed.recall,
                      color: "#3d6fb6",
                    },
                    {
                      label: "Tol. F1",
                      value: summary.relaxed.f1,
                      color: "#176b61",
                    },
                    {
                      label: "Ścisłe precision",
                      value: summary.strict.precision,
                      color: "#c98a3a",
                    },
                    {
                      label: "Ścisłe recall",
                      value: summary.strict.recall,
                      color: "#c98a3a",
                    },
                    {
                      label: "Ścisłe F1",
                      value: summary.strict.f1,
                      color: "#c0503a",
                    },
                    {
                      label: "Poprawny JSON",
                      value: summary.json_valid,
                      color: "#8e5bb5",
                    },
                    {
                      label: "Komplet encji",
                      value: summary.exact_match,
                      color: "#8e5bb5",
                    },
                  ]}
                />
              </div>
              <div className="col-12">
                <MetricBars
                  title="F1 według typu encji (tolerancyjne)"
                  items={perType.map(([type, score]) => ({
                    label: `${type} (${score.tp + score.fn})`,
                    value: score.f1,
                    color: "#176b61",
                  }))}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function EvaluationComparison({
  comparison,
}: {
  comparison: Array<EvaluationSummary & { checkpoint: string }>;
}) {
  if (comparison.length < 2) return null;
  const pointLabel = (position: number) =>
    checkpointLabel(comparison[position - 1]?.checkpoint ?? "checkpoint");
  return (
    <div className="row g-3 mt-1">
      <div className="col-12 col-xl-6">
        <LineChart
          title="F1 checkpointów"
          series={[
            {
              label: "Ścisła F1",
              color: "#176b61",
              points: comparison.map((result, index) => ({
                x: index + 1,
                y: result.strict.f1 ?? 0,
              })),
            },
            {
              label: "Tolerancyjna F1",
              color: "#3d6fb6",
              points: comparison.map((result, index) => ({
                x: index + 1,
                y: result.relaxed.f1 ?? 0,
              })),
            },
          ]}
          format={(value) => `${(value * 100).toFixed(1)}%`}
          xLabel={pointLabel}
          height={210}
          width={580}
        />
      </div>
      <div className="col-12 col-xl-6">
        <LineChart
          title="Format odpowiedzi"
          series={[
            {
              label: "Poprawny JSON",
              color: "#8e5bb5",
              points: comparison.map((result, index) => ({
                x: index + 1,
                y: result.json_valid ?? 0,
              })),
            },
            {
              label: "Komplet encji",
              color: "#c0503a",
              points: comparison.map((result, index) => ({
                x: index + 1,
                y: result.exact_match ?? 0,
              })),
            },
          ]}
          format={(value) => `${(value * 100).toFixed(1)}%`}
          xLabel={pointLabel}
          height={210}
          width={580}
        />
      </div>
    </div>
  );
}

function Training({ evaluationOnly = false }: { evaluationOnly?: boolean }) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [evaluationStatus, setEvaluationStatus] =
    useState<EvaluationStatus | null>(null);
  const [evaluationAdapters, setEvaluationAdapters] = useState<
    Record<string, string[]>
  >({});
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [baseModels, setBaseModels] = useState<string[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [adapterName, setAdapterName] = useState("bielik-qlora-v1");
  const [evaluationAdapter, setEvaluationAdapter] = useState("");
  const [evaluationCheckpoints, setEvaluationCheckpoints] = useState<string[]>(
    [],
  );
  const [evaluationSplits, setEvaluationSplits] = useState<Array<Split>>([
    "test",
  ]);
  const trainingPageRef = useRef<HTMLElement>(null);
  const [logsWidth, setLogsWidth] = useState(
    () => Number(localStorage.getItem("training-logs-width")) || 640,
  );
  const resizeLogs = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = trainingPageRef.current?.getBoundingClientRect();
    if (!bounds || !event.currentTarget.hasPointerCapture(event.pointerId))
      return;
    const width = Math.round(
      Math.min(Math.max(bounds.right - event.clientX, 280), bounds.width - 420),
    );
    setLogsWidth(width);
    localStorage.setItem("training-logs-width", String(width));
  };
  const [busy, setBusy] = useState(false);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [error, setError] = useState("");
  const [evaluationError, setEvaluationError] = useState("");
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [runId, setRunId] = useState("");
  const [viewedRun, setViewedRun] = useState<TrainingStatus | null>(null);
  useEffect(() => {
    if (!runId) {
      setViewedRun(null);
      return;
    }
    void api.trainingRun(runId).then(setViewedRun);
  }, [runId]);
  useEffect(() => {
    void api.trainingRuns().then(setRuns);
  }, [status?.state]);
  const [servingStatus, setServingStatus] = useState<ServingStatus | null>(
    null,
  );
  const [exportsStatus, setExportsStatus] = useState<ExportsStatus | null>(
    null,
  );
  const [bestCheckpoints, setBestCheckpoints] = useState<
    Record<string, { checkpoint: string; eval_loss: number }>
  >({});
  const refresh = () =>
    Promise.all([
      api.trainingStatus(corpusId),
      api.evaluationStatus(),
      api.servingStatus(),
      evaluationOnly ? api.exports() : Promise.resolve(null),
    ])
      .then(([training, evaluation, serving, exportsResult]) => {
        setStatus(training);
        setEvaluationStatus(evaluation);
        setServingStatus(serving);
        if (exportsResult) setExportsStatus(exportsResult);
      })
      .catch((requestError: Error) => setError(requestError.message));
  const readyExports = (exportsStatus?.exports ?? [])
    .filter((entry) => entry.state === "ready")
    .map((entry) => entry.id)
    .join(",");
  useEffect(() => {
    // A finished export adds a "merged-…" checkpoint to the evaluation list.
    if (readyExports)
      void api.evaluationAdapters().then(({ adapters, best }) => {
        setEvaluationAdapters(adapters);
        setBestCheckpoints(best);
      });
  }, [readyExports]);

  useEffect(() => {
    void api.corpora().then((items) => {
      setCorpora(items);
      setCorpusId((current) => current || items[0]?.id || "");
    });
    void api.trainingModels().then(({ models }) => {
      setBaseModels(models);
      setBaseModel((current) => current || models[0] || "");
    });
    void api.evaluationAdapters().then(({ adapters, best }) => {
      setEvaluationAdapters(adapters);
      setBestCheckpoints(best);
      const firstAdapter = Object.keys(adapters)[0] || "";
      setEvaluationAdapter(firstAdapter);
      setEvaluationCheckpoints(adapters[firstAdapter]?.slice(0, 2) || []);
    });
  }, []);
  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(interval);
  }, [corpusId]);

  const [freedNotice, setFreedNotice] = useState("");
  const start = async () => {
    setBusy(true);
    setError("");
    setFreedNotice("");
    try {
      const started = await api.startTraining({
        corpusId,
        baseModel,
        adapterName,
      });
      setStatus(started);
      const freed = [
        ...(started.freed?.serving_stopped
          ? ["zatrzymano wdrożony checkpoint"]
          : []),
        ...(started.freed?.ollama_unloaded.length
          ? [`zwolniono z Ollamy: ${started.freed.ollama_unloaded.join(", ")}`]
          : []),
      ];
      if (freed.length) setFreedNotice(`Zwolniono VRAM: ${freed.join("; ")}.`);
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

  const selectEvaluationAdapter = (nextAdapter: string) => {
    setEvaluationAdapter(nextAdapter);
    setEvaluationCheckpoints(
      evaluationAdapters[nextAdapter]?.slice(0, 2) || [],
    );
  };

  const toggleCheckpoint = (checkpoint: string) => {
    setEvaluationCheckpoints((current) =>
      current.includes(checkpoint)
        ? current.filter((item) => item !== checkpoint)
        : [...current, checkpoint],
    );
  };

  const toggleEvaluationSplit = (split: Split) => {
    setEvaluationSplits((current) =>
      current.includes(split)
        ? current.filter((item) => item !== split)
        : [...current, split],
    );
  };

  const startEvaluation = async () => {
    setEvaluationBusy(true);
    setEvaluationError("");
    try {
      setEvaluationStatus(
        await api.startEvaluation(
          corpusId,
          evaluationAdapter,
          evaluationCheckpoints,
          evaluationSplits,
        ),
      );
    } catch (requestError) {
      setEvaluationError(
        requestError instanceof Error
          ? requestError.message
          : "Nie udało się uruchomić ewaluacji.",
      );
    } finally {
      setEvaluationBusy(false);
    }
  };

  const stopEvaluation = async () => {
    if (
      !window.confirm(
        "Zatrzymać ewaluację? Niezakończone checkpointy trzeba będzie liczyć od nowa.",
      )
    )
      return;
    setEvaluationBusy(true);
    setEvaluationError("");
    try {
      setEvaluationStatus(await api.stopEvaluation());
    } catch (requestError) {
      setEvaluationError(
        requestError instanceof Error
          ? requestError.message
          : "Nie udało się zatrzymać ewaluacji.",
      );
    } finally {
      setEvaluationBusy(false);
    }
  };

  const splits = status?.splits;
  const isRunning = status?.state === "running";
  const isEvaluationRunning = evaluationStatus?.state === "running";
  const checkpointEvals = (status?.metrics ?? [])
    .filter((metric) => typeof metric.eval_loss === "number")
    .map((metric) => ({
      step: metric.step,
      evalLoss: metric.eval_loss as number,
      accuracy: metric.eval_mean_token_accuracy,
    }));
  const bestCheckpoint = checkpointEvals.reduce<
    (typeof checkpointEvals)[number] | undefined
  >(
    (best, item) => (!best || item.evalLoss < best.evalLoss ? item : best),
    undefined,
  );
  // Live, rescored results; comparison.json keeps whatever metric version wrote it.
  const comparison = (evaluationStatus?.checkpoints ?? []).flatMap((result) =>
    result.finished && result.summary
      ? [{ ...result.summary, checkpoint: result.checkpoint }]
      : [],
  );
  const summary = comparison.at(-1) ?? evaluationStatus?.summary;
  const evaluationStopped =
    evaluationStatus?.state === "exited" &&
    // 143 = SIGTERM, e.g. `docker stop` outside the UI.
    (Boolean(evaluationStatus.stopped) || evaluationStatus.exit_code === 143);
  const evaluationFailed =
    evaluationStatus?.state === "exited" &&
    Boolean(evaluationStatus.exit_code) &&
    !evaluationStopped;
  const evaluationLabel = isEvaluationRunning
    ? "Ewaluacja w toku"
    : evaluationStopped
      ? "Zatrzymana ręcznie"
      : evaluationFailed
        ? "Ewaluacja zakończona błędem"
        : comparison.length
          ? "Wyniki gotowe"
          : "Gotowa do uruchomienia";
  const formatPercent = (value: number | null | undefined) =>
    value == null ? "-" : `${(value * 100).toFixed(1)}%`;
  return (
    <section
      className={`training-page ${evaluationOnly ? "validation-page" : ""}`}
      ref={trainingPageRef}
      style={{ gridTemplateColumns: `minmax(0, 1fr) 6px ${logsWidth}px` }}
    >
      <div className="training-main">
        <p className="section-kicker">
          {evaluationOnly ? "EWALUACJA" : "QLORA"}
        </p>
        <h1>
          {evaluationOnly ? "Ewaluacja checkpointów" : "Uczenie i status"}
        </h1>
        {evaluationOnly && (
          <p className="text-secondary mb-3">
            Porównaj zapisane checkpointy na oznaczonych przykładach bez
            uruchamiania treningu.
          </p>
        )}
        {evaluationOnly && (
          <DeploymentPanel
            adapters={evaluationAdapters}
            best={bestCheckpoints}
            status={servingStatus}
            gpuBusy={isRunning || isEvaluationRunning}
            onStatus={setServingStatus}
            exportsStatus={exportsStatus}
            trainingRunning={isRunning}
            onExports={setExportsStatus}
          />
        )}
        {!evaluationOnly && (
          <div className="card shadow-sm border-0 mt-3">
            <div className="card-body">
              <div className="row g-3">
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
                  {corpusId && (
                    <div className="form-text">
                      {(["train", "validation", "test"] as Split[])
                        .map((split) => `${split}: ${splits?.[split] ?? "-"}`)
                        .join(" · ")}
                    </div>
                  )}
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
                  <Play size={16} className="me-1" /> Rozpocznij uczenie
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
              {freedNotice && (
                <p className="text-secondary small mt-3 mb-0">{freedNotice}</p>
              )}
              <h2 className="h5 mt-4">Profil QLoRA</h2>
              <code>{status?.profile ?? "profil QLoRA"}</code>
              <p className="text-secondary small mt-3 mb-0">
                Uczony jest nowy adapter LoRA, nie model bazowy. Eksport
                obejmuje tylko wybrany korpus i jego splity <code>train</code>{" "}
                oraz <code>validation</code>.
              </p>
              {status?.job?.adapter_name && (
                <p className="text-secondary small mt-2 mb-0">
                  Ostatnie zadanie: <code>{status.job.base_model}</code> do
                  adaptera <code>{status.job.adapter_name}</code>.
                </p>
              )}
            </div>
          </div>
        )}
        {evaluationOnly && (
          <div className="card shadow-sm border-0 mt-3">
            <div className="card-body">
              <div className="d-flex flex-wrap justify-content-between align-items-start gap-2">
                <div>
                  <h2 className="h5 mb-1">Porównanie checkpointów</h2>
                  <p className="text-secondary small mb-0">
                    Generowanie na wybranym splicie bez zmiany adaptera.
                  </p>
                </div>
                <span
                  className={`badge ${isEvaluationRunning ? "text-bg-info" : "text-bg-secondary"}`}
                >
                  {evaluationLabel}
                </span>
              </div>
              {checkpointEvals.length > 0 && (
                <div className="row g-3 mt-1">
                  <div className="col-12 col-xl-7">
                    <LineChart
                      title="Eval loss checkpointów"
                      series={[
                        {
                          label: "eval_loss",
                          color: "#c0503a",
                          points: checkpointEvals.map((item) => ({
                            x: item.step,
                            y: item.evalLoss,
                          })),
                        },
                      ]}
                      format={(value) => value.toFixed(4)}
                      height={210}
                      width={580}
                    />
                  </div>
                  <div className="col-12 col-xl-5">
                    <table className="table table-sm align-middle mb-0">
                      <thead>
                        <tr>
                          <th>Krok</th>
                          <th>eval_loss</th>
                          <th>Token acc.</th>
                        </tr>
                      </thead>
                      <tbody>
                        {checkpointEvals.map((item) => (
                          <tr
                            key={item.step}
                            className={
                              item.step === bestCheckpoint?.step
                                ? "table-success fw-semibold"
                                : ""
                            }
                          >
                            <td>
                              checkpoint-{item.step}
                              {item.step === bestCheckpoint?.step &&
                                " (najlepszy)"}
                            </td>
                            <td>{item.evalLoss.toFixed(4)}</td>
                            <td>{formatPercent(item.accuracy)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              <div className="row g-3 mt-1">
                <div className="col-12">
                  <label className="form-label" htmlFor="evaluation-corpus">
                    Korpus
                  </label>
                  <select
                    id="evaluation-corpus"
                    className="form-select"
                    value={corpusId}
                    onChange={(event) => setCorpusId(event.target.value)}
                    disabled={isEvaluationRunning}
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
                  <label className="form-label" htmlFor="evaluation-adapter">
                    Adapter
                  </label>
                  <select
                    id="evaluation-adapter"
                    className="form-select"
                    value={evaluationAdapter}
                    onChange={(event) =>
                      selectEvaluationAdapter(event.target.value)
                    }
                    disabled={isRunning || isEvaluationRunning}
                  >
                    <option value="">Wybierz adapter</option>
                    {Object.keys(evaluationAdapters).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="col-12 col-md-6">
                  <span className="form-label d-block mb-1">
                    Checkpointy do porównania
                  </span>
                  {(evaluationAdapters[evaluationAdapter] || []).map(
                    (checkpoint) => (
                      <div className="form-check" key={checkpoint}>
                        <input
                          id={`evaluation-checkpoint-${checkpoint}`}
                          className="form-check-input"
                          type="checkbox"
                          checked={evaluationCheckpoints.includes(checkpoint)}
                          onChange={() => toggleCheckpoint(checkpoint)}
                          disabled={isRunning || isEvaluationRunning}
                        />
                        <label
                          className="form-check-label"
                          htmlFor={`evaluation-checkpoint-${checkpoint}`}
                        >
                          {checkpointLabel(checkpoint)}
                          {servingStatus?.adapter_name === evaluationAdapter &&
                            servingStatus.checkpoint === checkpoint &&
                            ["ready", "loading"].includes(
                              servingStatus.state,
                            ) && (
                              <span className="badge text-bg-success ms-2">
                                wdrożony
                              </span>
                            )}
                        </label>
                      </div>
                    ),
                  )}
                </div>
                <div className="col-12">
                  <span className="form-label d-block mb-1">Dane do oceny</span>
                  {(["train", "validation", "test"] as const).map((split) => (
                    <div className="form-check form-check-inline" key={split}>
                      <input
                        id={`evaluation-${split}`}
                        className="form-check-input"
                        type="checkbox"
                        checked={evaluationSplits.includes(split)}
                        onChange={() => toggleEvaluationSplit(split)}
                        disabled={isRunning || isEvaluationRunning}
                      />
                      <label
                        className="form-check-label"
                        htmlFor={`evaluation-${split}`}
                      >
                        {split} ({splits?.[split] ?? 0})
                      </label>
                    </div>
                  ))}
                </div>
              </div>
              <div className="d-flex flex-wrap gap-2 mt-3">
                <button
                  className="btn btn-primary"
                  onClick={() => void startEvaluation()}
                  disabled={
                    evaluationBusy ||
                    isRunning ||
                    isEvaluationRunning ||
                    !corpusId ||
                    !evaluationAdapter ||
                    evaluationCheckpoints.length === 0 ||
                    evaluationSplits.length === 0
                  }
                >
                  <Play size={16} className="me-1" /> Uruchom ewaluację
                </button>
                <button
                  className="btn btn-outline-danger"
                  onClick={() => void stopEvaluation()}
                  disabled={evaluationBusy || !isEvaluationRunning}
                >
                  <Square size={16} className="me-1" /> Zatrzymaj
                </button>
              </div>
              {evaluationStatus?.progress && (
                <p className="text-secondary small mt-3 mb-0">
                  Postęp
                  {evaluationStatus.progress.checkpoint &&
                    ` (${evaluationStatus.progress.checkpoint})`}
                  : {evaluationStatus.progress.done} /{" "}
                  {evaluationStatus.progress.total}
                  {evaluationStatus.progress.elapsed > 0 &&
                    ` · ${Math.round(evaluationStatus.progress.elapsed)} s`}
                </p>
              )}
              {evaluationError && (
                <p className="text-danger small mt-3 mb-0">{evaluationError}</p>
              )}
              {evaluationStopped && (
                <p className="text-secondary small mt-3 mb-0">
                  Ewaluacja zatrzymana przyciskiem Zatrzymaj.
                </p>
              )}
              {evaluationFailed && (
                <p className="text-danger small mt-3 mb-0">
                  Ewaluacja przerwana (kod wyjścia {evaluationStatus?.exit_code}
                  ). Szczegóły w logach po prawej.
                </p>
              )}
              {!isEvaluationRunning &&
                !evaluationFailed &&
                !summary &&
                comparison.length === 0 && (
                  <p className="text-secondary small mt-3 mb-0">
                    Brak wyników porównania. Zaznacz checkpointy i uruchom
                    ewaluację, aby zobaczyć wykresy oraz tabelę wyników.
                  </p>
                )}
              {summary && (
                <div className="table-responsive mt-3">
                  <table className="table table-sm align-middle mb-0">
                    <thead>
                      <tr>
                        <th>Próba</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Ścisła</th>
                        <td>{formatPercent(summary.strict.precision)}</td>
                        <td>{formatPercent(summary.strict.recall)}</td>
                        <td>{formatPercent(summary.strict.f1)}</td>
                      </tr>
                      <tr>
                        <th>Tolerancyjna</th>
                        <td>{formatPercent(summary.relaxed.precision)}</td>
                        <td>{formatPercent(summary.relaxed.recall)}</td>
                        <td>{formatPercent(summary.relaxed.f1)}</td>
                      </tr>
                      <tr>
                        <th>Format odpowiedzi</th>
                        <td>JSON: {formatPercent(summary.json_valid)}</td>
                        <td>
                          Komplet encji: {formatPercent(summary.exact_match)}
                        </td>
                        <td>
                          Negatywne: {summary.negatives.correct}/
                          {summary.negatives.examples}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
              <EvaluationComparison comparison={comparison} />
              {comparison.length > 0 && (
                <div className="table-responsive mt-3">
                  <table className="table table-sm align-middle mb-0">
                    <thead>
                      <tr>
                        <th>Checkpoint</th>
                        <th>Ścisła F1</th>
                        <th>Tolerancyjna F1</th>
                        <th>JSON</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.map((result) => (
                        <tr key={result.checkpoint}>
                          <th>{checkpointLabel(result.checkpoint)}</th>
                          <td>{formatPercent(result.strict.f1)}</td>
                          <td>{formatPercent(result.relaxed.f1)}</td>
                          <td>{formatPercent(result.json_valid)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
        {evaluationOnly &&
          evaluationStatus?.checkpoints?.map((result) => (
            <CheckpointCharts key={result.checkpoint} result={result} />
          ))}
        {!evaluationOnly && (
          <div className="d-flex flex-wrap align-items-center gap-2 mt-3">
            <label className="form-label mb-0" htmlFor="training-run">
              Run
            </label>
            <select
              id="training-run"
              className="form-select w-auto"
              value={runId}
              onChange={(event) => setRunId(event.target.value)}
            >
              <option value="">Bieżący ({status?.state ?? "…"})</option>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.started_at
                    ? new Date(run.started_at).toLocaleString("pl-PL")
                    : run.run_id}{" "}
                  · {run.adapter_name} · {run.steps} kroków
                  {run.best_eval_loss != null &&
                    ` · min eval_loss ${run.best_eval_loss.toFixed(4)}`}
                  {run.exit_code ? ` · kod ${run.exit_code}` : ""}
                </option>
              ))}
            </select>
            {viewedRun && (
              <span className="badge text-bg-secondary">
                Archiwum — {viewedRun.job?.adapter_name}
              </span>
            )}
          </div>
        )}
        {!evaluationOnly && <TrainingDashboard status={viewedRun ?? status} />}
      </div>
      <div
        className="training-splitter"
        role="separator"
        aria-orientation="vertical"
        aria-label="Zmień szerokość panelu logów"
        onPointerDown={(event) =>
          event.currentTarget.setPointerCapture(event.pointerId)
        }
        onPointerMove={resizeLogs}
        onDoubleClick={() => {
          setLogsWidth(640);
          localStorage.removeItem("training-logs-width");
        }}
      />
      <div className="training-logs-column">
        <TrainingLogView
          logs={
            (evaluationOnly
              ? !isEvaluationRunning &&
                exportsStatus?.exports.some(
                  (entry) => entry.state === "running",
                )
                ? exportsStatus.logs
                : !isEvaluationRunning &&
                    servingStatus &&
                    ["loading", "ready", "failed"].includes(servingStatus.state)
                  ? servingStatus.logs
                  : evaluationStatus?.logs
              : (viewedRun ?? status)?.logs) || "Brak logów zadania."
          }
        />
      </div>
    </section>
  );
}

function Workspace() {
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const pathname = useLocation().pathname;
  const showSidebar = !["/training", "/evaluation"].some((path) =>
    pathname.startsWith(path),
  );
  const refresh = () => api.corpora().then(setCorpora);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem("corpus-sidebar-collapsed") === "1",
  );
  const toggleSidebar = () =>
    setSidebarCollapsed((current) => {
      localStorage.setItem("corpus-sidebar-collapsed", current ? "0" : "1");
      return !current;
    });
  useEffect(() => {
    void refresh();
  }, []);
  return (
    <main
      className={`workspace-shell ${showSidebar ? (sidebarCollapsed ? "sidebar-collapsed" : "") : "no-sidebar"}`}
    >
      <header className="workspace-header">
        <Bot size={21} /> Bielik LoRA Lab{" "}
        <nav className="workspace-header-nav">
          <NavLink to="/corpora">Korpusy</NavLink>
          <NavLink to="/chat">Lokalny Bielik</NavLink>
          <NavLink to="/training">Uczenie</NavLink>
          <NavLink to="/evaluation">Ewaluacja</NavLink>
        </nav>
        <button
          className="btn btn-sm btn-outline-light"
          onClick={() => void refresh()}
        >
          <RefreshCw size={15} />
        </button>
      </header>
      {showSidebar && sidebarCollapsed && (
        <aside className="corpus-sidebar collapsed">
          <button
            className="btn btn-sm btn-link corpus-rail-button"
            type="button"
            title="Rozwiń listę korpusów"
            aria-label="Rozwiń listę korpusów"
            onClick={toggleSidebar}
          >
            <PanelLeftOpen size={18} />
          </button>
          <NavLink
            className="btn btn-sm btn-primary corpus-rail-button"
            to="/corpora/new"
            title="Nowy korpus"
            aria-label="Nowy korpus"
          >
            <FilePlus2 size={16} />
          </NavLink>
          <nav>
            {corpora.map((corpus) => (
              <NavLink
                className="corpus-rail-link"
                key={corpus.id}
                to={`/corpora/${corpus.id}`}
                title={`${corpus.name} (${corpus.example_count})`}
                aria-label={corpus.name}
              >
                {corpus.name
                  .replace(/[^\p{L}\p{N}]/gu, "")
                  .slice(0, 2)
                  .toUpperCase()}
              </NavLink>
            ))}
          </nav>
        </aside>
      )}
      {showSidebar && !sidebarCollapsed && (
        <aside className="corpus-sidebar">
          <div className="sidebar-title">
            <Database size={16} /> KORPUSY
            <button
              className="btn btn-sm btn-link ms-auto p-0 text-secondary"
              type="button"
              title="Zwiń listę korpusów"
              aria-label="Zwiń listę korpusów"
              onClick={toggleSidebar}
            >
              <PanelLeftClose size={16} />
            </button>
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
      )}
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
            path="/validation"
            element={<Navigate to="/evaluation" replace />}
          />
          <Route
            path="/builder/:corpusId?"
            element={<Builder corpora={corpora} />}
          />
          <Route path="/chat" element={<Chat />} />
          <Route path="/training" element={<Training />} />
          <Route path="/evaluation" element={<Training evaluationOnly />} />
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
