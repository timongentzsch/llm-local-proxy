//! The intermediate representation shared by every dialect and provider.
//!
//! A downstream request is parsed once into [`ChatRequest`]; each provider
//! renders its own upstream body from that. Without it the proxy would need
//! one converter per (dialect, provider) pair.
//!
//! Common semantics use typed fields. Content without a lossless mapping uses
//! explicit opaque wire-format records; adapters must preserve or reject
//! them.
//!
//! Prompt caching never changes output, so every `cache` field is a hint:
//! `None` places no breakpoint, otherwise the block ends a cacheable prefix
//! kept for that TTL (`""` for the upstream's default). A provider honours
//! the hints its upstream can express and otherwise relies on the upstream's
//! automatic caching.

use crate::error::Result;
use crate::json::Object;
use serde_json::Value;
use std::collections::HashMap;

/// A cache hint: `None`, or a TTL (`""` for the upstream's default).
pub type Cache = Option<String>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Text {
    pub text: String,
    pub cache: Cache,
    /// Anthropic-format citations on replayed assistant text, for an
    /// upstream that verifies them; others read the text alone.
    pub citations: Option<Vec<Value>>,
}

impl Text {
    pub fn new(text: impl Into<String>) -> Self {
        Text {
            text: text.into(),
            cache: None,
            citations: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Image {
    pub url: String,
    pub cache: Cache,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolUse {
    /// May be empty; Chat Completions allows a call without one.
    pub id: String,
    pub name: String,
    /// As the client sent them: a JSON string, or an already parsed value.
    pub arguments: Value,
    /// The Responses namespace the called tool belongs to, if any.
    pub namespace: String,
    pub cache: Cache,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub text: String,
    pub is_error: bool,
    pub cache: Cache,
}

/// Signed reasoning; must round-trip byte-exactly or upstream rejects it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Thinking {
    pub text: String,
    pub signature: String,
    pub redacted: String,
}

/// Which wire format an opaque record was written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Anthropic,
    Responses,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Anthropic => "anthropic",
            Source::Responses => "responses",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Text(Text),
    Image(Image),
    ToolUse(ToolUse),
    ToolResult(ToolResult),
    Thinking(Thinking),
    /// Opaque Responses reasoning item carried verbatim between turns.
    Reasoning(Object),
    /// A Responses input/output item with no lossless cross-dialect mapping.
    NativeResponseItem(Object),
    /// An Anthropic-only content block retained verbatim for replay.
    NativeAnthropicBlock(Object),
    /// A finished provider-run search that a client echoes back in history.
    ///
    /// It replays verbatim to an upstream that speaks `source` (the
    /// pause_turn continuation Anthropic requires) and is omitted elsewhere:
    /// the search already ran, its answer is in the transcript, and the
    /// record was written for the client by this proxy.
    HostedSearch {
        item: Object,
        source: Source,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctionTool {
    pub name: String,
    pub parameters: Object,
    pub description: String,
    pub strict: Option<bool>,
    /// Extra fields belong to a wire format, not to a particular provider:
    /// the format `options` came from ("chat", "responses", "anthropic").
    pub source: String,
    pub options: Object,
    pub cache: Cache,
}

/// A search the provider runs, as its source format defined it.
///
/// `tools::responses_web_search` and `tools::anthropic_web_search` translate
/// it; options without an equivalent that change what is searched are
/// refused, hints the target cannot act on are not.
#[derive(Debug, Clone, PartialEq)]
pub struct WebSearchTool {
    pub native: Object,
    pub source: Source,
}

/// A member of a Responses namespace.
#[derive(Debug, Clone, PartialEq)]
pub enum NamespaceMember {
    Function(FunctionTool),
    /// A Responses tool definition retained without schema conversion.
    Native(Object),
}

/// A Responses namespace: tools grouped under one name.
///
/// `item` is the definition as sent, for targets that speak Responses;
/// others flatten `tools` with [`crate::tools::flatten`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolNamespace {
    pub name: String,
    pub tools: Vec<NamespaceMember>,
    pub item: Object,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tool {
    Function(FunctionTool),
    WebSearch(WebSearchTool),
    /// A Responses tool definition retained without schema conversion.
    Native(Object),
    Namespace(ToolNamespace),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChoiceKind {
    #[default]
    Auto,
    None,
    Required,
    Tool,
}

impl ChoiceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChoiceKind::Auto => "auto",
            ChoiceKind::None => "none",
            ChoiceKind::Required => "required",
            ChoiceKind::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolChoice {
    pub kind: ChoiceKind,
    pub name: String,
}

// --- response side ----------------------------------------------------------
// Finish reasons use Anthropic's seven-value vocabulary; the Chat Completions
// encoder narrows them to four.

/// One lifecycle step of a tool the *provider* runs, not the client.
///
/// Deliberately not a tool call: those oblige the client to execute something
/// and answer with a result, and a hosted search has already been executed
/// upstream. It is progress to show, never a tool round to take.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HostedToolEvent {
    pub tool: String,
    pub id: String,
    pub phase: String,
    /// What the provider searched for, when it said. Carried so an Anthropic
    /// client sees the `server_tool_use` input its upstream actually sent.
    pub query: String,
    /// Provider error code, when a hosted tool returned an error block.
    pub error_code: String,
    /// Native result payload when the provider exposes it for exact replay.
    pub result: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Citation {
    /// Empty for a citation into a document rather than a web page; formats
    /// that cite only URLs leave those out.
    pub url: String,
    /// As the upstream sent them: usually a string and two integers, Null
    /// when absent.
    pub title: Value,
    pub start_index: Value,
    pub end_index: Value,
    /// The Anthropic-format citation as issued. An upstream that verifies it
    /// on replay needs it whole, so an Anthropic client must receive it whole.
    pub native: Option<Object>,
    /// The text block it cites, as in [`StreamEvent::TextDelta`].
    pub span: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Usage {
    /// Total input including cache; Anthropic reports these apart.
    pub prompt: i64,
    pub completion: i64,
    pub total: Option<i64>,
    pub cache_read: i64,
    pub cache_write: i64,
    /// The part of cache_write kept for an hour; None when the upstream does
    /// not report cache TTLs.
    pub cache_write_1h: Option<i64>,
    pub thinking: i64,
    pub web_searches: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Finish {
    pub reason: String,
    pub incomplete_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

impl Default for Finish {
    fn default() -> Self {
        Finish {
            reason: "end_turn".into(),
            incomplete_reason: None,
            stop_sequence: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta {
        text: String,
        /// The upstream text block this belongs to, for formats that keep
        /// block boundaries (a cited passage is its own block); "" when
        /// there are none.
        span: String,
    },
    ThinkingDelta {
        text: String,
        /// The reasoning item this text belongs to, when the upstream names
        /// one, so an item-based client sees one id from `added` to `done`.
        item_id: String,
    },
    /// Closes a thinking block.
    ThinkingSignature {
        signature: String,
    },
    RedactedThinkingDelta {
        data: String,
    },
    /// A complete opaque reasoning item for stateless Responses replay.
    ReasoningItem {
        item: Object,
    },
    /// A complete native Responses output item.
    NativeItem {
        item: Object,
    },
    ToolCallStart {
        /// Stable within one response; providers number calls differently,
        /// and one may leave it out (Null).
        index: Value,
        id: String,
        name: String,
        arguments: String,
        namespace: String,
    },
    ToolCallArgs {
        index: Value,
        fragment: String,
    },
    /// The assembled call; carries no new bytes.
    ToolCallEnd {
        index: Value,
        id: String,
        name: String,
        arguments: String,
        namespace: String,
    },
    HostedTool(HostedToolEvent),
    Citation(Citation),
    Usage(Usage),
    Finish(Finish),
}

/// Ranked so only forward steps are emitted. Providers repeat their terminal
/// event -- a Responses search completes once as `web_search_call.completed`
/// and again as `output_item.done` -- and a replayed phase would duplicate
/// the client's lifecycle and double-count the search.
fn phase_rank(phase: &str) -> Option<i32> {
    match phase {
        "started" => Some(0),
        "searching" => Some(1),
        "completed" | "failed" => Some(2),
        _ => None,
    }
}

/// Record `phase` for search `id`; true when it advances the lifecycle.
pub fn hosted_tool_step(seen: &mut HashMap<String, String>, id: &str, phase: &str) -> bool {
    let Some(rank) = phase_rank(phase) else {
        return false;
    };
    let current = seen.get(id).and_then(|p| phase_rank(p)).unwrap_or(-1);
    if rank <= current {
        return false;
    }
    seen.insert(id.to_string(), phase.to_string());
    true
}

/// Translate upstream wire events into the shared response vocabulary.
pub trait Decoder: Send {
    fn decode(&mut self, event: &Value) -> Result<Vec<StreamEvent>>;
    fn finish(&mut self) -> Result<Vec<StreamEvent>>;
}

/// A client's request that output be constrained, not merely prompted.
///
/// Dialect-neutral because each dialect names the same capability its own
/// way (Responses `text.format`, Messages `output_config.format`). `kind` is
/// `json_schema` or `json_object`; a plain-text format is no constraint at
/// all and is dropped at the edge rather than carried as one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OutputFormat {
    pub kind: String,
    pub name: String,
    pub schema: Option<Object>,
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatRequest {
    pub model: String,
    /// Blocks rather than one string so cache breakpoints survive.
    pub system: Vec<Text>,
    pub turns: Vec<Turn>,
    pub tools: Vec<Tool>,
    pub tool_choice: Option<ToolChoice>,
    /// As sent (Null when absent): each provider validates it.
    pub max_tokens: Value,
    /// As sent (Null when absent).
    pub reasoning_effort: Value,
    /// Explicit budget; preferred over reasoning_effort where supported.
    pub thinking_budget: Option<i64>,
    /// "adaptive" or "disabled" when named; neither maps to a budget.
    pub thinking_mode: String,
    /// Anthropic thinking visibility: "summarized" or "omitted".
    pub thinking_display: String,
    /// OpenAI reasoning summary mode: "auto", "concise", "detailed" or "none".
    pub reasoning_summary: String,
    /// Responses reasoning context: "auto", "current_turn" or "all_turns".
    pub reasoning_context: String,
    /// OpenAI output verbosity: "low", "medium" or "high".
    pub verbosity: String,
    pub parallel_tool_calls: Option<bool>,
    pub stream: bool,
    /// Account affinity: requests of one session start on the same account.
    pub session: String,
    /// The name of the proxy key the request came with, for attribution.
    pub caller: String,
    /// The client's own prompt-cache key, for an upstream that takes one.
    pub cache_key: String,
    /// A breakpoint the upstream places automatically at the end of the prompt.
    pub cache: Cache,
    /// As sent; each provider validates what it can honour.
    pub params: Object,
    /// Schema-constrained output when the client asked for one. A provider
    /// that cannot constrain its upstream must reject this rather than answer
    /// with unconstrained prose the client will fail to parse.
    pub output_format: Option<OutputFormat>,
}

impl ChatRequest {
    /// No assistant turn yet, so no upstream prompt cache to keep warm.
    pub fn starts_conversation(&self) -> bool {
        !self.turns.iter().any(|turn| turn.role == Role::Assistant)
    }
}
