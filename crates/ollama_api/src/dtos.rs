use std::{error::Error, time::Duration};

use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::OllamaApiError;

/// Common options for most Ollama API calls
#[derive(Debug, Serialize, Default, Clone)]
pub struct OllamaRequestOptions {
    /// Random seed used for reproducible outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Controls randomness in generation (higher = more random)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Limits next token selection to the K most likely
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Cumulative probability threshold for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Minimum probability threshold for token selection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Stop sequences that will halt generation
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Context length size (number of tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u64>,
    /// Maimum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u64>,
    /// Set this to 0 to offload the model to the CPU. Dont touch this otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<u8>,
    /// Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance.
    /// It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u8>,
}

/// specifies the time that a model should remain in Memory after completing the generation
#[derive(Debug, Default, Clone, Copy)]
pub enum KeepAlive {
    #[default]
    Indefinitely,
    Duration(Duration),
}

/// DTO for generating Embeddings
#[derive(Debug, Serialize, Default, Clone)]
pub struct GenerateEmbeddingRequest<'a> {
    /// The model to use when generating the embeddings, f.e. ``"nomic-embed-text"``
    pub model: &'a str,
    /// The texts we want to generate the embeddings for
    pub input: Vec<String>,
    /// Whether to truncate inputs that exceed the context window
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    /// Number of dimensions to generate the embeddings for
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// The Duration for which the model will stay in memory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<KeepAlive>,
    /// Different options for the generation, such as using the gpu or cpu
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaRequestOptions>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GenerateEmbeddingResponse {
    /// The model used for generating the response, f.e. ``"nomic-embed-text"``
    pub model: String,
    /// The embeddings generated for the inputs
    pub embeddings: Vec<Vec<f32>>,
    /// The total duration of the generation in nanoseconds
    pub total_duration: u64,
    /// The total time spent loading the model in nanoseconds
    pub load_duration: u64,
    /// The number of input tokens that were processed for generating the embeddings
    pub prompt_eval_count: u64,
}

#[derive(Debug, Serialize, Clone)]
pub struct GenerateChatMessageRequest<'a> {
    /// The model name
    pub model: &'a str,
    /// The chat history as an array of message objects
    pub messages: &'a Vec<Message>,

    /// A list of function tools that the model may call during the chat
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    /// Format to return a response in. Can be
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponseFormat>,
    /// Runtime options to control the text generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaRequestOptions>,
    /// Whether to send a stream of output tokens from the model, or just the finished response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Controls whether to return seperate thinking output in addition to the content. (can be
    /// true and false for most models, "high", "medium", and "low" for models that support it)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkLevel>,
    /// The Duration in which the model will stay in memory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<KeepAlive>,
    /// Whether to return log probabilities of the output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of most likely tokens to return at each token position when logprobs are enabled
    pub top_logprobs: Option<u32>,
}

/// The API response to a [`GenerateChatMessageRequest`] request
#[derive(Debug, Deserialize, Default, Clone)]
pub struct GenerateChatMessageResponse {
    /// Model name used for generating the message
    pub model: String,
    /// ISO 8601 encoded timestamp
    pub created_at: String,
    /// The message created by the model.
    pub message: GeneratedMessage,
    /// Indicates whether the chat response has finished. This is only relevant if
    /// [`GenerateChatMessageRequest::stream`] is true
    pub done: bool,
    /// The reason for the response finishing
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Total time spent generating in nanoseconds
    pub total_duration: u64,
    /// Total time spent loading in the model in nanoseconds
    pub load_duration: u64,
    /// Number of tokens in the prompt
    pub prompt_eval_count: u64,
    /// Time spent evaluating the prompt in nanoseconds
    pub prompt_eval_duration: u64,
    /// Number of tokens generated in the response
    pub eval_count: u64,
    /// Time spent generating toekns in nanosecons
    pub eval_duration: u64,
    /// Log probability information for the generated tokens when logprobs are enabled
    #[serde(default)]
    pub logprobs: Vec<LogProb>,
}

/// The partial messages produced by the model, if using the [`GenerateChatMessageRequest::stream`]
/// option
#[derive(Debug, Deserialize, Clone)]
pub struct StreamChatPartialResponse {
    /// the name of the model, that produced the message
    pub model: String,
    /// an ISO 8601 encoded Timestamp of the response creation
    pub created_at: String,
    /// the partilally generated message
    pub message: GeneratedMessage,
    /// whether the model is done
    pub done: bool,
}

/// The possible responses Reurned by the API
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamChatResponse {
    Chunk(StreamChatPartialResponse),
    Last(GenerateChatMessageResponse),
}

/// The representation of a message genererated by the model. We can omit the 'role' field, as it
/// is always assistant.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct GeneratedMessage {
    /// The message content as text
    pub content: String,
    /// Optional thinking trace that is returned if [`GenerateChatMessageRequest::think`] was not
    /// false
    #[serde(default)]
    pub thinking: Option<String>,
    /// A list of base-64 encoded images in case the model produced any.
    #[serde(default)]
    pub images: Vec<String>,
    /// Tool call requests produced by the model
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// The representation of a chat Message. A Message MUST have a Role and a content.
#[derive(Debug, Serialize, Default, Clone)]
pub struct Message {
    /// Author of the message.
    pub role: Role,
    /// The message content as text
    pub content: String,
    /// A list of inline images for multimodal models.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// Tool call requests produced by the model
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Serialize, Clone, Copy, Default)]
pub enum Role {
    #[serde(rename = "system")]
    #[default]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    Tool,
}

/// Represents an explicit call to a tool
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

/// Represents and explicit call to a tool function, including the name, description and arguments
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallFunction {
    /// the name of the function to call
    pub name: String,
    /// describes what the function does
    #[serde(default)] // in case the model doesn't send the description
    pub description: Option<String>,
    /// A JSON object of the arguments to pass to the function
    #[serde(default)] // in case the model omits the arguments field, if there are none
    pub arguments: Option<Value>,
}

/// Representation of an available tool for the model to use
#[derive(Debug, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    /// contains the neccesary information for the model too call the tool
    pub function: ToolFunction,
}

/// The tool type is required by the API, but is always Function
#[derive(Debug, Serialize, Clone, Copy)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

/// Represents an available tool function
#[derive(Debug, Serialize, Clone)]
pub struct ToolFunction {
    /// The function name
    pub name: String,
    /// A JSON schema for the parameters of the function
    pub parameters: schemars::Schema,
    /// A short description of what the function does in human readable form
    pub description: String,
}

/// Represents the format in which the model should respond
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// Asks the model, to format its output as valid JSON
    Json,
    /// Asks the model to adjust its output to fit into the provided JSON Schema
    Schema(Schema),
}

/// Represents the verbosity of the thinking output of the model. The default is 'true'.
#[derive(Debug, Clone, Copy)]
pub enum ThinkLevel {
    Bool(bool),
    High,
    Medium,
    Low,
}

/// Representation of a selected token and its probability, as well as all other viable tokens at
/// this position and their respective probabilities
#[derive(Debug, Deserialize, Clone)]
pub struct LogProb {
    /// the text representation of the token
    token: String,
    /// the log probability of this token
    logprob: f32,
    /// the raw byte representation of the token
    bytes: Vec<u8>,
    /// Most likely tokens and their log probabilities at this position
    top_logporbs: Vec<LogProbSecondary>,
}

/// Representation of a secondary, unselected Token
#[derive(Debug, Deserialize, Clone)]
pub struct LogProbSecondary {
    /// the text representation of the token
    pub token: String,
    /// the log probability of this token
    pub logprob: f32,
    /// the raw byte representation of the token
    pub bytes: Vec<u8>,
}

impl Serialize for ThinkLevel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Low => serializer.serialize_str("low"),
            Self::Medium => serializer.serialize_str("medium"),
            Self::High => serializer.serialize_str("high"),
            Self::Bool(b) => serializer.serialize_bool(*b),
        }
    }
}

impl Serialize for ResponseFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Json => serializer.serialize_str("json"),
            Self::Schema(s) => s.serialize(serializer),
        }
    }
}

impl Serialize for KeepAlive {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Duration(d) => {
                serializer.serialize_str(format!("{}s", d.as_secs().to_string()).as_str())
            }
            Self::Indefinitely => serializer.serialize_str("-1s"),
        }
    }
}

impl From<GeneratedMessage> for Message {
    fn from(value: GeneratedMessage) -> Self {
        Self {
            role: Role::Assistant,
            content: value.content,
            images: value.images,
            tool_calls: value.tool_calls,
        }
    }
}
