#![allow(unused)]
use std::fmt::Debug;

use crate::dtos::{GenerateChatMessageRequest, GenerateChatMessageResponse, Role, StreamChatPartialResponse, StreamChatResponse, Tool, ToolCall};
use futures_util::stream::Stream;
use futures_util::{StreamExt, TryStreamExt};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio_util::io::StreamReader;
use tracing::{info, instrument};
pub mod dtos;

#[derive(Error, Debug, Clone)]
pub enum OllamaApiError {
    #[error("Can't connect to ollama: {0}")]
    Unreachable(String),
    #[error("The request timed out: {0}")]
    TimedOut(String),
    #[error("Failed to decode the server response: {0}")]
    DecodeFailiure(String),
    #[error("The request returned an error status: {0}")]
    ErrorStatus(String),
    #[error("Ther request was rejected because of an error in the body: {0}")]
    BadRequest(String),
    #[error("The request failed: {0}")]
    Custom(String),
}

impl From<reqwest::Error> for OllamaApiError {
    fn from(value: reqwest::Error) -> Self {
        if value.is_connect() {
            OllamaApiError::Unreachable(value.to_string())
        } else if value.is_timeout() {
            OllamaApiError::TimedOut(value.to_string())
        } else if value.is_decode() {
            OllamaApiError::DecodeFailiure(value.to_string())
        } else if value.is_status() {
            OllamaApiError::ErrorStatus(value.status().expect("is error status, requires status to be set").to_string())
        } else if value.is_request() {
            OllamaApiError::BadRequest(value.to_string())
        } else {
            Self::Custom(value.to_string())
        }
    }
}

impl From<serde_json::Error> for OllamaApiError {
    fn from(value: serde_json::Error) -> Self {
        Self::DecodeFailiure(value.to_string())
    }
}

pub struct ApiConnection {
    url: String,
    client: Client,
}

impl ApiConnection {
    pub fn new(url: String) -> Self {
        Self { url, client: Client::new() }
    }
    /// sends the request to the model and waits for a complete response. If you need a non blocking version that gives you access
    /// to a live token stream use [`Self::run_chat_prompt_stream`].
    pub async fn run_chat_prompt_blocking<'a>(
        &self,
        body: &mut GenerateChatMessageRequest<'a>,
    ) -> Result<GenerateChatMessageResponse, OllamaApiError> {
        let mut receiver = self.run_chat_prompt_stream(body).await?;

        let mut message_content: String = String::new();
        let mut tool_calls: Vec<dtos::ToolCall> = Vec::new();
        let mut images: Vec<String> = Vec::new();
        let mut thoughts: String = String::new();

        while let Some(msg_res) = receiver.recv().await {
            match msg_res? {
                StreamChatResponse::Chunk(chunk) => {
                    put_chunks_into_buffers(&mut message_content, &mut tool_calls, &mut images, &mut thoughts, chunk)?
                }
                StreamChatResponse::Last(mut last) => return Ok(handle_last(message_content, tool_calls, images, thoughts, last)),
            }
        }

        Err(OllamaApiError::Custom("The stream ended unexpectedly without 'Last' chunk".to_string()))
    }

    /// Runs the chat prompt as a stream. Returns a channel which will contain the partial messges
    /// produced by the model. If you don't need the stream, consider using
    /// [`ApiConnection::run_chat_prompt_blocking`]
    pub async fn run_chat_prompt_stream<'a>(
        &self,
        body: &mut GenerateChatMessageRequest<'a>,
    ) -> Result<mpsc::Receiver<Result<StreamChatResponse, OllamaApiError>>, OllamaApiError> {
        body.stream = Some(true);
        let resp = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .and_then(|r| r.error_for_status())?;
        let (s, r) = mpsc::channel(300);
        tokio::spawn(async move { handle_stream_response(resp, s).await });
        Ok(r)
    }
}
pub fn handle_last(
    mut content_buf: String,
    mut tool_buf: Vec<ToolCall>,
    mut image_buf: Vec<String>,
    mut thoughts_buf: String,
    mut last: GenerateChatMessageResponse,
) -> GenerateChatMessageResponse {
    content_buf.push_str(&last.message.content);
    tool_buf.extend_from_slice(&last.message.tool_calls);
    image_buf.extend_from_slice(&last.message.images);
    if !last.message.thinking.is_empty() {
        thoughts_buf.push_str(&last.message.thinking);
    }

    last.message.content = content_buf;
    last.message.tool_calls = tool_buf;
    last.message.images = image_buf;
    last.message.thinking = thoughts_buf;

    last
}

pub fn put_chunks_into_buffers(
    content_buf: &mut String,
    tool_buf: &mut Vec<ToolCall>,
    image_buf: &mut Vec<String>,
    thoughts_buf: &mut String,
    chunk: StreamChatPartialResponse,
) -> Result<(), OllamaApiError> {
    content_buf.push_str(&chunk.message.content);
    tool_buf.extend_from_slice(&chunk.message.tool_calls);
    image_buf.extend_from_slice(&chunk.message.images);
    if !chunk.message.thinking.is_empty() {
        thoughts_buf.push_str(&chunk.message.thinking);
    }

    if chunk.done {
        return Err(OllamaApiError::Custom("Invalid API operation. Sent 'done' in a non 'done' response".to_string()));
    }
    _ = chunk.created_at;
    _ = chunk.model;

    Ok(())
}

async fn handle_stream_response<T: DeserializeOwned>(mut resp: Response, sender: mpsc::Sender<Result<T, OllamaApiError>>) {
    let stream = resp.bytes_stream().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    let stream_reader = StreamReader::new(stream);
    let mut lines = BufReader::new(stream_reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let deserialized = serde_json::from_str::<T>(&line).map_err(|e| e.into());
        _ = sender.send(deserialized).await;
    }
}
