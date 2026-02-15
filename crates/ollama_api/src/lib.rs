#![allow(unused)]
use std::str::Bytes;

use crate::dtos::{
    GenerateChatMessageRequest, GenerateChatMessageResponse, Role, StreamChatResponse,
};
use futures_util::stream::Stream;
use futures_util::{StreamExt, TryStreamExt};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;
use thiserror::Error;
use tokio::sync::mpsc;
pub mod dtos;

#[derive(Error, Debug)]
pub enum OllamaApiError {
    #[error("Can't connect to ollama: {0}")]
    Unreachable(String),
    #[error("The request timed out: {0}")]
    TimedOut(String),
    #[error("Failed to decode the server response: {0}")]
    DecodeFailiure(String),
    #[error("The request returned an error status")]
    ErrorStatus(String),
    #[error("Ther request was rejected because of an error in the body: {0}")]
    BadRequest(String),
    #[error("The request failed: {0}")]
    Custom(String),
}

impl From<reqwest::Error> for OllamaApiError {
    fn from(value: reqwest::Error) -> Self {
        value.to_string();
        if value.is_connect() {
            OllamaApiError::Unreachable(value.to_string())
        } else if value.is_timeout() {
            OllamaApiError::TimedOut(value.to_string())
        } else if value.is_decode() {
            OllamaApiError::DecodeFailiure(value.to_string())
        } else if value.is_status() {
            OllamaApiError::ErrorStatus(
                value
                    .status()
                    .expect("is error status, requires status to be set")
                    .to_string(),
            )
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
        Self {
            url,
            client: Client::new(),
        }
    }
    /// sends the request to the model and waits for a complete response. This method will modify
    /// the body to set stream to false. If you need a non blocking version that gives you access
    /// to a live token stream use [`Self::run_chat_prompt_stream`].
    pub async fn run_chat_prompt_blocking<'a>(
        &self,
        body: GenerateChatMessageRequest<'a>,
    ) -> Result<GenerateChatMessageResponse, OllamaApiError> {
        if body.stream.is_none_or(|s| s) {}
        let url = format!("{}/api/chat", self.url);
        let resp = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let text = resp.text().await?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Runs the chat prompt as a stream. Returns a channel which will contain the partial messges
    /// produced by the model. If you don't need the stream, consider using
    /// [`ApiConnection::run_chat_prompt_blocking`]
    pub async fn run_chat_prompt_stream<'a>(
        &self,
        mut body: GenerateChatMessageRequest<'a>,
    ) -> mpsc::Receiver<Result<StreamChatResponse, OllamaApiError>> {
        let (s, r) = mpsc::channel(300);

        body.stream = Some(true);
        let resp = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .and_then(|r| r.error_for_status());

        match resp {
            Ok(resp) => {
                tokio::spawn(async move { handle_stream_response(resp, s).await });
            }
            Err(e) => _ = s.send(Err(e.into())).await,
        }
        r
    }
}

async fn handle_stream_response<T: DeserializeOwned>(
    response: Response,
    sender: mpsc::Sender<Result<T, OllamaApiError>>,
) {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(bytes_result) = stream.next().await {
        match bytes_result {
            Ok(bytes) => {
                buffer.extend_from_slice(&bytes);
                while let Some(line) = buffer
                    .iter()
                    .position(|b| *b == b'\n')
                    .and_then(|pos| Some(buffer.drain(..pos).collect::<Vec<u8>>()))
                {
                    _ = sender
                        .send(serde_json::from_slice::<T>(&line).map_err(|e| e.into()))
                        .await;
                }
            }
            Err(e) => _ = sender.send(Err(e.into())).await,
        }
    }
}
