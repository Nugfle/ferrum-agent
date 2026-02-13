use std::{collections::HashMap, time::Duration};

use reqwest::Client;
use thiserror::Error;
use tracing::info;

use crate::{
    ollama::dtos::{
        GenerateChatMessageRequest, GenerateChatMessageResponse, GenerateEmbeddingRequest,
        GenerateEmbeddingResponse, KeepAlive, Message, OllamaRequestOptions, ResponseFormat, Role,
    },
    tools::DynTool,
};

pub struct OllamaConnection {
    client: Client,
    url: String,
    history: Vec<Message>,
    tools: HashMap<String, Box<dyn DynTool>>,
}

#[derive(Error, Debug)]
pub enum OllamaError {
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

impl From<reqwest::Error> for OllamaError {
    fn from(value: reqwest::Error) -> Self {
        value.to_string();
        if value.is_connect() {
            OllamaError::Unreachable(value.to_string())
        } else if value.is_timeout() {
            OllamaError::TimedOut(value.to_string())
        } else if value.is_decode() {
            OllamaError::DecodeFailiure(value.to_string())
        } else if value.is_status() {
            OllamaError::ErrorStatus(
                value
                    .status()
                    .expect("is error status, requires status to be set")
                    .to_string(),
            )
        } else if value.is_request() {
            OllamaError::BadRequest(value.to_string())
        } else {
            Self::Custom(value.to_string())
        }
    }
}

impl OllamaConnection {
    pub fn new(url: String) -> Self {
        Self {
            client: Client::new(),
            url,
            history: Vec::new(),
            tools: HashMap::new(),
        }
    }

    pub async fn run_prompt(
        &mut self,
        model: &str,
        prompt: &str,
        format: Option<ResponseFormat>,
    ) -> Result<(String, Option<String>), OllamaError> {
        self.history.push(Message {
            role: Role::User,
            content: prompt.to_string(),
            images: Vec::new(),
            tool_calls: Vec::new(),
        });

        let body = GenerateChatMessageRequest {
            model: model,
            messages: &self.history,
            format,
            keep_alive: Some(KeepAlive::Duration(Duration::from_mins(5))),
            stream: Some(false),
            tools: Vec::new(),
            options: None,
            think: None,
            logprobs: None,
            top_logprobs: None,
        };

        let url = format!("{}/api/chat", self.url);
        let resp = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let text = resp.text().await?;

        let response_body: GenerateChatMessageResponse = serde_json::from_str(&text).unwrap();

        let message_text = response_body.message.content.clone();
        let thinking_trace = response_body.message.thinking.clone();

        self.history.push(response_body.message.into());

        Ok((message_text, thinking_trace))
    }

    pub async fn get_embedding(
        &self,
        input: Vec<String>,
        embeddings_model: &str,
        run_on_cpu: bool,
    ) -> Result<Vec<Vec<f32>>, OllamaError> {
        let body = GenerateEmbeddingRequest {
            model: embeddings_model,
            input,
            keep_alive: Some(KeepAlive::Indefinitely),
            options: Some(OllamaRequestOptions {
                num_gpu: if run_on_cpu { Some(0) } else { None },
                ..Default::default()
            }),
            ..Default::default()
        };

        let response = self
            .client
            .post(format!("{}/api/embed", self.url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        if response.status().is_success() {
            let response_body: GenerateEmbeddingResponse = response.json().await?;
            Ok(response_body.embeddings)
        } else {
            Err(OllamaError::Custom(format!(
                "Response Status is: {}",
                response.status()
            )))
        }
    }
}
