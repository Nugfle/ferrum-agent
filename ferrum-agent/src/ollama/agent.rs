use std::collections::HashMap;

use futures_util::future::pending;
use ollama_api::{
    ApiConnection, OllamaApiError,
    dtos::{GenerateChatMessageResponse, Message, Role, StreamChatPartialResponse, StreamChatResponse},
};
use tokio::select;
use tokio::sync::mpsc;
use tracing::{debug, info};

use crate::{tools::DynTool, ui::UIMessage};

#[derive(Debug)]
pub struct AgentHandle {
    pub command_sender: mpsc::Sender<AgentCommand>,
    pub message_reciever: mpsc::Receiver<UIMessage>,
}

#[derive(Debug, Clone)]
pub enum AgentCommand {
    GeneratePrompt(String),
    ChangeMode(AgentMode),
    ChangeModel(String),
    ClearHistory,
    Stop,
}

#[derive(Debug, Clone, Copy)]
pub enum AgentMode {
    Plan,
    Build,
}

pub struct OllamaAgent {
    history: Vec<Message>,
    mode: AgentMode,
    model: String,
    tools: HashMap<String, Box<dyn DynTool>>,

    /// this is the command channel from the UI (up)
    command_reciever: mpsc::Receiver<AgentCommand>,
    /// this is the connection to the UI (up)
    message_out: mpsc::Sender<UIMessage>,

    /// this is our way of communicating with the API (down)
    api_connection: ApiConnection,
    /// this is the connection to the OllamaAPI (down)
    api_msg_receiver: Option<mpsc::Receiver<Result<StreamChatResponse, OllamaApiError>>>,
}

impl OllamaAgent {
    pub fn new(url: String, tools: HashMap<String, Box<dyn DynTool>>, model: String) -> AgentHandle {
        let (command_sender, command_reciever) = mpsc::channel(8);
        let (message_sender, message_reciever) = mpsc::channel(8);
        tokio::spawn(async move {
            let mut this = Self {
                history: Vec::new(),
                mode: AgentMode::Build,
                model,
                tools,
                command_reciever,
                message_out: message_sender,
                api_connection: ApiConnection::new(url),
                api_msg_receiver: None,
            };
            // ToDo: add recovery
            this.run().await;
        });
        AgentHandle {
            command_sender,
            message_reciever,
        }
    }

    async fn start_generating_prompt(&mut self, prompt: String) {
        self.history.push(Message {
            role: Role::User,
            content: prompt,
            images: Vec::new(),
            tool_calls: Vec::new(),
        });
        let mut body = ollama_api::dtos::GenerateChatMessageRequest {
            model: &self.model,
            messages: &self.history,
            tools: self.tools.values().map(|tool| tool.as_ollama_api_tool()).collect(),
            format: None,
            options: None,
            stream: Some(true),
            think: None,
            keep_alive: None,
            logprobs: None,
            top_logprobs: None,
        };
        debug!("Start running prompt: {:?}", body);

        match self.api_connection.run_chat_prompt_stream(&mut body).await {
            Ok(v) => {
                self.api_msg_receiver = Some(v);
            }
            Err(e) => _ = self.message_out.send(UIMessage::APIError(e)).await,
        }
    }

    async fn run(&mut self) {
        info!("Starting the main Agent Loop");
        loop {
            // this future will not trigger if api_msg_receiver is None turning the select into a
            // simple await for the command channel
            let option_future = async {
                if let Some(rx) = &mut self.api_msg_receiver {
                    rx.recv().await
                } else {
                    pending().await
                }
            };
            select! {
                Some(command) = self.command_reciever.recv() =>{
                    match command {
                        AgentCommand::Stop => break,
                        AgentCommand::ClearHistory => self.history = Vec::new(),
                        AgentCommand::ChangeMode(mode) => self.mode = mode,
                        AgentCommand::ChangeModel(model) => self.model = model,
                        AgentCommand::GeneratePrompt(prompt) => self.start_generating_prompt(prompt).await
                    }
                },
                Some(api_result) = option_future => {
                    match api_result {
                        Ok(message) => {
                            match message {
                                StreamChatResponse::Chunk(chunk) => self.process_message_chunk(chunk).await,
                                StreamChatResponse::Last(last) => self.process_last_message(last).await,
                            }
                        }
                        Err(e) => {
                            self.message_out
                                .send(UIMessage::APIError(e)).await
                                .map_err(|e| OllamaApiError::Custom(format!("Can't reach the UI: {}", e)))
                                .unwrap();
                        },
                    }

                }
            }
        }
    }

    async fn process_message_chunk(&mut self, chunk: StreamChatPartialResponse) {
        debug!("got message chunk: {:#?}", chunk);
        // TODO: store stuff like tool calls, and execute them.
        self.message_out
            .send(UIMessage::from(chunk))
            .await
            .map_err(|e| OllamaApiError::Custom(format!("Can't reach the UI: {}", e)))
            .unwrap() // we crash if we can't reach the UI. 
    }
    async fn process_last_message(&mut self, last: GenerateChatMessageResponse) {
        debug!("got last message: {:#?}", last);
        // TODO: add tool use here
        self.message_out
            .send(UIMessage::from(last))
            .await
            .map_err(|e| OllamaApiError::Custom(format!("Can't reach the UI: {}", e)))
            .unwrap()
    }
}
