use std::{collections::HashMap, mem::take};

use futures_util::future::pending;
use ollama_api::{
    ApiConnection, OllamaApiError,
    dtos::{GenerateChatMessageResponse, Message, Role, StreamChatResponse, ToolCall},
};
use tokio::select;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::{
    tools::{DynTool, RunToolError},
    ui::{UIEvent, UIMessage, UIToolUseMessage},
};

#[derive(Debug, Clone)]
pub enum AgentCommand {
    GeneratePrompt(String),
    ChangeOutChannel(mpsc::Sender<UIEvent>),
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

#[derive(thiserror::Error, Debug, Clone)]
pub enum AgentError {
    #[error("The Request to the agent failed due to an API error: {0}")]
    APIError(OllamaApiError),
    #[error("The Agent failed when trying to run a tool: {0}")]
    ToolError(RunToolError),
    #[error("The Agent is unable to reach the UI: {0}")]
    UIUnreachable(String),
}

impl From<RunToolError> for AgentError {
    fn from(e: RunToolError) -> Self {
        Self::ToolError(e)
    }
}
impl From<OllamaApiError> for AgentError {
    fn from(e: OllamaApiError) -> Self {
        Self::APIError(e)
    }
}

pub struct OllamaAgent {
    history: Vec<Message>,
    mode: AgentMode,
    model: String,
    tools: HashMap<String, Box<dyn DynTool>>,

    /// this is the command channel from the UI (from up)
    command_reciever: mpsc::Receiver<AgentCommand>,
    /// this is the connection to the UI (to up)
    message_out: mpsc::Sender<UIEvent>,

    /// this is our way of communicating with the API (to down)
    api_connection: ApiConnection,
    /// this is the reciever for messages produced by the Ollama API (from down)
    api_msg_receiver: Option<mpsc::Receiver<Result<StreamChatResponse, OllamaApiError>>>,

    /// this holds the Stream Responses recieved for a prompt. It is used to assemble the full
    /// message when the last chunk is recieved
    current_message_buffer: Vec<StreamChatResponse>,
}

impl OllamaAgent {
    pub fn new(
        url: String,
        tools: HashMap<String, Box<dyn DynTool>>,
        model: String,
        system_prompt: Option<String>,
    ) -> (mpsc::Sender<AgentCommand>, mpsc::Receiver<UIEvent>) {
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
                current_message_buffer: Vec::new(),
            };
            if let Some(prompt) = system_prompt {
                this.history.push(Message {
                    role: Role::System,
                    content: prompt,
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                })
            }
            // ToDo: add recovery
            this.run().await;
        });
        (command_sender, message_reciever)
    }

    async fn start_generating_prompt(&mut self, prompt: Option<String>) {
        if let Some(prompt) = prompt {
            self.history.push(Message {
                role: Role::User,
                content: prompt,
                images: Vec::new(),
                tool_calls: Vec::new(),
            });
        }
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
            Err(e) => {
                error!("An error occured when trying to run the prompt: {}", e);
                _ = self.message_out.send(UIEvent::MessageRecieved(e.into())).await;
            }
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
                        AgentCommand::ChangeOutChannel(channel) => self.message_out = channel,
                        AgentCommand::ChangeMode(mode) => self.mode = mode,
                        AgentCommand::ChangeModel(model) => self.model = model,
                        AgentCommand::GeneratePrompt(prompt) => self.start_generating_prompt(Some(prompt)).await
                    }
                },
                Some(api_result) = option_future => {
                    match api_result {
                        Ok(stream_response) => {
                            if let Err(e) = self.process_stream_response(stream_response).await {
                                error!("Agent loop stopped because of error: {}", e);
                                break;
                            };
                        }
                        Err(e) => {
                            self.message_out
                                .send(UIEvent::MessageRecieved(e.into())).await
                                .map_err(|e|{
                                    error!("The Agent is unable to reach the UI: {e}");
                                    AgentError::UIUnreachable(e.to_string())
                                })
                                .unwrap();
                        },
                    }

                }
            }
        }
    }

    async fn process_stream_response(&mut self, msg: StreamChatResponse) -> Result<(), AgentError> {
        self.current_message_buffer.push(msg.clone());
        let is_last = msg.is_last();

        self.message_out.send(UIEvent::MessageRecieved(UIMessage::from(msg))).await.map_err(|e| {
            error!("The Agent is unable to reach the UI: {e}");
            AgentError::UIUnreachable(e.to_string())
        })?;

        if is_last {
            let final_message = self.construct_message();

            self.history.push(Message {
                role: Role::Assistant,
                content: final_message.message.content,
                images: final_message.message.images,
                tool_calls: final_message.message.tool_calls.clone(),
            });
            if !final_message.message.tool_calls.is_empty() {
                self.run_tool_calls(&final_message.message.tool_calls).await;
                // after running the tools we have to send the results back to the model
                self.start_generating_prompt(None).await;
            }
        }

        Ok(())
    }

    /// constructs the message from the parts stored in the buffer and clears the buffer
    fn construct_message(&mut self) -> GenerateChatMessageResponse {
        let buf = take(&mut self.current_message_buffer);
        let assembled_message = buf
            .into_iter()
            .fold(GenerateChatMessageResponse::default(), |mut combined, partial| match partial {
                StreamChatResponse::Chunk(mut c) => {
                    combined.message.tool_calls.append(&mut c.message.tool_calls);
                    combined.message.content.push_str(&c.message.content);
                    combined.message.images.append(&mut c.message.images);
                    combined.message.thinking.push_str(&c.message.thinking);
                    combined
                }
                StreamChatResponse::Last(mut l) => {
                    combined.message.tool_calls.append(&mut l.message.tool_calls);
                    combined.message.content.push_str(&l.message.content);
                    combined.message.images.append(&mut l.message.images);
                    combined.message.thinking.push_str(&l.message.thinking);
                    combined.done = true;
                    combined.model = l.model;
                    combined.created_at = l.created_at;
                    combined.logprobs = l.logprobs;
                    combined.eval_count = l.eval_count;
                    combined.done_reason = l.done_reason;
                    combined.load_duration = l.load_duration;
                    combined.eval_duration = l.eval_duration;
                    combined.total_duration = l.total_duration;
                    combined.prompt_eval_count = l.prompt_eval_count;
                    combined
                }
            });
        assembled_message
    }

    async fn run_tool_calls(&mut self, tool_calls: &[ToolCall]) {
        for tool_call in tool_calls {
            info!("running tool call: {}, with arguments: {}", tool_call.function.name, tool_call.function.arguments);
            let msg = match self.execute_tool_call(&tool_call).await {
                Ok(res) => Message {
                    role: Role::Tool,
                    content: res,
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                },
                Err(e) => Message {
                    role: Role::Tool,
                    content: format!("The tool call failed with the error: {e}"),
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                },
            };
            info!("finished tool call with result: {:?}", msg);
            self.message_out
                .send(UIEvent::MessageRecieved(UIMessage::ToolUse(UIToolUseMessage {
                    tool_name: tool_call.function.name.clone(),
                    arguments: tool_call.function.arguments.to_string(),
                    result: msg.content.clone(),
                })))
                .await
                .expect("can't reach UI");
            self.history.push(msg)
        }
    }

    async fn execute_tool_call(&mut self, tool_call: &ToolCall) -> Result<String, RunToolError> {
        if let Some(tool) = self.tools.get(&tool_call.function.name) {
            tool.run(tool_call.function.arguments.clone()).await
        } else {
            Err(RunToolError::ToolNotFound {
                tool_name: tool_call.function.name.clone(),
            })
        }
    }
}
