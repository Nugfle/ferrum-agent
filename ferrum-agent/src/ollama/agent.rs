use std::collections::HashMap;

use ollama_api::{OllamaApiError, dtos::Message};
use tokio::sync::mpsc;

use crate::{tools::DynTool, ui::UIMessage};

pub struct AgentHandle {
    pub command_sender: mpsc::Sender<AgentCommand>,
    pub message_reciever: mpsc::Receiver<UIMessage>,
}

pub enum AgentCommand {
    GeneratePrompt(String),
    ChangeMode(AgentMode),
    ChangeModell(String),
    Clear,
    Stop,
}

pub enum AgentMode {
    Plan,
    Build,
}

pub struct OllamaAgent {
    url: String,
    history: Vec<Message>,
    tools: HashMap<String, Box<dyn DynTool>>,
    command_reciever: mpsc::Receiver<AgentCommand>,
    message_out: mpsc::Sender<UIMessage>,
}

impl OllamaAgent {
    pub fn new(url: String, tools: HashMap<String, Box<dyn DynTool>>) -> AgentHandle {
        let (command_sender, command_reciever) = mpsc::channel(8);
        let (message_sender, message_reciever) = mpsc::channel(8);
        tokio::spawn(async move {
            let mut this = Self {
                url,
                history: Vec::new(),
                tools,
                command_reciever,
                message_out: message_sender,
            };
            // ToDo: add recovery
            this.run().await.expect("the agent crashed");
        });
        AgentHandle {
            command_sender,
            message_reciever,
        }
    }

    async fn run(&mut self) -> Result<(), OllamaApiError> {
        loop {}
        Ok(())
    }
}
