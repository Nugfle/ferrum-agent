use ollama_api::{
    OllamaApiError,
    dtos::{GenerateChatMessageResponse, StreamChatPartialResponse},
};
use ratatui::{
    Terminal,
    backend::Backend,
    crossterm::event::{self, Event, KeyCode, KeyModifiers},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use tokio::{
    select,
    time::{MissedTickBehavior, interval},
};
use tracing::info;

use std::{io, time::Duration};
use tui_textarea::TextArea;

use crate::ollama::agent::{AgentCommand, AgentHandle};

#[derive(Debug, Clone)]
pub enum UIMessage {
    TextMessage(UITextMessage),
    APIError(OllamaApiError),
    ToolUse(UIToolUseMessage),
}

impl From<StreamChatPartialResponse> for UIMessage {
    fn from(value: StreamChatPartialResponse) -> Self {
        Self::TextMessage(UITextMessage {
            author: "Assistent".to_string(),
            content: value.message.content,
        })
    }
}
impl From<GenerateChatMessageResponse> for UIMessage {
    fn from(value: GenerateChatMessageResponse) -> Self {
        Self::TextMessage(UITextMessage {
            author: "Assistent".to_string(),
            content: value.message.content,
        })
    }
}

impl UIMessage {
    pub fn get_author(&self) -> &str {
        match self {
            UIMessage::TextMessage(m) => &m.author,
            UIMessage::ToolUse(_) => "System",
            UIMessage::APIError(_) => "System",
        }
    }
    pub fn get_text(&self) -> String {
        match self {
            UIMessage::TextMessage(m) => m.content.clone(),
            UIMessage::ToolUse(t) => format!("using {} with arguments: {}. Status: {:?}", t.tool_name, t.arguments, t.status),
            UIMessage::APIError(e) => format!("the agent crashed: {}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UITextMessage {
    author: String,
    content: String,
}

#[derive(Debug, Clone)]
pub struct UIToolUseMessage {
    tool_name: String,
    arguments: String,
    status: ToolUseStatus,
}

#[derive(Debug, Clone, Copy)]
pub enum ToolUseStatus {
    InProgress,
    Success,
    Failure,
}

#[derive(Debug)]
pub struct App<'a> {
    agent_handle: AgentHandle,
    messages: Vec<UIMessage>,
    input: TextArea<'a>,
    current_model: String,
    is_processing: bool,
    should_quit: bool,
    vertical_scroll: u16,
}

impl<'a> App<'a> {
    pub fn new(agent_handle: AgentHandle) -> Self {
        let mut input = TextArea::default();
        input.set_block(Block::default().borders(Borders::ALL).title("Input (Enter to send)"));

        input.set_cursor_line_style(Style::default());

        Self {
            agent_handle,
            input,
            messages: Vec::new(),
            current_model: "my-coder".to_string(),
            is_processing: false,
            should_quit: false,
            vertical_scroll: 0,
        }
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        info!("start running App loop");
        let mut render_interval = interval(Duration::from_millis(50));
        render_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            select! {
                _ = render_interval.tick() => {
                    terminal.draw(|f| self.ui(f))?;
                    if let Event::Key(key) = event::read()? {
                        match key.code {
                            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                _ = self.agent_handle.command_sender.send(AgentCommand::Stop).await;
                                self.should_quit = true;
                            }
                            KeyCode::Enter => {
                                self.on_send().await;
                            }
                            _ => {
                                self.input.input(key);
                            }
                        }
                    }

                    if self.should_quit {
                        return Ok(());
                    }
                },
                Some(msg) = self.agent_handle.message_reciever.recv() =>  self.on_msg(msg).await,
            }
        }
    }

    async fn on_msg(&mut self, msg: UIMessage) {
        match self.messages.last_mut() {
            Some(UIMessage::TextMessage(latest)) => match msg {
                UIMessage::TextMessage(new) => {
                    if latest.author == new.author {
                        latest.content.push_str(&new.content);
                        return;
                    } else {
                        self.messages.push(UIMessage::TextMessage(new));
                    }
                }
                UIMessage::APIError(e) => {
                    self.messages.push(UIMessage::APIError(e));
                    self.is_processing = false;
                }
                UIMessage::ToolUse(_) => todo!(),
            },
            _ => self.messages.push(msg),
        }
    }

    async fn on_send(&mut self) {
        if self.input.lines()[0].is_empty() {
            return;
        }

        let content = self.input.lines().join("\n");
        info!("Sending a prompt request with content: {}", content);

        // Clear input by overwriting the widget
        self.input = TextArea::default();
        self.input
            .set_block(Block::default().borders(Borders::ALL).title("Input (Enter to send)"));

        self.messages.push(UIMessage::TextMessage(UITextMessage {
            author: "User".to_string(),
            content: content.clone(),
        }));

        let cmd = AgentCommand::GeneratePrompt(content);
        self.agent_handle.command_sender.send(cmd).await.expect("can't reach agent");
        self.is_processing = true;
    }

    fn ui(&mut self, f: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1), Constraint::Length(5)])
            .split(f.area());

        let mut text_lines = Vec::new();

        for m in &self.messages {
            // Header line
            text_lines.push(Line::from(vec![Span::styled(
                format!("{}: ", m.get_author()),
                Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
            )]));

            // Content line (Paragraph will wrap this automatically!)
            text_lines.push(Line::from(m.get_text()));

            // Spacer line
            text_lines.push(Line::from(""));
        }

        // 2. Create the Paragraph widget
        let messages_paragraph = Paragraph::new(text_lines)
            .block(Block::default().borders(Borders::ALL).title("Chat History"))
            .wrap(Wrap { trim: true })
            .scroll((self.vertical_scroll, 0)); // You need to track a u16 for scroll offset

        f.render_widget(messages_paragraph, chunks[0]);

        let status_style = if self.is_processing {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default().bg(Color::Blue).fg(Color::White)
        };

        let status_text = format!(" Model: {} | Mode: Standard | [Esc] Quit ", self.current_model);
        let status_bar = Paragraph::new(status_text)
            .style(status_style)
            .alignment(ratatui::layout::Alignment::Center);

        f.render_widget(status_bar, chunks[1]);
        f.render_widget(&self.input, chunks[2]);
    }
}
