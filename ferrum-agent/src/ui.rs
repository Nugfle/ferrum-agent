use ollama_api::OllamaApiError;
use ratatui::{
    Terminal,
    backend::Backend,
    crossterm::event::{self, Event, KeyCode, KeyModifiers},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use std::io;
use tui_textarea::TextArea;

use crate::ollama::agent::{AgentCommand, AgentHandle};

#[derive(Debug, Clone)]
pub enum UIMessage {
    TextMessage(UITextMessage),
    APIError(OllamaApiError),
    ToolUse(UIToolUseMessage),
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

pub struct App<'a> {
    agent_handle: AgentHandle,
    messages: Vec<UIMessage>,
    input: TextArea<'a>,
    current_model: String,
    is_processing: bool,
    should_quit: bool,
    list_state: ListState, // We need this to control scrolling
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
            list_state: ListState::default(),
        }
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        loop {
            terminal.draw(|f| self.ui(f))?;

            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        _ = self.agent_handle.command_sender.send(AgentCommand::Stop).await;
                        self.should_quit = true;
                    }
                    KeyCode::Char('m') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.current_model = if self.current_model == "my-coder" {
                            "llama3".to_string()
                        } else {
                            "my-coder".to_string()
                        };
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
        }
    }

    async fn on_send(&mut self) {
        if self.input.lines()[0].is_empty() {
            return;
        }

        let content = self.input.lines().join("\n");

        // Clear input by overwriting the widget
        self.input = TextArea::default();
        self.input
            .set_block(Block::default().borders(Borders::ALL).title("Input (Enter to send)"));

        self.messages.push(UIMessage::TextMessage(UITextMessage {
            author: "User".to_string(),
            content: content.clone(),
        }));
        let cmd = AgentCommand::GeneratePrompt(content);
        // ToDo: add recovery
        self.agent_handle.command_sender.send(cmd).await.expect("can't reach agent");
        self.is_processing = true;

        // Auto-scroll to the bottom
        // We select the index of the last item
        if !self.messages.is_empty() {
            self.list_state.select(Some(self.messages.len() - 1));
        }
    }

    fn ui(&mut self, f: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1), Constraint::Length(5)])
            .split(f.area());

        let messages: Vec<ListItem> = self
            .messages
            .iter()
            .map(|m| {
                let header = Line::from(vec![Span::styled(
                    format!("{}: ", m.get_author()),
                    Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
                )]);
                let content_lines = vec![header, Line::from(m.get_text()), Line::from("")];
                ListItem::new(content_lines)
            })
            .collect();

        let messages_list = List::new(messages)
            .block(Block::default().borders(Borders::ALL).title("Chat History"))
            .highlight_style(Style::default().add_modifier(Modifier::ITALIC));

        f.render_stateful_widget(messages_list, chunks[0], &mut self.list_state);

        let status_style = if self.is_processing {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default().bg(Color::Blue).fg(Color::White)
        };

        let status_text = format!(" Model: {} | Mode: Standard | [Esc] Quit | [Ctrl+M] Switch Model ", self.current_model);
        let status_bar = Paragraph::new(status_text)
            .style(status_style)
            .alignment(ratatui::layout::Alignment::Center);

        f.render_widget(status_bar, chunks[1]);
        f.render_widget(&self.input, chunks[2]);
    }
}
