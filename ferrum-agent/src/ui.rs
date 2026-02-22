use ollama_api::{
    OllamaApiError,
    dtos::{GenerateChatMessageResponse, StreamChatPartialResponse},
};
use ratatui::{
    Terminal,
    backend::Backend,
    crossterm::event::{self, Event, KeyCode, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Wrap},
};
use tokio::sync::mpsc;
use tracing::info;
use tui_input::{Input, backend::crossterm::EventHandler};

use crate::ollama::agent::AgentCommand;

#[derive(Debug, Clone)]
pub enum UIEvent {
    WindowEvent(Event),
    MessageRecieved(UIMessage),
}

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

    pub fn get_text_ref(&self) -> &str {
        match self {
            UIMessage::TextMessage(m) => &m.content,
            UIMessage::ToolUse(_) => "The agent is using a tool...",
            UIMessage::APIError(_) => "There was an error when trying to access the API",
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
pub struct App {
    messages: Vec<UIMessage>,
    input: Input,
    current_model: String,
    is_processing: bool,
    should_quit: bool,
    vertical_scroll: u16,

    agent_cmd_sender: mpsc::Sender<AgentCommand>,

    event_receiver: mpsc::Receiver<UIEvent>,
    event_sender: mpsc::Sender<UIEvent>,
}

async fn window_event_loop(event_sender: mpsc::Sender<UIEvent>) {
    loop {
        let event = event::read().expect("failed to read window event");
        event_sender
            .send(UIEvent::WindowEvent(event))
            .await
            .expect("failed to send UI event to main loop");
    }
}

impl App {
    pub async fn new(cmd_sender: mpsc::Sender<AgentCommand>) -> Self {
        let input = Input::default();

        let (s, r) = mpsc::channel(300);
        cmd_sender
            .send(AgentCommand::ChangeOutChannel(s.clone()))
            .await
            .expect("failed to send swap out channel request to Agent");

        let window_event_sender = s.clone();
        tokio::spawn(async move { window_event_loop(window_event_sender).await });

        Self {
            input,
            messages: Vec::new(),
            current_model: "my-coder".to_string(),
            is_processing: false,
            should_quit: false,
            vertical_scroll: 0,

            agent_cmd_sender: cmd_sender,

            event_receiver: r,
            event_sender: s,
        }
    }

    pub fn get_ui_event_sender(&self) -> mpsc::Sender<UIEvent> {
        self.event_sender.clone()
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) {
        info!("start running App loop");
        terminal.draw(|f| self.ui(f)).expect("failed to draw ui");
        while let Some(event) = self.event_receiver.recv().await {
            match event {
                UIEvent::WindowEvent(window_event) => match window_event {
                    Event::Key(key) => match key.code {
                        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            _ = self.agent_cmd_sender.send(AgentCommand::Stop).await;
                            self.should_quit = true;
                        }
                        KeyCode::Enter => {
                            self.on_send().await;
                        }
                        KeyCode::Up => self.vertical_scroll -= 1,
                        KeyCode::Down => self.vertical_scroll += 1,
                        _ => {
                            self.input.handle_event(&window_event);
                        }
                    },
                    _ => {}
                },
                UIEvent::MessageRecieved(msg) => self.on_msg(msg).await,
            }
            if self.should_quit {
                return;
            }
            terminal.draw(|f| self.ui(f)).expect("failed to draw ui");
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
        if self.input.value() == "" {
            return;
        }

        // we read out the value, then clear the input field
        let content = self.input.value().to_string();
        self.input.reset();

        info!("Sending a prompt request with content: {}", content);

        self.messages.push(UIMessage::TextMessage(UITextMessage {
            author: "User".to_string(),
            content: content.clone(),
        }));

        self.agent_cmd_sender
            .send(AgentCommand::GeneratePrompt(content))
            .await
            .expect("can't reach agent");
        self.is_processing = true;
    }

    fn ui(&mut self, f: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1), Constraint::Length(5)])
            .split(f.area());

        self.render_chat_history(f, chunks[0]);
        self.render_status_text(f, chunks[1]);
        self.render_input(f, chunks[2]);
    }

    fn render_chat_history(&mut self, f: &mut ratatui::Frame, chunk: Rect) {
        let mut text_lines: Vec<ratatui::text::Line> = Vec::new();

        for m in &self.messages {
            text_lines.push(Line::styled(format!("{}: ", m.get_author()), Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)));

            let mes = m.get_text_ref();
            let mut rendered = tui_markdown::from_str(mes);
            text_lines.append(&mut rendered.lines);

            text_lines.push(Line::from(""));
        }

        let messages_paragraph = Paragraph::new(text_lines)
            .block(Block::default().borders(Borders::ALL).title("Chat History"))
            .wrap(Wrap { trim: true })
            .scroll((self.vertical_scroll, 0));

        f.render_widget(messages_paragraph, chunk);
    }

    fn render_status_text(&mut self, f: &mut ratatui::Frame, chunk: Rect) {
        let status_style = if self.is_processing {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default().bg(Color::Blue).fg(Color::White)
        };
        let status_text = format!(" Model: {} | Mode: Standard | [Esc] Quit ", self.current_model);
        let status_bar = Paragraph::new(status_text)
            .style(status_style)
            .alignment(ratatui::layout::Alignment::Center);

        f.render_widget(status_bar, chunk);
    }

    fn render_input(&mut self, f: &mut ratatui::Frame, chunk: Rect) {
        let width = chunk.width.max(3) - 3;
        let scroll = self.input.visual_scroll(width as usize);
        let input_widget = Paragraph::new(self.input.value())
            .style(Style::default())
            .scroll((0, scroll as u16))
            .block(Block::bordered().title("Input"));

        f.render_widget(input_widget, chunk);
    }
}
