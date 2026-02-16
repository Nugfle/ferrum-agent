use ratatui::{
    Terminal,
    crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::CrosstermBackend,
};
use std::{collections::HashMap, io};
use tracing::info;
use tracing_appender::rolling;

use crate::{ollama::agent::OllamaAgent, ui::App};

mod ollama;
mod tools;
mod ui;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_appender = rolling::daily("logs", "ferrum.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt().with_writer(non_blocking).with_ansi(false).init();

    let tools = HashMap::new();
    let (cmd_sender, _) = OllamaAgent::new("http://localhost:11434".to_string(), tools, "my-coder".to_string());

    info!("loaded tools and started agent!");

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    info!("Initialized terminal!");

    let mut app = App::new(cmd_sender).await;
    app.run(&mut terminal).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}
