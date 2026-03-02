use ratatui::{
    Terminal,
    crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::{Backend, CrosstermBackend},
};
use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{self, Read, Stdout, Write},
};
use tracing::info;
use tracing_appender::rolling;

use crate::{
    ollama::agent::OllamaAgent,
    tools::{DynTool, read_file::FileReaderTool},
    ui::App,
};

mod ollama;
mod tools;
mod ui;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_appender = rolling::daily("logs", "ferrum.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt().with_writer(non_blocking).with_ansi(false).init();

    let system_prompt = load_system_prompt();
    let tools = setup_tools();

    let (cmd_sender, _) = OllamaAgent::new("http://localhost:11434".to_string(), tools, "qwen3.5:397b-cloud".to_string(), Some(system_prompt));
    info!("loaded tools and started agent!");

    let mut terminal = setup_terminal()?;
    info!("Initialized terminal!");

    let mut app = App::new(cmd_sender).await;
    app.run(&mut terminal).await;

    info!("The UI has been stopped. Starting cleanup...");
    cleanup(terminal)
}

fn load_system_prompt() -> String {
    let mut system_prompt = String::new();
    _ = File::open("./SOUL.md")
        .expect("SOUL.md not found")
        .read_to_string(&mut system_prompt)
        .unwrap();
    system_prompt
}

fn setup_tools() -> HashMap<String, Box<dyn DynTool>> {
    let mut tools: HashMap<String, Box<dyn DynTool>> = HashMap::new();

    let fr_tool = FileReaderTool;
    tools.insert(fr_tool.name().to_string(), Box::new(fr_tool));

    tools
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>, Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn cleanup<B: Backend + Write>(mut terminal: Terminal<B>) -> Result<(), Box<dyn Error>>
where
    <B as Backend>::Error: 'static,
{
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}
