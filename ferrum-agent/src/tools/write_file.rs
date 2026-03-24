use std::default;

use schemars::JsonSchema;
use serde::Deserialize;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncWriteExt},
};

use crate::tools::Tool;

pub struct WriteFileTool;

#[derive(Debug, JsonSchema, Deserialize, Clone, Default)]
pub enum WriteMode {
    Append,
    #[default]
    Overwrite,
}

#[derive(Debug, JsonSchema, Deserialize, Clone)]
pub struct WriteFileArgs {
    path: String,
    content: String,
    #[serde(default)]
    mode: WriteMode,
}

impl Tool for WriteFileTool {
    type Arguments = WriteFileArgs;
    type ToolError = tokio::io::Error;
    const NAME: &str = "write_file";
    const DESCRIPTION: &str = "Append content to an existing file or overwrite/create a new file.";

    fn run_tool(&self, args: Self::Arguments) -> std::pin::Pin<Box<dyn Future<Output = Result<String, Self::ToolError>> + Send>> {
        Box::pin(async move {
            let mut f = match args.mode {
                WriteMode::Append => File::options().append(true).open(args.path).await?,
                WriteMode::Overwrite => File::options().write(true).create(true).open(args.path).await?,
            };
            f.write_all(args.content.as_bytes()).await?;
            Ok("Successfully updated the file".to_string())
        })
    }
}
