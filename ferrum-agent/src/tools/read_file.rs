use schemars::JsonSchema;
use serde::Deserialize;
use tokio::{fs::File, io::AsyncReadExt};

use crate::tools::Tool;

pub struct FileReaderTool;

#[derive(Debug, JsonSchema, Deserialize, Clone)]
pub struct ReadFileArgs {
    path: String,
}

impl Tool for FileReaderTool {
    type Arguments = ReadFileArgs;
    type ToolError = tokio::io::Error;
    const NAME: &str = "read_file";
    const DESCRIPTION: &str = "Read the contents of the file provided by path.";

    fn run_tool(&self, args: Self::Arguments) -> std::pin::Pin<Box<dyn Future<Output = Result<String, Self::ToolError>> + Send>> {
        Box::pin(async move {
            let mut f = File::open(args.path).await?;
            let mut buf = String::new();
            f.read_to_string(&mut buf).await?;
            Ok(buf)
        })
    }
}
