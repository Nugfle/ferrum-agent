use tokio::{fs::File, io::AsyncReadExt};

use crate::tools::Tool;

pub struct FileReaderTool;

impl Tool for FileReaderTool {
    type Arguments = String;
    type ToolError = tokio::io::Error;
    const NAME: &str = "read_file";
    const DESCRIPTION: &str = "This tool allows you to open a file and read its content. It requires the path as an argument";

    fn run_tool(&self, path: Self::Arguments) -> std::pin::Pin<Box<dyn Future<Output = Result<String, Self::ToolError>> + Send>> {
        Box::pin(async move {
            let mut f = File::open(path).await?;
            let mut buf = String::new();
            f.read_to_string(&mut buf).await?;
            Ok(buf)
        })
    }
}
