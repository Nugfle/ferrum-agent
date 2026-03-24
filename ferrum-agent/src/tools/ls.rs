use schemars::JsonSchema;
use serde::Deserialize;
use tokio::fs;
use tracing::info;

use crate::tools::Tool;

pub struct ListFilesTool;

#[derive(Debug, JsonSchema, Deserialize, Clone)]
pub struct ListFileArgs {
    path: String,
}

impl Tool for ListFilesTool {
    type Arguments = ListFileArgs;
    type ToolError = tokio::io::Error;
    const NAME: &str = "list_directory";
    const DESCRIPTION: &str = "List all files and directories at the provided path. Directories are marked by a trailing slash.";

    fn run_tool(&self, args: Self::Arguments) -> std::pin::Pin<Box<dyn Future<Output = Result<String, Self::ToolError>> + Send>> {
        Box::pin(async move {
            let mut entries = String::from("Directory entries: ");
            info!("listing directory entries in path: '{}'", &args.path);
            let mut dir = fs::read_dir(&args.path).await?;
            while let Some(entry) = dir.next_entry().await? {
                if entry.file_type().await?.is_dir() {
                    entries.push_str(&format!("{}/, ", entry.file_name().display()));
                } else {
                    entries.push_str(&format!("{}, ", entry.file_name().display()));
                }
            }
            Ok(entries.trim_end_matches(", ").to_string())
        })
    }
}
