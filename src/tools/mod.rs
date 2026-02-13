use schemars::{JsonSchema, schema::RootSchema};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::pin::Pin;
use thiserror::Error;

pub mod read_file;

pub trait Tool: Send + Sync {
    const NAME: &str;
    const DESCRIPTION: &str;
    type Arguments: JsonSchema + DeserializeOwned + Send;

    fn run_tool(
        &self,
        args: Self::Arguments,
    ) -> Pin<Box<dyn Future<Output = Result<String, RunToolError>> + Send>>;

    fn get_argument_schema(&self) -> RootSchema {
        schemars::schema_for!(Self::Arguments)
    }
}

#[derive(Error, Debug)]
pub enum RunToolError {
    #[error(
        "Deserializing Arguments for {tool_name} tool failed. Expected Arguments with Schema: {expected_schema}: {serde_error}"
    )]
    InvalidArguments {
        tool_name: String,
        expected_schema: String,
        serde_error: String,
    },
}

pub trait DynTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> RootSchema;

    // Takes generic JSON, returns a Future
    fn run(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, RunToolError>> + Send + '_>>;
}

impl<T: Tool> DynTool for T {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn description(&self) -> &'static str {
        Self::DESCRIPTION
    }
    fn schema(&self) -> RootSchema {
        self.get_argument_schema()
    }

    fn run(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, RunToolError>> + Send + '_>> {
        let argument_schema = self.get_argument_schema();
        Box::pin(async move {
            let args: T::Arguments =
                serde_json::from_value(input).map_err(|e| RunToolError::InvalidArguments {
                    tool_name: Self::NAME.to_string(),
                    expected_schema: serde_json::to_string(&argument_schema)
                        .expect("no hashmap with non string keys"),
                    serde_error: e.to_string(),
                })?;
            self.run_tool(args).await
        })
    }
}
