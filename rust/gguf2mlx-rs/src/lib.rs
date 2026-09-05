mod arch;
pub mod config;
pub mod metadata;
pub mod staging;
pub mod tensors;
pub mod tokenizer;

use pyo3::prelude::*;

#[pyfunction]
fn detect_architecture(general_architecture: Option<String>, general_name: Option<String>) -> String {
    arch::detect_architecture(general_architecture.as_deref(), general_name.as_deref())
}

#[pymodule]
fn gguf2mlx_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_architecture, m)?)?;
    Ok(())
}
