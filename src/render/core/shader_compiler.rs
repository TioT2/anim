//! Shader compiler implementation module

use std::collections::HashMap;

/// Shader compiler
pub struct ShaderCompiler {
    /// DXC library
    library: hassle_rs::DxcLibrary,

    /// DXC compiler
    compiler: hassle_rs::DxcCompiler,

    /// Instance
    _instance: hassle_rs::Dxc,

    /// Set of builtin shaders
    static_shaders: HashMap<std::path::PathBuf, &'static str>,
}

/// Shader compilation error
#[derive(Debug)]
pub enum ShaderCompilerError {
    /// Cannot find absolute of the path
    InvalidPath(std::io::Error),

    /// Input file not found
    FileNotFound(std::ffi::OsString),

    /// DXC internal error
    DxcError(hassle_rs::HassleError),

    /// Shader compilation error occured
    CompilationError(String),
}

impl From<hassle_rs::HassleError> for ShaderCompilerError {
    fn from(value: hassle_rs::HassleError) -> Self {
        ShaderCompilerError::DxcError(value)
    }
}

impl ShaderCompiler {
    /// Create shader compiler
    pub fn new() -> Result<Self, ShaderCompilerError> {
        let instance = hassle_rs::Dxc::new(None)?;
        let library = instance.create_library()?;
        let compiler = instance.create_compiler()?;

        fn static_path(path: &str) -> Result<std::path::PathBuf, ShaderCompilerError> {
            let path: std::path::PathBuf = path.into();

            std::path::absolute(path).map_err(ShaderCompilerError::InvalidPath)
        }

        let static_shaders = {
            macro_rules! entry {
                ($($name: literal),* $(,)?) => {
                    [$((
                        static_path(concat!("/anim/", $name))?,
                        include_str!(concat!("static/", $name))
                    ),)*]
                };
            }

            HashMap::from_iter(entry!(
                "common.hlsl",
                "matrix_compute.hlsl",
                "model.hlsl",
            ).into_iter())
        };

        Ok(Self {
            _instance: instance,
            library,
            compiler,
            static_shaders
        })
    }

    /// Load shader text
    pub fn load_shader_text(
        &self,
        path: &str
    ) -> Result<std::borrow::Cow<'static, str>, ShaderCompilerError> {
        let path = std::path::absolute(std::path::PathBuf::try_from(path).unwrap())
            .map_err(ShaderCompilerError::InvalidPath)?;

        if let Some(static_shader) = self.static_shaders.get(&path) {
            return Ok(std::borrow::Cow::Borrowed(*static_shader));
        }

        Err(ShaderCompilerError::FileNotFound(path.into_os_string()))
    }

    /// Compile shader from HLSL text to SPIR-V bytecode
    pub fn compile_shader(
        &self,
        path: &str,
        main_fn_name: &str,
        hlsl_profile: &str
    ) -> Result<Vec<u32>, ShaderCompilerError> {
        let text = self.load_shader_text(path)?;
        let source_text_blob = self.library.create_blob_with_encoding_from_str(text.as_ref())?;

        /// DXC include handler
        struct IncludeHandler<'t> {
            /// Underlying compiler
            compiler: &'t ShaderCompiler,
        }

        impl<'t> hassle_rs::DxcIncludeHandler for IncludeHandler<'t> {
            fn load_source(&mut self, filename: String) -> Option<String> {
                self.compiler
                    .load_shader_text(&filename)
                    .map(|c| c.to_string())
                    .ok()
            }
        }

        // Perform compilation
        let result = self.compiler.compile(
            &source_text_blob,
            path,
            main_fn_name,
            hlsl_profile,
            &["-spirv", "-I", "/"],
            Some(&mut IncludeHandler { compiler: self }),
            &[]
        );

        match result {
            Ok(op_result) => {
                let result_blob = op_result.get_result()?;

                let (chunks, _) = result_blob.as_slice::<u8>().as_chunks::<4>();
                let mut result = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    result.push(u32::from_ne_bytes(*chunk));
                }

                Ok(result)
            }
            Err((op_result, _)) => {
                let error_blob = op_result.get_error_buffer()?;
                let error_string = self.library.get_blob_as_string(&error_blob.into())?;

                Err(ShaderCompilerError::CompilationError(error_string))
            }
        }
    }
}
