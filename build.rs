extern crate cpp_build;
use std::path::Path;

#[cfg(target_os = "macos")]
static SO_EXT: &str = "dylib";
#[cfg(target_os = "linux")]
static SO_EXT: &str = "so";
#[cfg(target_os = "windows")]
static SO_EXT: &str = "dll";

static ET_LIBS: [&str; 16] = [
    "extension_data_loader",
    "mpsdelegate",
    "qnn_executorch_backend",
    "portable_ops_lib",
    "extension_module",
    "xnnpack_backend",
    "XNNPACK",
    "cpuinfo",
    "pthreadpool",
    "vulkan_backend",
    "optimized_kernels",
    "optimized_ops_lib",
    "optimized_native_cpu_ops_lib",
    "quantized_kernels",
    "quantized_ops_lib",
    "custom_ops",
];

fn main() {
    // find available lib / object files in the executorch cmake-out
    // all libs should whole-archive link
    // libs:
    // extension_data_loader
    // mpsdelegate
    // qnn_executorch_backend : shared
    // portable_ops_lib
    // extension_module : shared
    // xnnpack_backend
    // XNNPACK
    // cpuinfo
    // pthreadpool
    // vulkan_backend
    // optimized_kernels
    // optimized_ops_lib
    // optimized_native_cpu_ops_lib
    // quantized_kernels
    // quantized_ops_lib
    // custom_ops
    // 

    // from env EXECUTORCH_INSTALL_PREFIX default "executorch/cmake-out/lib"
    let base_path_str = std::env::var("EXECUTORCH_INSTALL_PREFIX").unwrap_or_else(|_| "executorch/cmake-out".to_string());
    let base_path = Path::new(&base_path_str).join("lib");

    // find path in executorch/cmake-out/lib/*.{so,dylib,dll,a}
    let mut lib_paths: Vec<&str> = Vec::new();
    for lib in ET_LIBS {
        let path = match lib {
            "extension_module" => format!("lib{}.{}", lib, SO_EXT),
            "qnn_executorch_backend" => format!("lib{}.{}", lib, SO_EXT),
            _ => format!("lib{}.a", lib),
        };
        if base_path.join(&path).exists() {
            lib_paths.push(lib.trim_start_matches("lib"));
        }
    }

    assert!(!lib_paths.is_empty(), "No lib files found in executorch/cmake-out/lib");

    let mut config = cpp_build::Config::new();
    config.flag_if_supported("-std=c++17");

    for lib in lib_paths {
        // POSIX systems
        config.flag_if_supported(&format!("--whole-archive -l{} --no-whole-archive", lib));
        // Darwin systems
        config.flag_if_supported(&format!("-force_load,{}", lib));
    }

    config.build("src/lib.rs");
}
