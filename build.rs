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
    let base_path_str = std::env::var("EXECUTORCH_INSTALL_PREFIX").unwrap_or_else(|_| "executorch/cmake-out".to_string());
    let lib_path = Path::new(&base_path_str).join("lib");

    let mut lib_paths: Vec<String> = Vec::new();
    for lib in ET_LIBS {
        let filename = match lib {
            "extension_module" => format!("lib{}.{}", lib, SO_EXT),
            "qnn_executorch_backend" => format!("lib{}.{}", lib, SO_EXT),
            _ => format!("lib{}.a", lib),
        };
        if lib_path.join(&filename).exists() {
            lib_paths.push(filename);
        }
    }

    assert!(!lib_paths.is_empty(), "No lib files found in executorch/cmake-out/lib");

    let mut config = cpp_build::Config::new();
    config.flag("-std=c++17");

    for lib in lib_paths {
        config.flag(&format!("-Wl,--whole-archive -l{} -Wl,--no-whole-archive", lib.trim_start_matches("lib").trim_end_matches(".a")));
        // if lib.ends_with(SO_EXT) {
        //     // println!("cargo:rustc-link-lib=dylib={}", lib.trim_start_matches("lib").trim_end_matches(SO_EXT));
        // } else {
        //     config.flag(&format!("--whole-archive -l{} --no-whole-archive", lib));
        //     // println!("cargo:rustc-link-lib=static={}", lib.trim_start_matches("lib").trim_end_matches(".a"));
        // }
    }
    // println!("cargo:rustc-link-search=native={}", lib_path.display());

    config.flag(&format!("-L{}", lib_path.display()));
    
    config.build("src/lib.rs");
}
