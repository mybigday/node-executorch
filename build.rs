extern crate cpp_build;
use std::path::Path;
use std::fs;

fn link_shared(lib: &str) {
    println!("cargo:rustc-link-lib=dylib={}", lib);
}

fn link_whole(lib: &str) {
    println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
}

fn link(lib: &str) {
    println!("cargo:rustc-link-lib=static={}", lib);
}

fn main() {
    println!("cargo:rerun-if-changed=executorch/version.txt");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/");

    let node_platform = match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "linux" => "linux",
        "macos" => "darwin",
        "windows" => "win32",
        _ => panic!("Unsupported platform"),
    };
    let node_arch = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        _ => panic!("Unsupported arch"),
    };

    let install_path = cmake::build(".");
    println!("cargo:rustc-link-search=native={}/lib", install_path.display());

    if node_platform == "win32" {
        link("c++");
        link("c++abi");
        link("unwind");
    }

    // common
    link_whole("executorch");
    link("executorch_core");
    link("extension_module_static");
    link("extension_data_loader");
    link("tokenizer");
    link("sampler");

    // XNNPACK
    link_whole("xnnpack_backend");
    link("XNNPACK");
    link("microkernels-prod");
    link("cpuinfo");
    link("pthreadpool");

    if cfg!(debug_assertions) {
        link_whole("portable_ops_lib");
        link("portable_kernels");
    } else {
        link("extension_tensor");
        link("extension_threadpool");

        // Optimized Ops
        link_whole("optimized_native_cpu_ops_lib");
        link("optimized_kernels");
        link("portable_kernels");
        link("eigen_blas");

        // Quantized Ops
        link_whole("quantized_ops_lib");
        link("quantized_kernels");

        // Custom Ops
        link_whole("custom_ops");

        // XNNPACK extra
        if node_platform != "darwin" && node_arch == "arm64" {
            link_whole("kleidiai");
        }

        // CoreML
        if node_platform == "darwin" {
            link_whole("coremldelegate");
            link("sqlite3");
            println!("cargo:rustc-link-arg=-framework Foundation");
            println!("cargo:rustc-link-arg=-framework CoreML");
            println!("cargo:rustc-link-arg=-framework Accelerate");
        }

        // QNN
        if node_platform == "win32" && node_arch == "arm64" && std::env::var("QNN_SDK_ROOT").is_ok() {
            println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");
            link_shared("qnn_executorch_backend");
            let src = Path::new(&install_path).join("lib/qnn_executorch_backend.dll");
            let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".to_string());
            let dst = Path::new(&out_dir).join("bin").join(node_platform).join(node_arch).join("qnn_executorch_backend.dll");
            fs::copy(src, dst).unwrap();
        }
    }

    // for nodejs/electron usage
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,bin/{}/{}",
        node_platform, node_arch
    );
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,node_modules/bin/{}/{}",
        node_platform, node_arch
    );
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,resources/node_modules/bin/{}/{}",
        node_platform, node_arch
    );

    cpp_build::Config::new()
        .flag("-std=c++17")
        .build("src/lib.rs");
}
