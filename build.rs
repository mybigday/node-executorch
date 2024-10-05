extern crate build_target;
extern crate cpp_build;
use std::path;
use build_target::{Arch, Os};
use cmake::Config;

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

    // // let executorch = cmake::build(".");
    // let mut et = Config::new("executorch");
    // // debug flags
    // if std::env::var("PROFILE").unwrap() == "debug" {
    //     et.define("EXECUTORCH_ENABLE_LOGGING", "ON");
    //     et.define("EXECUTORCH_BUILD_XNNPACK", "ON");
    //     et.define("EXECUTORCH_BUILD_EXTENSION_DATA_LOADER", "ON");
    //     et.define("EXECUTORCH_BUILD_EXTENSION_MODULE", "ON");
    // } else {
    //     et.define("EXECUTORCH_BUILD_EXTENSION_DATA_LOADER", "ON");
    //     et.define("EXECUTORCH_BUILD_EXTENSION_MODULE", "ON");
    //     et.define("EXECUTORCH_BUILD_KERNELS_CUSTOM", "ON");
    //     et.define("EXECUTORCH_BUILD_KERNELS_QUANTIZED", "ON");
    //     et.define("EXECUTORCH_BUILD_KERNELS_OPTIMIZED", "ON");
    //     et.define("EXECUTORCH_BUILD_EXTENSION_TENSOR", "ON");
    //     et.define("EXECUTORCH_BUILD_CPUINFO", "ON");
    //     et.define("EXECUTORCH_BUILD_XNNPACK", "ON");
    //     et.define("EXECUTORCH_BUILD_PTHREADPOOL", "ON");
    //     et.define("EXECUTORCH_BUILD_SDK", "ON");
    //     et.define("EXECUTORCH_ENABLE_LOGGING", "ON");
    //     if os == Os::Windows && arch == Arch::AARCH64 {
    //         if let Ok(qnn_sdk_root) = std::env::var("QNN_SDK_ROOT") {
    //             et.define("QNN_SDK_ROOT", qnn_sdk_root);
    //             et.define("EXECUTORCH_BUILD_QNN", "ON");
    //         }
    //     }
    //     if os == Os::MacOs {
    //         et.define("EXECUTORCH_BUILD_COREML", "ON");
    //     }
    // }
    // // resolve toolchain file absolute path
    // // project_root::get_project_root().unwrap() + cmake/mingw-w64-aarch64.clang.toolchain.cmake
    // let toolchain_file = Path::new(&project_root::get_project_root().unwrap()).join("cmake/mingw-w64-aarch64.clang.toolchain.cmake");
    // et.define("CMAKE_TOOLCHAIN_FILE", toolchain_file.display().to_string());
    // let et_path = et.build();
    // println!("cargo:rustc-link-search=native={}/lib", et_path.display());

    let build_script = path::Path::new(&project_root::get_project_root().unwrap()).join("scripts/build.sh");
    assert!(build_script.exists(), "build.sh not found");
    let build_script_path = build_script.to_str().unwrap();
    let out_dir = path::absolute("build").unwrap().display().to_string();

    // if env has no EXECUTORCH_INSTALL_PREFIX, run scripts/build.sh
    if std::env::var("EXECUTORCH_INSTALL_PREFIX").is_err() {
        let mut cmd = std::process::Command::new(build_script_path);
        cmd.args(&[node_platform, node_arch, &out_dir]);
        cmd.status().unwrap();
    }

    let et_path = std::env::var("EXECUTORCH_INSTALL_PREFIX").unwrap_or_else(|_| out_dir);

    println!("cargo:rustc-link-search=native={}/lib", et_path);

    let et_all = Config::new(".")
        .define("EXECUTORCH_INSTALL_PATH", et_path)
        .build();
    println!("cargo:rustc-link-search=native={}/lib", et_all.display());
    println!("cargo:rustc-link-lib=static=executorch_all");

    // let install_prefix = std::env::var("EXECUTORCH_INSTALL_PREFIX")
    //     .unwrap_or_else(|_| "executorch/cmake-out".to_string());
    // let lib_path = Path::new(&install_prefix).join("lib");

    // println!("cargo:rustc-link-search=native={}", lib_path.display());

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
