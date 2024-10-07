extern crate build_target;
extern crate cpp_build;
use build_target::Os;
use std::path::Path;

fn link_lib(lib_path: &Path, lib: &str, whole_link: bool) -> Result<(), ()> {
    let so_ext = match build_target::target_os().unwrap() {
        Os::Linux => "so",
        Os::MacOs => "a",
        Os::Windows => "dll",
        _ => panic!("Unsupported OS"),
    };
    let filename = match lib {
        "extension_module" => format!("lib{}.{}", lib, so_ext),
        "qnn_executorch_backend" => format!("lib{}.{}", lib, so_ext),
        _ => format!("lib{}.a", lib),
    };
    if lib_path.join(&filename).exists() {
        if filename.ends_with(so_ext) && so_ext != "a" {
            println!("cargo:rustc-link-lib=dylib={}", lib);
        } else {
            if whole_link {
                println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
            } else {
                println!("cargo:rustc-link-lib=static={}", lib);
            }
        }
        return Ok(());
    }
    Err(())
}

fn main() {
    println!("cargo:rerun-if-changed=src/sampler.rs");
    println!("cargo:rerun-if-changed=src/tensor.rs");
    println!("cargo:rerun-if-changed=src/tensor.hpp");
    println!("cargo:rerun-if-changed=src/module.rs");
    println!("cargo:rerun-if-changed=src/module.hpp");
    println!("cargo:rerun-if-changed=src/method_meta.rs");
    println!("cargo:rerun-if-changed=src/evalue.rs");
    println!("cargo:rerun-if-changed=src/evalue.hpp");
    println!("cargo:rerun-if-changed=src/eterror.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");

    let install_prefix = std::env::var("EXECUTORCH_INSTALL_PREFIX")
        .unwrap_or_else(|_| "executorch/cmake-out".to_string());
    let lib_path = Path::new(&install_prefix).join("lib");

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

    println!("cargo:rustc-link-search=native={}", lib_path.display());

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

    assert!(link_lib(&lib_path, "executorch", true).is_ok());
    if !link_lib(&lib_path, "executorch_no_prim_ops", true).is_ok() {
        assert!(link_lib(&lib_path, "executorch_core", true).is_ok());
    }
    if !link_lib(&lib_path, "extension_module_static", false).is_ok() {
        assert!(link_lib(&lib_path, "extension_module", false).is_ok());
    }
    assert!(link_lib(&lib_path, "extension_data_loader", false).is_ok());

    // Optimized Kernels
    if link_lib(&lib_path, "optimized_native_cpu_ops_lib", true).is_ok() {
        assert!(link_lib(&lib_path, "optimized_kernels", false).is_ok());
        assert!(link_lib(&lib_path, "portable_kernels", false).is_ok());
        // assert!(link_lib(&lib_path, "cpublas", false).is_ok());
        assert!(link_lib(&lib_path, "eigen_blas", false).is_ok());
    } else {
        assert!(link_lib(&lib_path, "portable_ops_lib", true).is_ok());
        assert!(link_lib(&lib_path, "portable_kernels", false).is_ok());
    }

    // Quantized Kernels
    if link_lib(&lib_path, "quantized_ops_lib", true).is_ok() {
        assert!(link_lib(&lib_path, "quantized_kernels", false).is_ok());
    }

    // Custom Ops
    let _ = link_lib(&lib_path, "custom_ops", true);

    // Tensor extension
    let _ = link_lib(&lib_path, "extension_tensor", false);

    // Runner Util extension
    let _ = link_lib(&lib_path, "extension_runner_util", false);

    // misc.
    let _ = link_lib(&lib_path, "cpuinfo", false);
    let _ = link_lib(&lib_path, "pthreadpool", false);

    // XNNPACK
    if link_lib(&lib_path, "xnnpack_backend", true).is_ok() {
        assert!(link_lib(&lib_path, "XNNPACK", false).is_ok());
        assert!(link_lib(&lib_path, "microkernels-prod", false).is_ok());
        let _ = link_lib(&lib_path, "kleidiai", false);
    }

    // Vulkan
    let _ = link_lib(&lib_path, "vulkan_backend", true);

    // QNN
    let _ = link_lib(&lib_path, "qnn_executorch_backend", true);

    cpp_build::Config::new()
        .flag("-std=c++17")
        .build("src/lib.rs");
}
