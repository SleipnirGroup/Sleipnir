use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sleipnir_root = manifest_dir
        .parent()
        .expect("rust/ must sit inside the Sleipnir repo")
        .to_path_buf();

    let disable_diagnostics = env::var("CARGO_FEATURE_DIAGNOSTICS").is_err();

    let mut config = cmake::Config::new(&sleipnir_root);
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("SLEIPNIR_BUILD_BENCHMARKS", "OFF")
        .define("SLEIPNIR_BUILD_EXAMPLES", "OFF")
        .define("SLEIPNIR_BUILD_PYTHON", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("CMAKE_CXX_STANDARD", "23");
    if disable_diagnostics {
        config.define("SLEIPNIR_DISABLE_DIAGNOSTICS", "ON");
    }
    let dst = config.build();

    let build_dir = dst.join("build");
    let install_include = dst.join("include");
    let install_lib = dst.join("lib");

    let eigen_include = find_dep_include(&build_dir, "eigen")
        .unwrap_or_else(|| panic!("could not locate fetched Eigen headers under {}", build_dir.display()));
    let small_vector_include = find_dep_include(&build_dir, "small_vector")
        .unwrap_or_else(|| panic!("could not locate fetched small_vector headers under {}", build_dir.display()));

    let shim_dir = manifest_dir.join("cxx");

    let mut shim_build = cxx_build::bridge("src/ffi.rs");
    shim_build
        .file(shim_dir.join("shim.cpp"))
        .include(&install_include)
        .include(&eigen_include)
        .include(&small_vector_include)
        .include(&shim_dir)
        .flag_if_supported("-std=c++23")
        .flag_if_supported("/std:c++latest")
        .flag_if_supported("-Wno-unused-parameter");
    if disable_diagnostics {
        shim_build.define("SLEIPNIR_DISABLE_DIAGNOSTICS", None);
    }
    shim_build.compile("hafgufa_shim");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cxx/shim.h");
    println!("cargo:rerun-if-changed=cxx/shim.cpp");
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("include").display());
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("src").display());
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("CMakeLists.txt").display());

    println!("cargo:rustc-link-search=native={}", install_lib.display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib64").display());
    println!("cargo:rustc-link-lib=static=Sleipnir");

    link_cxx_stdlib();
}

fn find_dep_include(build_dir: &Path, name_fragment: &str) -> Option<PathBuf> {
    let deps = build_dir.join("_deps");
    if !deps.exists() {
        return None;
    }
    let name_fragment = name_fragment.to_lowercase();
    let entries = std::fs::read_dir(&deps).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = path.file_name()?.to_string_lossy().to_lowercase();
        if file_name.contains(&name_fragment) && file_name.ends_with("-src") {
            let include = path.join("include");
            if include.exists() {
                return Some(include);
            }
            return Some(path);
        }
    }
    None
}

fn link_cxx_stdlib() {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("msvc") {
        // MSVC links the C++ runtime automatically.
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
    }
}
