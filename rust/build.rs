use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sleipnir_root = manifest_dir
        .parent()
        .expect("rust/ must sit inside the Sleipnir repo")
        .to_path_buf();

    // Sleipnir — both the C++ library and this Rust binding — requires
    // a C++23 toolchain. `<print>` is the canonical smoke test: it
    // ships in GCC 14+ libstdc++ and libc++ 19+. Probe it up front so
    // the user gets a single clear error instead of a compiler
    // diagnostic buried 500 lines into a cmake build log.
    if !probe_cpp23() {
        panic!(
            "hafgufa requires a C++23 toolchain. Your `{}` compiler \
             can't compile `#include <print>` with `-std=c++23` — \
             upgrade to GCC 14+ / libstdc++ 14+ / libc++ 19+ / MSVC \
             19.37+, or set `$CXX` to a compiler that supports C++23.",
            env::var("CXX").unwrap_or_else(|_| "c++".into())
        );
    }

    let mut config = cmake::Config::new(&sleipnir_root);
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("SLEIPNIR_BUILD_BENCHMARKS", "OFF")
        .define("SLEIPNIR_BUILD_EXAMPLES", "OFF")
        .define("SLEIPNIR_BUILD_PYTHON", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("CMAKE_CXX_STANDARD", "23");
    let dst = config.build();

    let build_dir = dst.join("build");
    let install_include = dst.join("include");
    let install_lib = dst.join("lib");

    let eigen_include = find_dep_include(&build_dir, "eigen").unwrap_or_else(|| {
        panic!(
            "could not locate fetched Eigen headers under {}",
            build_dir.display()
        )
    });
    let small_vector_include = find_dep_include(&build_dir, "small_vector").unwrap_or_else(|| {
        panic!(
            "could not locate fetched small_vector headers under {}",
            build_dir.display()
        )
    });

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
    shim_build.compile("hafgufa_shim");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cxx/shim.h");
    println!("cargo:rerun-if-changed=cxx/shim.cpp");
    println!(
        "cargo:rerun-if-changed={}",
        sleipnir_root.join("include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        sleipnir_root.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        sleipnir_root.join("CMakeLists.txt").display()
    );

    println!("cargo:rustc-link-search=native={}", install_lib.display());
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib64").display()
    );
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

/// Check that the host C++ toolchain supports C++23 by compiling a
/// trivial `#include <print>` translation unit with `-std=c++23`.
/// `<print>` is a C++23 stdlib addition (GCC 14+ libstdc++, libc++ 19+,
/// MSVC 19.37+) — using it as a proxy catches the common case where
/// the compiler frontend accepts `-std=c++23` but the bundled stdlib
/// is too old.
fn probe_cpp23() -> bool {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let probe_src = out_dir.join("probe_print.cc");
    let probe_obj = out_dir.join("probe_print.o");
    if std::fs::write(&probe_src, "#include <print>\nint main(){}\n").is_err() {
        return false;
    }

    // Pick the compiler the same way cmake/cc-rs would: honour $CXX,
    // fall back to the target-default.
    let cxx = env::var("CXX").unwrap_or_else(|_| {
        let target = env::var("TARGET").unwrap_or_default();
        if target.contains("msvc") {
            "cl".to_string()
        } else if target.contains("apple") {
            "clang++".to_string()
        } else {
            "c++".to_string()
        }
    });

    let ok = Command::new(&cxx)
        .arg("-std=c++23")
        .arg("-c")
        .arg(&probe_src)
        .arg("-o")
        .arg(&probe_obj)
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let _ = std::fs::remove_file(&probe_src);
    let _ = std::fs::remove_file(&probe_obj);
    ok
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
