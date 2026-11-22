#!/bin/sh
# Cloudflare Pages build command: sh scripts/cloudflare-build.sh
# Output directory: dist
set -eu

WASM_BINDGEN_VERSION=0.2.130

if ! command -v cargo >/dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
rustup target add wasm32-unknown-unknown

if [ "$(wasm-bindgen --version 2>/dev/null)" != "wasm-bindgen $WASM_BINDGEN_VERSION" ]; then
  name="wasm-bindgen-$WASM_BINDGEN_VERSION-x86_64-unknown-linux-musl"
  curl -sSfL "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/$WASM_BINDGEN_VERSION/$name.tar.gz" |
    tar xz --strip-components=1 -C "$HOME/.cargo/bin" "$name/wasm-bindgen"
fi

npm run build
npm run build:pages
