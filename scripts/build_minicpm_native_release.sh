#!/usr/bin/env bash
# Reproducible serial Release build for the native MiniCPM server.
# Swift 6.3 can inherit stale module-map search paths from a shared scratch
# directory, so this script keeps the build isolated and supplies all module
# map parents explicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SCRATCH_PATH="${MINICPM_NATIVE_SCRATCH_PATH:-$PWD/.build-minicpm-release-native}"
PRODUCT="${MINICPM_NATIVE_PRODUCT:-minicpm-mlx-server}"
module_include_args=()

while IFS= read -r module_map; do
  module_include_args+=("-Xswiftc" "-I" "-Xswiftc" "$(dirname "$module_map")")
done < <(find "$SCRATCH_PATH/checkouts" -name module.modulemap -print 2>/dev/null)

while IFS= read -r module_map; do
  module_include_args+=("-Xswiftc" "-I" "-Xswiftc" "$(dirname "$module_map")")
done < <(find "$SCRATCH_PATH/arm64-apple-macosx/release" \
  -path '*.build/module.modulemap' -print 2>/dev/null)

build_args=(
  -j 1
  -c release
  --scratch-path "$SCRATCH_PATH"
  --product "$PRODUCT"
)
if ((${#module_include_args[@]} > 0)); then
  build_args+=("${module_include_args[@]}")
fi
swift build "${build_args[@]}"

# MLX resolves its precompiled kernels next to the executable. SwiftPM does
# not copy this resource itself, so a newly isolated Release directory would
# otherwise build successfully and then fail at startup with "default
# metallib ... not found".
"$SCRIPT_DIR/build_mlx_metallib.sh" release --scratch-path "$SCRATCH_PATH"

echo "[minicpm-native] release product:"
echo "  $SCRATCH_PATH/arm64-apple-macosx/release/$PRODUCT"
