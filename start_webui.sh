#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8080}"

echo "=========================================================="
echo "  🎙️  Starting VoicePolish Studio WebUI on port $PORT"
echo "  📂 Working Directory: $DIR"
echo "  ⚡ Hardware: AMD Ryzen 7 7700X (AVX-512 SIMD)"
echo "=========================================================="

cd "$DIR"
"$DIR/target/release/voicepolish" serve --port "$PORT" --open
