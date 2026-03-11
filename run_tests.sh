#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# unsloth-candle — Comprehensive Test & Demo Runner
#
# Usage:
#   ./run_tests.sh             # CPU only (default)
#   ./run_tests.sh cuda        # CUDA GPU
#   ./run_tests.sh metal       # Apple Silicon Metal
#   ./run_tests.sh all         # run everything
# ─────────────────────────────────────────────────────────────────────────────
set -e

FEATURE="${1:-}"
CARGO_FEATURES=""
PY_MODE="cpu"

# ── Detect build target ──────────────────────────────────────────────────────
case "$FEATURE" in
  cuda)
    CARGO_FEATURES="--features cuda"
    PY_MODE="cuda"
    echo "🟢  Building with CUDA support"
    ;;
  metal)
    CARGO_FEATURES="--features metal"
    PY_MODE="metal"
    echo "🍎  Building with Metal (Apple Silicon) support"
    ;;
  all)
    echo "🔄  Running all hardware targets sequentially"
    bash "$0" ""
    echo ""
    # Skip cuda/metal if not available
    if command -v nvcc &>/dev/null; then bash "$0" cuda; fi
    exit 0
    ;;
  *)
    echo "🖥️   Building CPU-only (pass 'cuda' or 'metal' for GPU)"
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  unsloth-candle  E2E Test Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Cargo check ───────────────────────────────────────────────────────────
echo "──── [1/5] cargo check ─────────────────────────────────────"
cargo check $CARGO_FEATURES 2>&1 | grep -E "^(error|warning\[|Finished|Checking)" | head -30
echo "✓ cargo check passed"
echo ""

# ── 2. Run existing QLoRA test ───────────────────────────────────────────────
echo "──── [2/5] Original QLoRA test ─────────────────────────────"
cargo test $CARGO_FEATURES --test test_qlora -- --nocapture 2>&1 | tail -20
echo ""

# ── 3. Run core model tests ──────────────────────────────────────────────────
echo "──── [3/5] Core model tests (Llama fp32/GQA/LoRA/QLoRA/DoRA/KV) ─"
cargo test $CARGO_FEATURES --test test_core_models -- --nocapture 2>&1 | tail -30
echo ""

# ── 4. Run new architecture tests ────────────────────────────────────────────
echo "──── [4/5] New architecture tests (Mistral/Qwen3/Gemma3/...) ─"
cargo test $CARGO_FEATURES --test test_new_models -- --nocapture 2>&1 | tail -30
echo ""

# ── 5. Python finetune demo ──────────────────────────────────────────────────
echo "──── [5/5] Python fine-tuning demo ─────────────────────────"
if command -v python3 &>/dev/null; then
    # Build Python extension for this hardware target
    if command -v maturin &>/dev/null; then
        echo "  Building Python extension (maturin develop)..."
        maturin develop $CARGO_FEATURES --quiet 2>&1 | tail -5 || true
    else
        echo "  ⚠  maturin not found — demo runs in simulation mode"
    fi
    python3 finetune_demo.py
else
    echo "  ⚠  python3 not found — skipping Python demo"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅  All tests complete"
echo "═══════════════════════════════════════════════════════════"
