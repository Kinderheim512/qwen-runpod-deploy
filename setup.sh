#!/usr/bin/env bash
#
# setup.sh — Déploiement rapide de Qwen3.8-27B sur un Pod RunPod (A6000 48GB)
#
# Usage:
#   bash setup.sh                 # menu interactif
#   bash setup.sh full            # Qwen3.8-27B-FP8 officiel via vLLM
#   bash setup.sh light           # Qwen3.8-27B quantifié 4-bit via Ollama
#   bash setup.sh uncensored      # Qwen3.8-27B-Uncensored (abliterated) via Ollama
#
# Pensé pour tourner sur un Pod RunPod avec image de base "RunPod PyTorch 2.x"
# et un GPU A6000 (48GB VRAM). Doit être lancé en root ou avec sudo.

set -euo pipefail

# ------------------------------------------------------------------
# Paramètres
# ------------------------------------------------------------------
PORT="${PORT:-8000}"
MAX_CTX="${MAX_CTX:-131072}"          # 128k par défaut, le modèle supporte jusqu'à 262k natif
WORKDIR="${WORKDIR:-/workspace}"
MODEL_CHOICE="${1:-}"

# ------------------------------------------------------------------
# Vérification GPU
# ------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
  echo "❌ nvidia-smi introuvable. Ce script doit tourner sur un Pod GPU RunPod."
  exit 1
fi

echo "GPU détecté :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ------------------------------------------------------------------
# Menu interactif si aucun argument n'est passé
# ------------------------------------------------------------------
if [[ -z "$MODEL_CHOICE" ]]; then
  echo ""
  echo "Quelle variante de Qwen3.8-27B installer ?"
  echo "  1) full        - Qwen3.8-27B-FP8 officiel (vLLM, ~28GB VRAM, meilleure qualité/débit)"
  echo "  2) light       - Qwen3.8-27B GGUF Q4_K_M via Ollama (~17GB VRAM, léger, rapide à charger)"
  echo "  3) uncensored  - Qwen3.8-27B-Uncensored (abliterated) via Ollama (~17GB VRAM)"
  read -rp "Votre choix [1-3]: " CHOICE_NUM
  case "$CHOICE_NUM" in
    1) MODEL_CHOICE="full" ;;
    2) MODEL_CHOICE="light" ;;
    3) MODEL_CHOICE="uncensored" ;;
    *) echo "Choix invalide."; exit 1 ;;
  esac
fi

# ------------------------------------------------------------------
# Option 1 : version complète FP8 via vLLM
# ------------------------------------------------------------------
install_full() {
  echo "▶ Installation de Qwen3.8-27B-FP8 via vLLM (~28GB VRAM)"
  pip install --upgrade pip
  pip install "vllm>=0.9" --extra-index-url https://download.pytorch.org/whl/cu121

  cat > "$WORKDIR/start_vllm.sh" <<EOF
#!/usr/bin/env bash
vllm serve Qwen/Qwen3.8-27B-FP8 \\
  --host 0.0.0.0 \\
  --port ${PORT} \\
  --gpu-memory-utilization 0.90 \\
  --max-model-len ${MAX_CTX} \\
  --kv-cache-dtype fp8 \\
  --reasoning-parser qwen3 \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder
EOF
  chmod +x "$WORKDIR/start_vllm.sh"
  echo "✅ Installation terminée. Lancez le serveur avec : $WORKDIR/start_vllm.sh"
  echo "   API compatible OpenAI disponible sur : http://0.0.0.0:${PORT}/v1"
}

# ------------------------------------------------------------------
# Option 2 : version légère GGUF via Ollama
# ------------------------------------------------------------------
install_light() {
  echo "▶ Installation de Qwen3.8-27B quantifié (Q4_K_M, ~17GB VRAM) via Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
  (ollama serve &>/tmp/ollama.log &)
  sleep 5
  ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M
  echo "✅ Installation terminée. Lancez avec : ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M"
}

# ------------------------------------------------------------------
# Option 3 : version uncensored (abliterated) via Ollama
# ------------------------------------------------------------------
install_uncensored() {
  cat <<'WARN'
⚠️  Vous installez une version "uncensored" (abliterated) de Qwen3.8-27B.
    Les garde-fous de sécurité du modèle ont été retirés par un tiers
    (orthogonalisation de la direction de refus). Ce modèle peut produire
    du contenu que le modèle de base refuserait normalement.
    Vous êtes seul responsable de l'usage que vous en faites et devez
    respecter la loi en vigueur dans votre juridiction.
WARN
  read -rp "Continuer ? [o/N] " CONFIRM
  if [[ ! "$CONFIRM" =~ ^[oOyY]$ ]]; then
    echo "Annulé."
    exit 0
  fi

  echo "▶ Installation de Qwen3.8-27B-Uncensored (Q4_K_M, ~17GB VRAM) via Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
  (ollama serve &>/tmp/ollama.log &)
  sleep 5
  ollama pull hf.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M
  echo "✅ Installation terminée. Lancez avec : ollama run hf.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "$MODEL_CHOICE" in
  full)        install_full ;;
  light)       install_light ;;
  uncensored)  install_uncensored ;;
  *)
    echo "❌ Choix inconnu : $MODEL_CHOICE (attendu: full | light | uncensored)"
    exit 1
    ;;
esac
