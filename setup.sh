#!/usr/bin/env bash
#
# setup.sh — Déploiement rapide de Qwen3.8-27B sur un Pod RunPod (A6000 48GB)
#
# Usage:
#   bash setup.sh                     # menu interactif
#   bash setup.sh full                # Qwen3.8-27B-FP8 officiel via vLLM
#   bash setup.sh light               # Qwen3.8-27B quantifié 4-bit via Ollama
#   bash setup.sh uncensored          # Qwen3.8-27B-Uncensored (abliterated) via Ollama
#   bash setup.sh full openfox        # idem + interface OpenFox connectée dessus
#   bash setup.sh light openfox       # idem
#   bash setup.sh uncensored openfox  # idem
#   bash setup.sh openfox             # installe seulement OpenFox (modèle déjà en place)
#
# Variables d'environnement optionnelles :
#   HF_TOKEN      token Hugging Face (recommandé, évite le rate-limit 429 sur les téléchargements)
#   PORT          port de l'API vLLM (défaut 8000)
#   MAX_CTX       longueur de contexte max (défaut 131072)
#   WORKDIR       dossier de travail persistant (défaut /workspace)
#   OPENFOX_PORT  port de l'interface OpenFox (défaut 10369)
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
HF_TOKEN="${HF_TOKEN:-}"
OPENFOX_PORT="${OPENFOX_PORT:-10369}"
MODEL_CHOICE="${1:-}"
EXTRA="${2:-}"

# ------------------------------------------------------------------
# Outils système requis (nvidia-smi a besoin de pciutils dans
# certaines images RunPod minimalistes, sinon il échoue silencieusement)
# ------------------------------------------------------------------
echo "▶ Vérification des paquets système (pciutils, curl, git)..."
apt-get update -qq
apt-get install -y -qq pciutils curl git >/dev/null

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
# Login Hugging Face si un token est fourni (évite le rate-limit 429)
# ------------------------------------------------------------------
if [[ -n "$HF_TOKEN" ]]; then
  pip install -q -U huggingface_hub
  hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  echo "✅ Connecté à Hugging Face avec le token fourni."
else
  echo "ℹ️  Aucun HF_TOKEN fourni. Ça fonctionne en général, mais si vous êtes"
  echo "   rate-limité (erreur 429), relancez avec : HF_TOKEN=hf_xxx bash setup.sh ..."
fi

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

  read -rp "Installer aussi OpenFox (interface de codage agentique) ? [o/N] " OPENFOX_ANSWER
  if [[ "$OPENFOX_ANSWER" =~ ^[oOyY]$ ]]; then
    EXTRA="openfox"
  fi
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
# Fonction commune : télécharge un GGUF depuis HF puis le crée dans
# Ollama via un Modelfile local. Contourne le bug Ollama :
# "realm host huggingface.co does not match original host hf.co"
# qui empêche `ollama pull hf.co/...` de fonctionner de façon fiable.
# ------------------------------------------------------------------
ollama_create_from_hf() {
  local hf_repo="$1"       # ex: unsloth/Qwen3.8-27B-GGUF
  local gguf_file="$2"     # ex: Qwen3.8-27B-Q4_K_M.gguf
  local ollama_name="$3"   # ex: qwen3.8-27b

  pip install -q -U huggingface_hub
  mkdir -p "$WORKDIR/models"

  echo "  Téléchargement de $gguf_file depuis $hf_repo..."
  if ! hf download "$hf_repo" "$gguf_file" --local-dir "$WORKDIR/models"; then
    echo "❌ Échec du téléchargement. Causes fréquentes :"
    echo "   - nom de fichier incorrect (vérifiez sur https://huggingface.co/$hf_repo/tree/main)"
    echo "   - dépôt 'gated' : acceptez les conditions sur la page HuggingFace du modèle"
    echo "   - HF_TOKEN manquant ou invalide"
    exit 1
  fi

  cat > "$WORKDIR/models/Modelfile-${ollama_name}" <<EOF
FROM $WORKDIR/models/$gguf_file
EOF

  ollama create "$ollama_name" -f "$WORKDIR/models/Modelfile-${ollama_name}"
}

# ------------------------------------------------------------------
# Option 2 : version légère GGUF via Ollama
# ------------------------------------------------------------------
install_light() {
  echo "▶ Installation de Qwen3.8-27B quantifié (Q4_K_M, ~17GB VRAM) via Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
  (OLLAMA_HOST=0.0.0.0:11434 ollama serve &>/tmp/ollama.log &)
  sleep 5
  ollama_create_from_hf "unsloth/Qwen3.8-27B-GGUF" "Qwen3.8-27B-UD-Q4_K_M.gguf" "qwen3.8-27b"
  echo "✅ Installation terminée. Lancez avec : ollama run qwen3.8-27b"
  echo "   API disponible sur le port 11434 (pas 8000 — celui-ci est réservé à vLLM)."
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

  cat <<'GATE'
⚠️  Le dépôt HuggingFace de ce modèle est "gated" : vous devez accepter
    les conditions d'accès manuellement, une seule fois, dans un navigateur :
    https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF
    Cliquez sur "Log In" puis acceptez les conditions affichées sur la page.
    Sans ça, le téléchargement échouera même avec un token valide.
GATE
  read -rp "Avez-vous déjà accepté les conditions sur cette page ? [o/N] " GATE_OK
  if [[ ! "$GATE_OK" =~ ^[oOyY]$ ]]; then
    echo "Faites-le d'abord, puis relancez : bash setup.sh uncensored"
    exit 0
  fi

  echo "▶ Installation de Qwen3.8-27B-Uncensored (Q4_K_M, ~17GB VRAM) via Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
  (OLLAMA_HOST=0.0.0.0:11434 ollama serve &>/tmp/ollama.log &)
  sleep 5
  ollama_create_from_hf "orcarouter/Qwen3.8-27B-Uncensored-GGUF" "Qwen3.8-27B-Uncensored-Q4_K_M.gguf" "qwen3.8-27b-uncensored"
  echo "✅ Installation terminée. Lancez avec : ollama run qwen3.8-27b-uncensored"
  echo "   API disponible sur le port 11434 (pas 8000 — celui-ci est réservé à vLLM)."
}

# ------------------------------------------------------------------
# Option 4 : OpenFox — interface de codage agentique branchée sur le modèle
# https://github.com/co-l/openfox
# ------------------------------------------------------------------
install_openfox() {
  local backend="$1"    # vllm | ollama
  local llm_url="$2"    # ex: http://localhost:8000/v1

  echo "▶ Installation d'OpenFox (assistant de codage agentique)"

  local node_major=0
  if command -v node &>/dev/null; then
    node_major="$(node -v | sed -E 's/^v([0-9]+).*/\1/')"
  fi
  if [[ "$node_major" -lt 24 ]]; then
    echo "  Node.js >= 24 requis, installation..."
    curl -fsSL https://deb.nodesource.com/setup_24.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs >/dev/null
  fi

  npm i -g openfox >/dev/null

  cat > "$WORKDIR/start_openfox.sh" <<EOF
#!/usr/bin/env bash
OPENFOX_HOST=0.0.0.0 \\
OPENFOX_PORT=${OPENFOX_PORT} \\
OPENFOX_BACKEND=${backend} \\
OPENFOX_LLM_URL=${llm_url} \\
openfox --no-browser
EOF
  chmod +x "$WORKDIR/start_openfox.sh"
  echo "✅ OpenFox installé. Lancez-le avec : $WORKDIR/start_openfox.sh"
  echo "   Interface sur le port ${OPENFOX_PORT} (à exposer via Connect → HTTP Service sur RunPod)."
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "$MODEL_CHOICE" in
  full)
    install_full
    [[ "$EXTRA" == "openfox" ]] && install_openfox "vllm" "http://localhost:${PORT}/v1"
    ;;
  light)
    install_light
    [[ "$EXTRA" == "openfox" ]] && install_openfox "ollama" "http://localhost:11434/v1"
    ;;
  uncensored)
    install_uncensored
    [[ "$EXTRA" == "openfox" ]] && install_openfox "ollama" "http://localhost:11434/v1"
    ;;
  openfox)
    # installe seulement l'interface, en supposant qu'un serveur vLLM tourne déjà sur $PORT
    install_openfox "vllm" "http://localhost:${PORT}/v1"
    ;;
  *)
    echo "❌ Choix inconnu : $MODEL_CHOICE (attendu: full | light | uncensored | openfox)"
    exit 1
    ;;
esac
