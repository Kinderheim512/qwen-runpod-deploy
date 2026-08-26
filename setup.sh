#!/usr/bin/env bash
#
# setup.sh — Assistant de déploiement Qwen3.8-27B sur Pod RunPod (A6000 48GB+)
#
# Mode assistant (aucun argument) : pose les questions dans l'ordre (variante,
# OpenFox, Open WebUI, vision), vérifie CUDA/disque AVANT tout téléchargement
# lourd, puis installe tout et affiche un récapitulatif final avec les
# valeurs exactes à utiliser dans OpenFox/Open WebUI.
#
# Mode scripté (rétrocompatible, arguments dans n'importe quel ordre) :
#   bash setup.sh full
#   bash setup.sh light
#   bash setup.sh uncensored
#   bash setup.sh light openfox
#   bash setup.sh full openfox webui
#   bash setup.sh light vision            # + petit modèle vision qwen3.5:0.8b
#   bash setup.sh openfox                 # ajoute seulement OpenFox
#   bash setup.sh webui                   # ajoute seulement Open WebUI
#
# Variables d'environnement optionnelles (persistées automatiquement dans
# $WORKDIR/.env et rechargées à chaque nouveau shell sur ce Pod) :
#   HF_TOKEN, PORT, MAX_CTX, WORKDIR, OPENFOX_PORT, WEBUI_PORT
#
set -euo pipefail

# ------------------------------------------------------------------
# Répertoires et persistance
# ------------------------------------------------------------------
INVOKE_DIR="$(pwd)"                       # où l'utilisateur a lancé le script (ex: le clone du repo)
WORKDIR="${WORKDIR:-/workspace}"          # volume persistant RunPod
ENV_FILE="${WORKDIR}/.env"
MARKER_FILE="${WORKDIR}/.installed_variant"
OLLAMA_MODELS_DIR="${WORKDIR}/ollama-models"
VISION_MODEL="qwen3.5:0.8b"
VLLM_MIN_CUDA="12.1"

mkdir -p "$WORKDIR"

# Recharge les variables sauvegardées lors d'un run précédent sur ce Pod
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

PORT="${PORT:-8000}"
MAX_CTX="${MAX_CTX:-131072}"
OPENFOX_PORT="${OPENFOX_PORT:-10369}"
WEBUI_PORT="${WEBUI_PORT:-3000}"
HF_TOKEN="${HF_TOKEN:-}"

write_env() {
  cat > "$ENV_FILE" <<EOF
export HF_TOKEN="${HF_TOKEN}"
export PORT="${PORT}"
export MAX_CTX="${MAX_CTX}"
export OPENFOX_PORT="${OPENFOX_PORT}"
export WEBUI_PORT="${WEBUI_PORT}"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_MODELS="${OLLAMA_MODELS_DIR}"
EOF
  local marker_begin="# >>> setup.sh (ajouté automatiquement) >>>"
  if ! grep -qF "$marker_begin" ~/.bashrc 2>/dev/null; then
    {
      echo ""
      echo "$marker_begin"
      echo "[[ -f ${ENV_FILE} ]] && source ${ENV_FILE}"
      echo "# <<< setup.sh <<<"
    } >> ~/.bashrc
  fi
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
show_disk() {
  local label="${1:-}"
  echo ""
  echo "── Espace disque${label:+ ($label)} ──────────────────────────────"
  df -h / "$WORKDIR" 2>/dev/null | awk 'NR==1 || !seen[$1]++'
  echo ""
}

ask() {
  # ask "question" "o|n (défaut)"  ->  affiche "o" ou "n" sur stdout
  local prompt="$1" def="${2:-n}" ans
  if [[ "$def" == "o" ]]; then
    read -rp "${prompt} [O/n] " ans; ans="${ans:-o}"
  else
    read -rp "${prompt} [o/N] " ans; ans="${ans:-n}"
  fi
  [[ "$ans" =~ ^[oOyY]$ ]] && echo "o" || echo "n"
}

version_ge() {
  # vrai si $1 >= $2 (comparaison de versions "x.y")
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]
}

detect_cuda_version() {
  nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1
}

# Issue 2 — driver trop ancien pour vLLM : le détecter AVANT tout téléchargement
check_cuda_for_vllm() {
  local cuda_ver
  cuda_ver="$(detect_cuda_version)"
  if [[ -z "$cuda_ver" ]]; then
    echo "⚠️  Impossible de détecter la version CUDA du driver (nvidia-smi indisponible)."
    return 0
  fi
  echo "Driver CUDA détecté : $cuda_ver (minimum requis pour vLLM : ${VLLM_MIN_CUDA})"
  if ! version_ge "$cuda_ver" "$VLLM_MIN_CUDA"; then
    echo ""
    echo "❌ Le driver de ce Pod (CUDA $cuda_ver) est trop ancien pour vLLM (>= $VLLM_MIN_CUDA requis)."
    echo "   Vous obtiendrez : RuntimeError: The NVIDIA driver on your system is too old."
    echo "   Options : redéployer sur un Pod avec un driver plus récent, ou choisir la variante"
    echo "   'light' (Ollama/GGUF), bien plus tolérante sur les vieux drivers."
    return 1
  fi
  return 0
}

# Issue 3 — flashinfer casse sur Python < 3.11 (array.array[int] non subscriptable).
# Patché proactivement juste après l'installation, avant le premier vllm serve.
patch_flashinfer_if_needed() {
  local pybin pyver major minor
  pybin="$(command -v python3 || command -v python || true)"
  [[ -z "$pybin" ]] && return 0
  pyver="$("$pybin" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  [[ -z "$pyver" ]] && return 0
  major="${pyver%%.*}"; minor="${pyver##*.}"
  if [[ "$major" -eq 3 && "$minor" -lt 11 ]]; then
    local fi_path files
    fi_path="$("$pybin" -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))' 2>/dev/null || true)"
    if [[ -n "$fi_path" && -d "$fi_path" ]]; then
      files="$(grep -rl 'array\.array\[' "$fi_path" 2>/dev/null || true)"
      if [[ -n "$files" ]]; then
        echo "🩹 Python ${pyver} détecté : patch de compatibilité flashinfer en cours..."
        while IFS= read -r f; do
          grep -q "^from __future__ import annotations" "$f" 2>/dev/null || \
            sed -i '1i from __future__ import annotations' "$f"
        done <<< "$files"
        echo "✅ flashinfer patché ($(echo "$files" | wc -l) fichier(s))."
      fi
    fi
  fi
}

# Issue 4 — bascule entre variantes = cache HF orphelin sur le disque conteneur.
clean_container_caches() {
  echo "🧹 Nettoyage des caches sur le disque conteneur (pip, cache HF hub)..."
  rm -rf ~/.cache/huggingface/hub 2>/dev/null || true
  pip cache purge -q 2>/dev/null || true
  echo "✅ Nettoyage terminé."
}

check_variant_switch() {
  local new_variant="$1"
  if [[ -f "$MARKER_FILE" ]]; then
    local prev
    prev="$(cat "$MARKER_FILE")"
    if [[ -n "$prev" && "$prev" != "$new_variant" ]]; then
      echo ""
      echo "ℹ️  Variante précédemment installée : $prev — installation demandée : $new_variant"
      show_disk "avant nettoyage"
      if [[ "$(ask "Nettoyer les caches de l'ancienne variante pour libérer de la place ?" o)" == "o" ]]; then
        clean_container_caches
        show_disk "après nettoyage"
      fi
    fi
  fi
  echo "$new_variant" > "$MARKER_FILE"
}

# Issue "idempotency" — ne jamais re-télécharger des Go de poids déjà présents
is_already_installed() {
  local variant="$1" pybin
  pybin="$(command -v python3 || command -v python || true)"
  case "$variant" in
    full)
      [[ -n "$pybin" ]] && "$pybin" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)" 2>/dev/null
      ;;
    light)
      command -v ollama &>/dev/null && ollama list 2>/dev/null | grep -q "^qwen3\.8-27b:"
      ;;
    uncensored)
      command -v ollama &>/dev/null && ollama list 2>/dev/null | grep -q "^qwen3\.8-27b-uncensored:"
      ;;
    *) return 1 ;;
  esac
}

# Issue 1 — écrit le script de lancement dans WORKDIR (persistant) ET pose un
# lien symbolique dans le dossier où l'utilisateur a lancé setup.sh, pour que
# "./start_xxx.sh" fonctionne quel que soit l'endroit d'où on l'exécute.
link_start_script() {
  local script_name="$1"
  if [[ "$INVOKE_DIR" != "$WORKDIR" ]]; then
    ln -sf "$WORKDIR/$script_name" "$INVOKE_DIR/$script_name" 2>/dev/null || true
  fi
  echo "   Chemin absolu : $WORKDIR/$script_name"
  [[ "$INVOKE_DIR" != "$WORKDIR" ]] && echo "   Raccourci créé : $INVOKE_DIR/$script_name"
}

# ------------------------------------------------------------------
# Paquets système requis (Issue observée précédemment sur images RunPod
# minimalistes : pciutils/zstd/tmux manquants selon les images)
# ------------------------------------------------------------------
echo "▶ Vérification des paquets système (pciutils, curl, git, zstd, tmux)..."
apt-get update -qq
apt-get install -y -qq pciutils curl git zstd tmux >/dev/null

if ! command -v nvidia-smi &>/dev/null; then
  echo "❌ nvidia-smi introuvable. Ce script doit tourner sur un Pod GPU RunPod."
  exit 1
fi
echo "GPU détecté :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
# Parsing des arguments — ordre libre, rétrocompatible
# ------------------------------------------------------------------
VARIANT=""
INSTALL_OPENFOX=0
INSTALL_WEBUI=0
WANT_VISION=0
WIZARD=0

if [[ $# -eq 0 ]]; then
  WIZARD=1
else
  for arg in "$@"; do
    case "$arg" in
      full|light|uncensored)
        if [[ -n "$VARIANT" && "$VARIANT" != "$arg" ]]; then
          echo "❌ Deux variantes différentes passées en argument ($VARIANT, $arg)."
          exit 1
        fi
        VARIANT="$arg"
        ;;
      openfox) INSTALL_OPENFOX=1 ;;
      webui)   INSTALL_WEBUI=1 ;;
      vision)  WANT_VISION=1 ;;
      *)
        echo "❌ Argument inconnu : $arg (attendu: full | light | uncensored | openfox | webui | vision)"
        exit 1
        ;;
    esac
  done
fi

# ------------------------------------------------------------------
# Assistant interactif (nouveau requirement : wizard complet)
# ------------------------------------------------------------------
if [[ "$WIZARD" -eq 1 ]]; then
  echo ""
  echo "=== Assistant d'installation Qwen3.8-27B ==="
  echo "Quelle variante installer ?"
  echo "  1) full        - Qwen3.8-27B-FP8 officiel (vLLM, ~28GB VRAM, meilleure qualité/débit)"
  echo "  2) light       - Qwen3.8-27B GGUF Q4_K_M via Ollama (~17GB VRAM, léger, rapide)"
  echo "  3) uncensored  - Qwen3.8-27B-Uncensored (abliterated) via Ollama (~17GB VRAM)"
  read -rp "Votre choix [1-3]: " CHOICE_NUM
  case "$CHOICE_NUM" in
    1) VARIANT="full" ;;
    2) VARIANT="light" ;;
    3) VARIANT="uncensored" ;;
    *) echo "Choix invalide."; exit 1 ;;
  esac

  [[ "$(ask "Installer aussi OpenFox (interface de codage agentique) ?")" == "o" ]] && INSTALL_OPENFOX=1
  [[ "$(ask "Installer aussi Open WebUI (chat + upload de documents / RAG) ?")" == "o" ]] && INSTALL_WEBUI=1
  [[ "$(ask "Installer aussi un petit modèle vision (${VISION_MODEL}) ?")" == "o" ]] && WANT_VISION=1

  echo ""
  echo "=== Vérifications avant de télécharger quoi que ce soit ==="
  show_disk "avant installation"
  if [[ "$VARIANT" == "full" ]]; then
    if ! check_cuda_for_vllm; then
      if [[ "$(ask "Continuer quand même avec 'full' malgré le driver trop ancien ?")" != "o" ]]; then
        echo "Passage à la variante 'light' à la place."
        VARIANT="light"
      fi
    fi
  fi
  echo ""
  read -rp "Appuyez sur Entrée pour lancer l'installation (Ctrl+C pour annuler)..." _
fi

if [[ "$VARIANT" == "full" && "$WIZARD" -eq 0 ]]; then
  check_cuda_for_vllm || echo "⚠️  Poursuite malgré l'avertissement CUDA (mode scripté)."
fi

# ------------------------------------------------------------------
# Option 1 : version complète FP8 via vLLM
# ------------------------------------------------------------------
install_full() {
  check_variant_switch "full"

  if is_already_installed "full"; then
    echo "✅ vLLM déjà installé — pas de re-téléchargement, passage direct au démarrage."
  else
    echo "▶ Installation de Qwen3.8-27B-FP8 via vLLM (~28GB VRAM)"
    show_disk "avant installation vLLM"
    pip install --upgrade pip -q
    pip install "vllm>=0.9" --extra-index-url https://download.pytorch.org/whl/cu121
    patch_flashinfer_if_needed
    show_disk "après installation vLLM"
  fi

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
  link_start_script "start_vllm.sh"
  echo "✅ Installation terminée."
  SERVER_PORT="$PORT"
  SERVER_ENGINE="OpenAI compatible (vLLM)"

  if [[ "$WANT_VISION" -eq 1 ]]; then
    echo ""
    echo "ℹ️  La vision n'est disponible que sur les variantes Ollama (light/uncensored)."
    echo "   Le modèle vision ne sera pas installé pour 'full' — passez par 'light'"
    echo "   ou 'uncensored' si vous en avez besoin en plus du texte."
  fi

  if [[ "$WIZARD" -eq 1 ]] && [[ "$(ask "Lancer le serveur vLLM maintenant (session tmux 'vllm') ?" o)" == "o" ]]; then
    tmux new -s vllm -d "$WORKDIR/start_vllm.sh"
    echo "✅ vLLM lancé dans la session tmux 'vllm' (tmux attach -t vllm pour la voir)."
  fi
}

# ------------------------------------------------------------------
# Fonction commune : télécharge un GGUF depuis HF puis le crée dans
# Ollama via un Modelfile local. Contourne le bug Ollama :
# "realm host huggingface.co does not match original host hf.co".
# ------------------------------------------------------------------
ollama_create_from_hf() {
  local hf_repo="$1" gguf_file="$2" ollama_name="$3"

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

# Démarre (ou réutilise) le serveur Ollama dans une session tmux persistante,
# toujours avec OLLAMA_MODELS sur le volume persistant et OLLAMA_HOST externe.
ensure_ollama_running() {
  mkdir -p "$OLLAMA_MODELS_DIR"
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if tmux has-session -t ollama 2>/dev/null; then
    echo "✅ Session tmux 'ollama' déjà active — réutilisation."
  else
    tmux new -s ollama -d \
      "OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=${OLLAMA_MODELS_DIR} OLLAMA_KEEP_ALIVE=-1 ollama serve > /tmp/ollama.log 2>&1"
    sleep 4
    echo "✅ Ollama démarré dans la session tmux 'ollama' (OLLAMA_MODELS=${OLLAMA_MODELS_DIR})."
  fi
}

install_vision_model_if_wanted() {
  if [[ "$WANT_VISION" -eq 1 ]]; then
    echo "▶ Installation du modèle vision ${VISION_MODEL}..."
    OLLAMA_HOST=0.0.0.0:11434 ollama pull "$VISION_MODEL"
    VISION_INSTALLED=1
    echo "✅ Modèle vision ${VISION_MODEL} prêt."
  fi
}

# ------------------------------------------------------------------
# Option 2 : version légère GGUF via Ollama
# ------------------------------------------------------------------
install_light() {
  check_variant_switch "light"
  ensure_ollama_running

  if is_already_installed "light"; then
    echo "✅ qwen3.8-27b déjà présent dans Ollama — pas de re-téléchargement."
  else
    show_disk "avant téléchargement"
    ollama_create_from_hf "unsloth/Qwen3.8-27B-GGUF" "Qwen3.8-27B-UD-Q4_K_M.gguf" "qwen3.8-27b"
    show_disk "après téléchargement"
  fi

  install_vision_model_if_wanted
  write_env

  echo "✅ Installation terminée. Lancez avec : ollama run qwen3.8-27b"
  echo "   API disponible sur le port 11434 (pas 8000 — celui-ci est réservé à vLLM)."
  SERVER_PORT="11434"
  SERVER_ENGINE="Ollama"
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
  if [[ "$(ask "Continuer ?")" != "o" ]]; then
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
  if [[ "$(ask "Avez-vous déjà accepté les conditions sur cette page ?")" != "o" ]]; then
    echo "Faites-le d'abord, puis relancez : bash setup.sh uncensored"
    exit 0
  fi

  check_variant_switch "uncensored"
  ensure_ollama_running

  if is_already_installed "uncensored"; then
    echo "✅ qwen3.8-27b-uncensored déjà présent dans Ollama — pas de re-téléchargement."
  else
    show_disk "avant téléchargement"
    ollama_create_from_hf "orcarouter/Qwen3.8-27B-Uncensored-GGUF" "Qwen3.8-27B-Uncensored-Q4_K_M.gguf" "qwen3.8-27b-uncensored"
    show_disk "après téléchargement"
  fi

  install_vision_model_if_wanted
  write_env

  echo "✅ Installation terminée. Lancez avec : ollama run qwen3.8-27b-uncensored"
  echo "   API disponible sur le port 11434 (pas 8000 — celui-ci est réservé à vLLM)."
  SERVER_PORT="11434"
  SERVER_ENGINE="Ollama"
}

# ------------------------------------------------------------------
# Option 4 : OpenFox — interface de codage agentique branchée sur le modèle
# ------------------------------------------------------------------
install_openfox() {
  local backend="$1" llm_url="$2"
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
  link_start_script "start_openfox.sh"
  echo "✅ OpenFox installé."

  if [[ "$WIZARD" -eq 1 ]] && [[ "$(ask "Lancer OpenFox maintenant (session tmux 'openfox') ?" o)" == "o" ]]; then
    tmux new -s openfox -d "$WORKDIR/start_openfox.sh"
    echo "✅ OpenFox lancé dans la session tmux 'openfox'."
  fi
}

# ------------------------------------------------------------------
# Option 5 : Open WebUI — chat + upload de documents / RAG (via pip)
# ------------------------------------------------------------------
install_open_webui() {
  echo "▶ Installation d'Open WebUI (chat + upload de documents / RAG)"
  pip install -q -U open-webui

  cat > "$WORKDIR/start_webui.sh" <<EOF
#!/usr/bin/env bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OPENAI_API_BASE_URL=http://127.0.0.1:${PORT}/v1
open-webui serve --host 0.0.0.0 --port ${WEBUI_PORT}
EOF
  chmod +x "$WORKDIR/start_webui.sh"
  link_start_script "start_webui.sh"
  echo "✅ Open WebUI installé (premier lancement un peu long)."

  if [[ "$WIZARD" -eq 1 ]] && [[ "$(ask "Lancer Open WebUI maintenant (session tmux 'webui') ?" o)" == "o" ]]; then
    tmux new -s webui -d "$WORKDIR/start_webui.sh"
    echo "✅ Open WebUI lancé dans la session tmux 'webui'."
  fi
}

# ------------------------------------------------------------------
# Récapitulatif final — toutes les valeurs à copier-coller
# ------------------------------------------------------------------
print_summary() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo " RÉCAPITULATIF — à garder sous la main"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  echo "Variante installée : ${VARIANT}"
  echo "Port à exposer sur RunPod (Connect → Expose HTTP Ports) : ${SERVER_PORT}"
  echo ""
  echo "── Valeurs pour 'Add Provider' dans OpenFox / Open WebUI ─────"
  echo "  Engine / type    : ${SERVER_ENGINE}"
  echo "  Base URL         : http://localhost:${SERVER_PORT}/v1"
  echo "  API key          : (laisser vide)"
  echo "  Local provider   : décoché"
  echo ""
  if [[ "${VISION_INSTALLED:-0}" -eq 1 ]]; then
    echo "── Modèle vision ───────────────────────────────────────────"
    echo "  Nom du modèle    : ${VISION_MODEL}"
    echo "  À entrer dans le champ 'vision backend' d'OpenFox."
    echo ""
  fi
  echo "── Sessions tmux actives ──────────────────────────────────────"
  tmux ls 2>/dev/null || echo "  (aucune session tmux active pour le moment)"
  echo ""
  echo "── Reprendre plus tard ─────────────────────────────────────"
  echo "  cd ${INVOKE_DIR} && git pull && bash setup.sh ${VARIANT}"
  echo "════════════════════════════════════════════════════════════"
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
VISION_INSTALLED=0
SERVER_PORT=""
SERVER_ENGINE=""

case "$VARIANT" in
  full)       install_full ;;
  light)      install_light ;;
  uncensored) install_uncensored ;;
  "")
    # aucun variant demandé : seulement openfox/webui en ajout à une install existante
    ;;
  *)
    echo "❌ Choix inconnu : $VARIANT (attendu: full | light | uncensored)"
    exit 1
    ;;
esac

# Pour un ajout "à sec" (openfox/webui sans variante en argument), on retrouve
# la variante déjà installée via le fichier marqueur plutôt que de deviner.
EFFECTIVE_VARIANT="$VARIANT"
if [[ -z "$EFFECTIVE_VARIANT" && -f "$MARKER_FILE" ]]; then
  EFFECTIVE_VARIANT="$(cat "$MARKER_FILE")"
  if [[ -z "$SERVER_PORT" ]]; then
    case "$EFFECTIVE_VARIANT" in
      full)              SERVER_PORT="$PORT"; SERVER_ENGINE="OpenAI compatible (vLLM)" ;;
      light|uncensored)  SERVER_PORT="11434"; SERVER_ENGINE="Ollama" ;;
    esac
  fi
fi

if [[ "$INSTALL_OPENFOX" -eq 1 ]]; then
  case "$EFFECTIVE_VARIANT" in
    full)              install_openfox "vllm"   "http://localhost:${PORT}/v1" ;;
    light|uncensored)  install_openfox "ollama" "http://localhost:11434/v1" ;;
    *)
      echo "⚠️  Aucune variante détectée (ni argument, ni installation précédente)."
      echo "   OpenFox sera configuré par défaut sur vLLM (port ${PORT})."
      install_openfox "vllm" "http://localhost:${PORT}/v1"
      ;;
  esac
fi

if [[ "$INSTALL_WEBUI" -eq 1 ]]; then
  install_open_webui
fi

write_env

if [[ -n "$VARIANT" || "$INSTALL_OPENFOX" -eq 1 || "$INSTALL_WEBUI" -eq 1 ]]; then
  VARIANT="${VARIANT:-$EFFECTIVE_VARIANT}"
  print_summary
fi
