# Qwen3.8-27B — Assistant de déploiement RunPod (A6000 48GB+)

`setup.sh` est maintenant un **assistant complet** : lancé sans argument, il pose les
questions dans l'ordre, vérifie CUDA et l'espace disque *avant* tout téléchargement
lourd, installe tout, et termine par un récapitulatif avec les valeurs exactes à copier
dans OpenFox / Open WebUI. Les usages scriptés existants restent identiques.

Repo : https://github.com/Kinderheim512/qwen-runpod-deploy

## Démarrage rapide

```bash
cd /workspace
git clone https://github.com/Kinderheim512/qwen-runpod-deploy.git
cd qwen-runpod-deploy
chmod +x setup.sh
bash setup.sh
```

L'assistant demande, dans l'ordre : la **variante** du modèle, si vous voulez **OpenFox**,
si vous voulez **Open WebUI**, si vous voulez un **modèle vision**. Il affiche ensuite
l'espace disque et la compatibilité CUDA, et attend une confirmation avant de télécharger
quoi que ce soit.

## Usage scripté (rétrocompatible, CI/scripts)

Les arguments peuvent être combinés dans n'importe quel ordre :

```bash
bash setup.sh full
bash setup.sh light
bash setup.sh uncensored
bash setup.sh light openfox
bash setup.sh full openfox webui
bash setup.sh light vision          # + petit modèle vision qwen3.5:0.8b
bash setup.sh openfox               # ajoute OpenFox à une install déjà en place
bash setup.sh webui                 # idem pour Open WebUI
```

## Variantes du modèle

| Variante      | Dépôt HF                                              | Fichier                                | VRAM   | Moteur |
|---------------|----------------------------------------------------------|-----------------------------------------|--------|--------|
| `full`        | `Qwen/Qwen3.8-27B-FP8`                                    | —                                        | ~28 GB | vLLM (port 8000) |
| `light`       | `unsloth/Qwen3.8-27B-GGUF`                                | `Qwen3.8-27B-UD-Q4_K_M.gguf`             | ~17 GB | Ollama (port 11434) |
| `uncensored`  | `orcarouter/...-Uncensored-GGUF`                          | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`     | ~17 GB | Ollama (port 11434) |

L'A6000 (48 GB) fait tourner les trois confortablement. Note : architecture Ampere, sans
tensor cores FP8 natifs — `full` fonctionne mais via déquantification à la volée.

## Ce que l'assistant vérifie automatiquement avant d'installer

Ces problèmes ont tous été rencontrés en usage réel — ils sont maintenant gérés par le
script plutôt que documentés comme dépannage manuel :

- **Driver CUDA trop ancien pour vLLM** : `nvidia-smi` est interrogé avant tout
  téléchargement. Si le driver ne supporte pas CUDA ≥ 12.1 et que vous avez choisi `full`,
  le script prévient clairement et propose de basculer sur `light` plutôt que de laisser
  échouer après plusieurs Go téléchargés.
- **Bug flashinfer sur Python < 3.11** (`TypeError: 'type' object is not subscriptable`) :
  patché automatiquement juste après l'installation de vLLM, avant le premier démarrage.
- **Cache HuggingFace orphelin** en cas de changement de variante (`full` → `light` ou
  inversement) : le script détecte la variante précédente via un fichier marqueur, affiche
  `df -h` avant/après, et propose de nettoyer le cache pip/HF sur le disque conteneur.
- **Ollama qui stocke les modèles sur le petit disque conteneur** : `OLLAMA_MODELS` pointe
  toujours vers `$WORKDIR/ollama-models` (volume persistant), jamais l'emplacement par
  défaut `/root/.ollama/models`.
- **Ollama injoignable depuis l'extérieur** : `OLLAMA_HOST=0.0.0.0:11434` est toujours
  appliqué avant de démarrer le serveur.
- **Emplacement des scripts de lancement** : `start_vllm.sh` / `start_openfox.sh` /
  `start_webui.sh` sont écrits dans `$WORKDIR` (persistant) *et* un raccourci est créé dans
  le dossier où vous avez lancé `setup.sh`, pour que `./start_xxx.sh` fonctionne toujours.
- **Re-téléchargement inutile** : relancer `bash setup.sh <variante>` sur un Pod qui l'a
  déjà détecte l'installation existante et saute directement au démarrage du serveur.

## Persistance entre sessions

`HF_TOKEN`, les ports, et la config Ollama (`OLLAMA_HOST`, `OLLAMA_MODELS`) sont
automatiquement sauvegardés dans `$WORKDIR/.env` et rechargés dans tout nouveau terminal
sur le même Pod (ajout auto d'une ligne `source` dans `~/.bashrc`). Plus besoin de retaper
les `export` à chaque nouvelle session de terminal.

## Sessions tmux

Tous les services longue durée (Ollama, vLLM, OpenFox, Open WebUI) tournent dans des
sessions **tmux nommées** (`ollama`, `vllm`, `openfox`, `webui`), qui survivent à la
fermeture du terminal web RunPod :

```bash
tmux ls                  # lister les sessions actives
tmux attach -t ollama    # se rebrancher dessus
# Ctrl+b puis d : se détacher sans tuer le processus
```

## Modèle vision (nouveau)

Passez `vision` en argument (ou répondez "oui" dans l'assistant) pour installer en plus
un petit modèle vision (`qwen3.5:0.8b`, ~500MB-1GB) sur le backend Ollama, utilisable
comme "vision backend" dans OpenFox.

⚠️ La vision n'est disponible **que sur les variantes Ollama** (`light`/`uncensored`).
Avec `full` (vLLM), le script vous prévient et ignore l'installation du modèle vision —
ça évite de faire tourner deux stacks d'inférence en parallèle sur le même GPU.

## Récapitulatif de fin d'installation

Chaque installation se termine par un bloc récapitulatif : port à exposer sur RunPod,
valeurs exactes pour "Add Provider" dans OpenFox/Open WebUI (engine, URL avec `/v1`, clé
API vide), nom du modèle vision si installé, et sessions tmux actives.

## Utiliser le modèle une fois installé

**Variante `full` (vLLM, port 8000)** — exposez le port via Connect → HTTP Service :
```bash
curl https://VOTRE-POD-ID-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.8-27B-FP8", "messages": [{"role": "user", "content": "Bonjour"}]}'
```

**Variantes `light`/`uncensored` (Ollama, port 11434)** :
```bash
ollama run qwen3.8-27b
# ou
ollama run qwen3.8-27b-uncensored
```

## OpenFox — assistant de codage agentique

https://github.com/co-l/openfox — pas une interface de chat généraliste, mais un
assistant de codage autonome. Deux façons de l'utiliser :

**Sur le Pod** (accès aux fichiers du Pod uniquement) :
```bash
bash setup.sh openfox
tmux new -s openfox -d '/workspace/start_openfox.sh'
```

**En local sur votre PC** (accès à vos fichiers locaux, connexion au modèle via tunnel
SSH plutôt que le proxy HTTPS public — plus stable pour du streaming API prolongé) :
utilisez les scripts `start-openfox.ps1` / `stop-tunnel.ps1` fournis séparément, qui
ouvrent le tunnel puis lancent OpenFox local automatiquement.

## Open WebUI — chat + upload de documents / RAG

```bash
bash setup.sh webui
tmux new -s webui -d '/workspace/start_webui.sh'
```
Exposez le port `3000` (`WEBUI_PORT`) via Connect → HTTP Service.

## Nettoyage VRAM (`gpu-clean.sh`)

Si un chargement de modèle plante en cours de route, des processus zombies peuvent
retenir de la VRAM sans apparaître dans `nvidia-smi --query-compute-apps`, forçant les
chargements suivants en fallback CPU (très lent). Utilisez `gpu-clean.sh` (fourni
séparément) pour libérer la VRAM en cas de doute — il demande confirmation avant de tuer
quoi que ce soit.

## Variables d'environnement

```bash
HF_TOKEN=hf_xxx PORT=8000 MAX_CTX=131072 WORKDIR=/workspace \
  OPENFOX_PORT=10369 WEBUI_PORT=3000 bash setup.sh full openfox webui vision
```

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | Token HuggingFace — évite le rate-limit 429, requis pour dépôts gated |
| `PORT` | `8000` | Port de l'API vLLM |
| `MAX_CTX` | `131072` | Longueur de contexte max (jusqu'à 262144 natif, 1M via YaRN) |
| `WORKDIR` | `/workspace` | Dossier de travail persistant |
| `OPENFOX_PORT` | `10369` | Port de l'interface OpenFox |
| `WEBUI_PORT` | `3000` | Port d'Open WebUI |

## ⚠️ À propos de la variante "uncensored"

Modèle communautaire tiers (`abliterated`) dont les garde-fous ont été substantiellement
retirés. Dépôt HuggingFace "gated" (acceptation manuelle requise une fois). Usage sous
votre entière responsabilité légale et éthique — le script demande confirmation explicite
avant installation.

## Dimensionnement du Pod

- **Container Disk** : 30–50 Go (système, éphémère)
- **Volume Disk** (`/workspace`, persistant) : 80–100 Go pour une variante, 150–200 Go
  pour les trois en parallèle

## Coûts RunPod

Une A6000 coûte environ 0,4–0,8 $/h selon la disponibilité. Arrêtez le Pod (pas de
suppression) entre deux sessions — un Pod arrêté ne facture que le stockage.
