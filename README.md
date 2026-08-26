# Qwen3.8-27B — Quick Deploy sur RunPod (A6000 48GB)

Script d'installation en une commande pour déployer **Qwen3.8-27B** sur un Pod RunPod équipé d'une A6000. Trois variantes du modèle, plus une interface de codage agentique (OpenFox) en option.

Repo : https://github.com/Kinderheim512/qwen-runpod-deploy

## Variantes du modèle

| Variante      | Dépôt HuggingFace                                       | Fichier                                | VRAM   | Moteur | Cas d'usage                            |
|---------------|-----------------------------------------------------------|-----------------------------------------|--------|--------|------------------------------------------|
| `full`        | `Qwen/Qwen3.8-27B-FP8`                                     | —                                        | ~28 GB | vLLM   | Meilleure qualité, API OpenAI, prod       |
| `light`       | `unsloth/Qwen3.8-27B-GGUF`                                 | `Qwen3.8-27B-UD-Q4_K_M.gguf`             | ~17 GB | Ollama | Chargement rapide, usage perso            |
| `uncensored`  | `orcarouter/Qwen3.8-27B-Uncensored-GGUF`                   | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`     | ~17 GB | Ollama | Recherche, red-teaming — voir avertissement plus bas |

Une A6000 (48 GB) fait tourner confortablement les trois. Le BF16 complet (~56 GB) ne rentre pas sur une seule carte, d'où le choix FP8/GGUF.

⚠️ **Note architecture** : l'A6000 est Ampere (compute capability 8.6), sans tensor cores FP8 natifs. La variante `full` fonctionne mais via déquantification à la volée — moins rapide qu'sur un GPU Ada/Hopper. Pour un meilleur débit sur A6000, `light` est souvent préférable.

## Démarrage rapide sur RunPod

1. Créez un **Pod** (pas Serverless) sur [runpod.io](https://runpod.io) avec une GPU **A6000**, image de base `runpod/pytorch:2.x-cuda12.x` (CUDA 12.4 ou 12.6 recommandé).
2. Ouvrez le terminal web du Pod (onglet **Connect** → **Start Web Terminal**).
3. Collez ces commandes :

```bash
cd /workspace
git clone https://github.com/Kinderheim512/qwen-runpod-deploy.git
cd qwen-runpod-deploy
chmod +x setup.sh
```

4. **(Recommandé)** Créez un token HuggingFace (https://huggingface.co/settings/tokens, scope lecture suffit) — évite le rate-limit 429 sur les téléchargements.

5. **Si vous comptez installer `uncensored`** : allez d'abord accepter les conditions d'accès sur https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF (dépôt "gated" — connectez-vous puis cliquez pour accepter). Sans ça, le téléchargement échoue même avec un token valide.

6. Lancez l'installation :

```bash
# menu interactif — vous propose aussi d'ajouter OpenFox
bash setup.sh

# ou directement, sans menu :
HF_TOKEN=hf_xxx bash setup.sh full          # Qwen3.8-27B-FP8 via vLLM
HF_TOKEN=hf_xxx bash setup.sh light         # version quantifiée 4-bit via Ollama
HF_TOKEN=hf_xxx bash setup.sh uncensored    # version uncensored via Ollama

# avec l'interface de codage OpenFox en plus :
HF_TOKEN=hf_xxx bash setup.sh full openfox
HF_TOKEN=hf_xxx bash setup.sh light openfox
HF_TOKEN=hf_xxx bash setup.sh uncensored openfox

# ajouter OpenFox après coup, si le modèle tourne déjà :
bash setup.sh openfox
```

Le script installe automatiquement `pciutils`, `curl` et `git` si absents (nécessaire pour `nvidia-smi` sur certaines images RunPod minimalistes).

## Utiliser le modèle une fois installé

**Si vous avez choisi `full` (vLLM, port 8000)** :

```bash
./start_vllm.sh
```

Exposez le port **8000** via **Connect → HTTP Service** sur RunPod. Test rapide :

```bash
curl https://VOTRE-POD-ID-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B-FP8",
    "messages": [{"role": "user", "content": "Bonjour"}]
  }'
```

**Si vous avez choisi `light` ou `uncensored` (Ollama, port 11434)** :

Le serveur Ollama tourne déjà en arrière-plan après le script. Utilisez directement :

```bash
ollama run qwen3.8-27b
# ou
ollama run qwen3.8-27b-uncensored
```

⚠️ Ollama écoute sur le port **11434**, pas 8000 (réservé à vLLM). Exposez `11434` via **Edit Pod → Expose HTTP Ports** (nécessite un redémarrage du Pod — sans perte de données, `/workspace` est persistant).

## Interface de codage agentique : OpenFox

[OpenFox](https://github.com/co-l/openfox) est un assistant de codage autonome (façon Cursor/Cline en local), pas une interface de chat généraliste — pensé pour du développement, avec plan → build, vérification de critères, intégration LSP.

Après installation (`... openfox` ou `bash setup.sh openfox`) :

```bash
./start_openfox.sh
```

Exposez le port **10369** (`OPENFOX_PORT`) via **Connect → Expose HTTP Ports**. L'interface est déjà configurée pour pointer vers le bon backend (vLLM ou Ollama) selon la variante choisie.

## Relancer sur un Pod existant ou après un redémarrage

`/workspace` persiste tant que le Pod n'est pas supprimé (stop/restart le préserve, mais pas la suppression). Pour reprendre :

```bash
cd /workspace/qwen-runpod-deploy && git pull && bash setup.sh full
```

## Variables d'environnement (optionnel)

```bash
HF_TOKEN=hf_xxx PORT=8000 MAX_CTX=131072 WORKDIR=/workspace OPENFOX_PORT=10369 bash setup.sh full openfox
```

- `HF_TOKEN` : token HuggingFace — recommandé, évite le rate-limit 429 et obligatoire pour les dépôts "gated"
- `PORT` : port d'écoute de l'API vLLM (défaut `8000`)
- `MAX_CTX` : longueur de contexte max (défaut `131072`, le modèle supporte jusqu'à 262144 nativement, extensible à 1M via YaRN)
- `WORKDIR` : dossier de travail (défaut `/workspace`, qui persiste sur le volume RunPod)
- `OPENFOX_PORT` : port de l'interface OpenFox (défaut `10369`)

## ⚠️ À propos de la variante "uncensored"

Cette variante utilise un modèle communautaire tiers (`abliterated`) dont les garde-fous de sécurité ont été substantiellement retirés. Elle est fournie ici uniquement parce qu'elle existe publiquement en open weights (Apache 2.0, dépôt gated nécessitant acceptation manuelle des conditions) — son usage reste sous votre entière responsabilité légale et éthique. Le script affiche un avertissement et demande une confirmation explicite avant de l'installer.

## Dimensionnement du Pod

- **Container Disk** : 30–50 Go (système, CUDA, vLLM/Ollama — éphémère, jamais de modèles ici)
- **Volume Disk** (`/workspace`, persistant) : 80–100 Go pour une seule variante à la fois, 150–200 Go pour les trois en parallèle

## Coûts RunPod

Une A6000 coûte environ 0,4–0,8 $/h selon la disponibilité. Pensez à **arrêter le Pod** (pas le supprimer) quand vous ne l'utilisez pas — un Pod arrêté ne facture que le stockage, pas le compute.
