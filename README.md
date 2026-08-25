# Qwen3.8-27B — Quick Deploy sur RunPod (A6000 48GB)

Script d'installation en une commande pour déployer Qwen3.8-27B sur un Pod RunPod équipé d'une A6000. Trois variantes disponibles.

## Variantes

| Variante      | Modèle                                              | VRAM   | Moteur | Cas d'usage                          |
|---------------|------------------------------------------------------|--------|--------|---------------------------------------|
| `full`        | `Qwen/Qwen3.8-27B-FP8`                                | ~28 GB | vLLM   | Meilleure qualité, API OpenAI, prod   |
| `light`       | `unsloth/Qwen3.8-27B-GGUF` (Q4_K_M)                   | ~17 GB | Ollama | Chargement rapide, usage perso        |
| `uncensored`  | `orcarouter/Qwen3.8-27B-Uncensored-GGUF` (Q4_K_M)     | ~17 GB | Ollama | Recherche, red-teaming — voir avertissement ci-dessous |

Une A6000 (48 GB) fait tourner confortablement les trois. Le BF16 complet (~56 GB) ne rentre pas sur une seule carte, d'où le choix FP8/GGUF.

## Déploiement sur RunPod

1. Créez un **Pod** (pas Serverless) sur [runpod.io](https://runpod.io) avec une GPU **A6000**, image de base `runpod/pytorch:2.x-cuda12.x`.
2. Ouvrez le terminal web du Pod.
3. Clonez ce repo et lancez le script :

```bash
git clone https://github.com/VOTRE_USER/VOTRE_REPO.git
cd VOTRE_REPO
chmod +x setup.sh

# menu interactif
bash setup.sh

# ou directement :
bash setup.sh full          # Qwen3.8-27B-FP8 via vLLM
bash setup.sh light         # version quantifiée 4-bit via Ollama
bash setup.sh uncensored    # version uncensored via Ollama
```

4. Pour `full` (vLLM) : lancez ensuite `./start_vllm.sh` puis exposez le port `8000` via l'onglet **HTTP Proxy** du Pod RunPod (Connect → HTTP Port).
5. Pour `light` / `uncensored` (Ollama) : le modèle est prêt via `ollama run <modèle>` directement dans le terminal.

## Variables d'environnement (optionnel)

```bash
PORT=8000 MAX_CTX=131072 WORKDIR=/workspace bash setup.sh full
```

- `PORT` : port d'écoute de l'API vLLM (défaut `8000`)
- `MAX_CTX` : longueur de contexte max (défaut `131072`, le modèle supporte jusqu'à 262144 nativement, extensible à 1M via YaRN)
- `WORKDIR` : dossier de travail (défaut `/workspace`, qui persiste sur le volume RunPod)

## ⚠️ À propos de la variante "uncensored"

Cette variante utilise un modèle communautaire tiers (`abliterated`) dont les garde-fous de sécurité ont été retirés. Elle est fournie ici uniquement parce qu'elle existe publiquement en open weights (Apache 2.0) — son usage reste sous votre entière responsabilité légale et éthique. Le script affiche un avertissement et demande une confirmation avant de l'installer.

## Coûts RunPod

Une A6000 coûte environ 0,4–0,8 $/h selon la disponibilité. Pensez à **arrêter le Pod** quand vous ne l'utilisez pas (un Pod arrêté ne facture que le stockage, pas le compute).
