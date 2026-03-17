# Luminar Subnet (netuid 87)

**Luminar** is a Bittensor subnet for **real-time object detection in traffic videos**.  
Miners submit a single `agent.py` file with their inference logic. Validators run it inside a secure, disposable Docker sandbox, score the output against hidden ground truth using micro-F1 + IoU matching, and automatically set 100% on-chain weight to the current best agent.

**Website**: [https://luminar.network/](https://luminar.network/)  
**Status**: Live on Bittensor Finney • NetUID 87

---

## Prerequisites

- Python 3.13+
- Docker (with NVIDIA Container Toolkit for GPU acceleration)
- Bittensor wallet (`btcli`)
- `uv` (recommended) or `pip`

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/luminar-subnet.git
cd luminar-subnet
uv sync                  
uv pip install -e .
```

## Build Sandbox Image (required for validators)

```bash
docker build -f Dockerfile.sandbox -t luminar-sandbox:latest .
```

## Running a Miner
Miners do not run a daemon — you just write and submit code.

### Step-by-step

  1. Copy and edit agent.py
  2. Keep the file structure exactly as provided.
  3. Only modify the code between these markers:
  ```bash
      ############################# START #############################
      # ←←← PUT YOUR ENTIRE MODEL LOGIC HERE (imports, classes, etc.)
      ############################# END #############################
  ```

  Two phases your agent must support (already in the template):
  - --setup (network ON): download models, weights, etc. to /cache/
  - --infer (network OFF): read videos from /data/input, write output.csv to /data/output and call the functions from above.


  Required output format (see luminar/validator/scoring.py):
  ```bash
    video_id,frame_idx,category,n_items,box_2d
    video_0000.mp4,0,car,2,"[[x1,y1,x2,y2], [x1,y1,x2,y2]]"
    ...
  ```

  4. Submit the agent:
  ```bash
    python luminar.py submit --agent your_agent.py
  ```
  
  The CLI will:
  - Prompt for wallet/hotkey/password (or use flags: --wallet.name, --hotkey)
  - Sign with your hotkey
  - Upload to the backend
  - After submission, all validators will automatically evaluate your agent.
  - One hotkey can submit another agent after 48 hours of first submission.



## Running a Validator

### Step 1: Register Validator
```bash
btcli subnet register --netuid 87 --wallet.name your_wallet_name --wallet.hotkey your_vali_hotkey
```

### Step 2: Build sandbox image
```bash
docker build -f Dockerfile.sandbox -t luminar-sandbox:latest .
```

### Step 3: Create .env file, copy and update it from .env.example.

### Step 4: Start Validator
  ```bash
    python -m neurons.validator \
      --wallet.name <your_wallet_name> \
      --wallet.hotkey <your_vali_hotkey> \
      --subtensor.network finney \
      --benchmark-dir benchmark/traffic
  ```

## What the validator does? 

  - Polls backend for new unevaluated submissions
  - Downloads agent + runs malware scan
  - Plagiarism check against current best agent (AST similarity)
  - Two-phase sandbox:
    - Phase 1 (--setup): network enabled, cache writable
    - Phase 2 (--infer): network disabled, processes traffic videos

  - Scores with robust micro-F1 (IoU ≥ 0.5 matching)
  - Posts score to backend
  - Blacklists cheaters
  - Every ~360 blocks: sets 100% weight to the current best agent (cached locally). However, we are currently burning 100% of emissions.

  - Validator logs will be saved in `logs/luminar_vali.log`.