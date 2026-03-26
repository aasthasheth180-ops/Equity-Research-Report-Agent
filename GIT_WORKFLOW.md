# Git Workflow — Equity Research Agent
**Repo:** https://github.com/manojmatala/Equity-Research-Report-Agent

---

## One-Time Setup (you do this once)

### 1. Move config files to repo root

After running the notebook, move these 3 files from `notebook/` to your project root
(the folder that contains `notebook/` and `data/`):

```bash
# In Git Bash, cd to your project root first
cd "C:/Users/manoj/OneDrive/Manu/Domain knowlege/7.Financial Engineering/1.Financial engineering/19.Spring 2026/2. Large Language model/project"

# Move root-level config files out of notebook/
mv notebook/.gitignore  .gitignore
mv notebook/.env.example .env.example
mv notebook/requirements.txt requirements.txt
```

### 2. Copy .env.example → .env and fill in your keys

```bash
cp .env.example .env
# Now edit .env with your actual keys (never commit this file)
```

### 3. Initialize git and connect to GitHub

```bash
git init
git remote add origin https://github.com/manojmatala/Equity-Research-Report-Agent.git

# If the remote already exists, update it instead:
git remote set-url origin https://github.com/manojmatala/Equity-Research-Report-Agent.git
```

### 4. Initial push

```bash
git add .
git commit -m "Initial commit: equity research agent with LangGraph + RAG + DCF + WordPress publisher"
git branch -M main
git push -u origin main
```

---

## Your Repo Structure (clean two-folder layout)

```
Equity-Research-Report-Agent/
├── notebook/               ← all code lives here
│   ├── main.ipynb
│   ├── orchestrator.py
│   ├── llm_engine.py
│   ├── tools.py
│   ├── tools_dcf.py
│   ├── rag_store.py
│   ├── state.py
│   ├── data_loader.py
│   └── publisher.py
├── data/                   ← gitignored outputs; only .gitkeep is tracked
│   ├── .gitkeep
│   └── gs_filings/
│       └── .gitkeep
├── .gitignore              ← excludes .env, data files, __pycache__
├── .env.example            ← template for secrets (safe to commit)
└── requirements.txt
```

---

## Collaborator Onboarding (what you tell teammates)

```bash
# 1. Clone the repo
git clone https://github.com/manojmatala/Equity-Research-Report-Agent.git
cd Equity-Research-Report-Agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up secrets
cp .env.example .env
# Edit .env with your own OPENROUTER_API_KEY and WP credentials

# 4. Create the data folders (gitignored, not in repo)
mkdir -p data/gs_filings

# 5. Add your input data files to data/gs_filings/
# (share CSVs and PDFs out-of-band — e.g. Google Drive, email)

# 6. Open main.ipynb and run cells top to bottom
```

---

## Branch Strategy (recommended for a team of 2–4)

The analogy: `main` is the production line. Feature branches are individual workbenches.
Never push directly to `main`.

```
main            ← stable, always runnable
├── feature/better-prompts        ← improving llm_engine.py
├── feature/add-jpmorgan          ← adding JPM to INST_ID_MAP
├── feature/dcf-sensitivity       ← adding sensitivity table to tools_dcf.py
└── fix/valuation-tool-routing    ← bug fix
```

### Your daily workflow

```bash
# Before starting any work — sync with main
git checkout main
git pull origin main

# Create a branch for your change
git checkout -b feature/your-feature-name

# ... make changes ...

# Stage and commit
git add notebook/llm_engine.py          # stage specific files, not git add .
git commit -m "feat: tighten investment thesis prompt constraints"

# Push your branch
git push origin feature/your-feature-name

# Open a Pull Request on GitHub to merge into main
# → Review → Merge → Delete branch
```

### Commit message convention (keeps history readable)

```
feat: add JPMorgan to INST_ID_MAP
fix: handle missing CET1 ratio in run_dcf
refactor: extract _md_to_html into publisher.py
docs: update .env.example with WP credentials
```

---

## Handling Conflicts

Conflicts happen when two people edit the same file. Most common in `llm_engine.py`
(section prompts) and `data_loader.py` (adding new banks).

```bash
# If git pull shows a conflict:
git pull origin main

# Git marks conflicts in the file like this:
# <<<<<<< HEAD
#   your version
# =======
#   teammate's version
# >>>>>>> feature/their-branch

# Edit the file to keep the right version, then:
git add notebook/llm_engine.py
git commit -m "merge: resolve section prompt conflict"
```

**Prevention:** Assign file ownership. One person owns `llm_engine.py`, one owns `tools.py`, etc.
(You already have this pattern in your codebase comments — `# owned by Person B`)

---

## Pulling Teammate Changes

```bash
git checkout main
git pull origin main                    # get latest
git checkout your-feature-branch
git rebase main                         # replay your changes on top of latest main
# resolve any conflicts, then:
git push origin your-feature-branch --force-with-lease
```

---

## Quick Reference

| Task | Command |
|------|---------|
| See what changed | `git status` |
| See the diff | `git diff notebook/tools.py` |
| Undo last uncommitted change | `git restore notebook/tools.py` |
| See commit history | `git log --oneline -10` |
| Switch branches | `git checkout branch-name` |
| Delete local branch after merge | `git branch -d feature/done` |
