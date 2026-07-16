# Reprodução ToLD-Br (notebooks)

Reprodução do artigo 2010.04543v1 (*Toxic Language Detection in Social Media for
Brazilian Portuguese*, Leite et al., 2020), feita **inteiramente a partir dos notebooks
do repositório oficial**, modernizados para rodar em 2026.

A reprodução segue o princípio de **diff mínimo**: preserva-se ao máximo o código original
dos autores (estrutura das células, classes, nomes e fluxo) e altera-se apenas o
estritamente necessário, classificando cada mudança como **(T)** tecnologia descontinuada,
**(B)** correção de defeito ou **(A)** fidelidade ao artigo. Todas as alterações estão
catalogadas e justificadas em [`notebooks/CHANGES.md`](notebooks/CHANGES.md).

- **Notebooks:** [`notebooks/`](notebooks/) — ver [`notebooks/README.md`](notebooks/README.md)
- **Relatório de resultados (vs. artigo):** [`REPORT.md`](REPORT.md)
- **Catálogo de mudanças:** [`notebooks/CHANGES.md`](notebooks/CHANGES.md)

## Setup

A reprodução roda inteiramente pelos notebooks. Escolha **um** caminho de setup e depois
registre o kernel Jupyter `toldbr-repro`.

**Windows (PowerShell):**
```powershell
.\setup_env.ps1                 # torch com CUDA 12.4 (GPU NVIDIA)
# .\setup_env.ps1 -Python "py -3.12"   # fallback de wheel no Python 3.13
python -m ipykernel install --user --name toldbr-repro
```

**Linux/macOS (bash):**
```bash
./setup_env.sh                  # torch com CUDA 12.4 (GPU NVIDIA)
# ./setup_env.sh --cpu          # torch CPU-only
# o script já registra o kernel toldbr-repro
```

**Multiplataforma (conda):**
```bash
conda env create -f environment.yml
conda activate toldbr-repro
# instale o torch conforme a aceleração:
#   GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124
#   CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name toldbr-repro
```

> **GPU vs CPU.** Os notebooks ativam *mixed precision* apenas quando há GPU
> (`fp16=torch.cuda.is_available()`), então rodam também em CPU — mas o treino dos modelos
> BERT (`03`) fica bem mais lento sem GPU. O ambiente de referência é uma RTX 4070 Ti SUPER
> (16 GB, CUDA 12.4). As versões exatas das bibliotecas estão em `requirements-lock.txt`.

## Rodar

Abra os notebooks em `notebooks/` (Jupyter/VS Code) com o kernel `toldbr-repro`, ou
execute via linha de comando a partir de `reproduction/`:

```powershell
$kernel = "--ExecutePreprocessor.kernel_name=toldbr-repro"
$tout   = "--ExecutePreprocessor.timeout=-1"
jupyter nbconvert --to notebook --execute $tout $kernel --inplace notebooks\01_generate_dataset.ipynb
# ... e assim por diante para 02..08
```

Em Linux/macOS, use o python do venv (`$HOME/.cache/toldbr-repro-venv/bin/python`) no lugar de `$py`:
```bash
VPY="$HOME/.cache/toldbr-repro-venv/bin/python"
K="--ExecutePreprocessor.kernel_name=toldbr-repro"
"$VPY" -m jupyter nbconvert --to notebook --execute --inplace $K notebooks/01_generate_dataset.ipynb
# ... 02, 03 (use --ExecutePreprocessor.timeout=-1), 07, 08
```

**Pipeline do artigo (núcleo):** `01 → 02 → 03 → 07 → 08`.
**Complementares (fora do escopo do artigo):** `04` (curva), `05` (nuvens de palavras),
`06` (demo do modelo pré-treinado — *link* original fora do ar, **não executável**).

Ordem de execução (detalhes em [`notebooks/README.md`](notebooks/README.md)):

| # | Notebook | O que produz |
|---|----------|--------------|
| 01 | `01_generate_dataset.ipynb` | gera/valida `ToLD-BR.csv` e os arquivos de IAA (como no original) |
| 02 | `02_automl_baseline.ipynb` | baseline BoW+SVM (binário e multi-rótulo) |
| 03 | `03_classification.ipynb` | 5 modelos binários + curva de aprendizado + multi-rótulo (~1,5–2 h de GPU) |
| 04 *(complementar)* | `04_learning_curve.ipynb` | Figura 3 (consome `results/learning_curve.json` do 03) |
| 05 *(complementar)* | `05_wordcloud.ipynb` | nuvens de palavras por categoria |
| 06 *(complementar)* | `06_example_pretrained.ipynb` | demo do modelo binário pré-treinado (download externo) |
| 07 | `07_error_analysis.ipynb` | Tabela 12 (FN-rate por categoria); consome predições do 03 |
| 08 | `08_compare_figures.ipynb` | figuras comparativas artigo × reprodução (R.1–R.4); consome resultados do 03 |

O `04`, o `07` e o `08` dependem do `03`; os demais são independentes entre si.

> O baseline **auto-sklearn** original (Linux-only) está apenas **documentado** no notebook 02,
> que usa **BoW+SVM** como baseline executável. O notebook 06 depende de um modelo
> pré-treinado cujo link do Google Drive (2020) não permite mais o download — sua
> estrutura é validada, mas a execução ponta-a-ponta não é possível.

## Saídas

- `results/*.json` — métricas de cada experimento
- `results/figures/*.png` — matrizes de confusão, curva de aprendizado, nuvens de palavras
- [`REPORT.md`](REPORT.md) — comparação final com os números do artigo
