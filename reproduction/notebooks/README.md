# Notebooks da reproducao ToLD-Br

Versoes modernizadas (2026) dos notebooks do autor, com diff minimo e mudancas catalogadas em
`CHANGES.md`. Reproduzem os numeros do artigo 2010.04543v1 (ver `../REPORT.md`).

## Ambiente
A partir de `reproduction/`, escolha um caminho de setup: **Windows** `.\setup_env.ps1`;
**Linux/macOS** `./setup_env.sh`; ou, multiplataforma, `conda env create -f environment.yml`.
Depois registre o kernel (`python -m ipykernel install --user --name toldbr-repro`; o
`setup_env.sh` ja faz isso) e selecione o kernel `toldbr-repro`. Todas as dependencias estao
em `requirements.txt` (inclui `nbconvert`, `ipykernel`, `nltk`, `wordcloud`, `gdown`); versoes
exatas em `requirements-lock.txt`. Notas de GPU/CPU no `../README.md`.

## Ordem de execucao

**Nucleo (pipeline do artigo):**
1. `01_generate_dataset.ipynb` — gera/valida o `ToLD-BR.csv` e os TSVs de IAA por categoria
   (como no notebook original; o calculo do IAA em si e externo e nao e executado).
2. `02_automl_baseline.ipynb` — baseline BoW+SVM (binario e multi-rotulo). [~minutos]
3. `03_classification.ipynb` — 5 modelos binarios + curva de aprendizado + multi-rotulo
   (gera `learning_curve.json` e as predicoes). [~1,5-2 h de GPU]
4. `07_error_analysis.ipynb` — Tabela 12 (FN-rate por categoria). NOVO (extensao; sem original do autor).
5. `08_compare_figures.ipynb` — gera as figuras comparativas Artigo × Reproducao a partir dos
   `results/*.json` e do `experiments/data/learning_curve.json`. NOVO (extensao).

O `07` e o `08` dependem do `03` (usam `preds_binary_mbert_br.csv` e os
`binary_*.json`/`multilabel_mbert.json`, respectivamente).

**Complementares (fora do escopo do artigo):**
- `04_learning_curve.ipynb` — Figura 3 (consome o json do 03).
- `05_wordcloud.ipynb` — nuvens de palavras por categoria.
- `06_example_pretrained.ipynb` — demo do modelo binario pre-treinado (download externo via
  gdown; *link* original do Google Drive fora do ar — **nao executavel**).

O `04` tambem depende do `03`. Os demais notebooks complementares sao independentes entre si.

## Saidas
`../results/*.json`, `../results/figures/*.png`.

## Notas
A ordem e as saidas estao descritas acima; o catalogo de mudancas esta em `CHANGES.md`.
