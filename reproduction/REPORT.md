# Relatório de Reprodução — ToLD-Br (Leite et al., 2020)

Reprodução de *"Toxic Language Detection in Social Media for Brazilian Portuguese:
New Dataset and Multilingual Analysis"* (Leite, Silva, Bontcheva, Scarton — AACL-IJCNLP 2020;
arXiv:2010.04543v1), a partir dos dados e do código do repositório oficial.

> Este documento é o relatório detalhado da reprodução, escrito para servir de material-fonte
> para um trabalho acadêmico sobre o esforço de reprodução. **Todos os números "Reproduzido"
> vêm de uma única execução consistente dos notebooks** (`reproduction/notebooks/`, run de
> 2026-06-21), lidos diretamente dos artefatos em `results/` e geráveis novamente reexecutando
> os notebooks na ordem `01 → 07`.

---

## Resumo executivo — veredito (melhor / pior / igual)

> Numa reprodução, o objetivo não é "superar" o artigo e sim **coincidir** com ele; portanto
> "igual" (dentro da margem de ruído de ~1–2 pontos) é o resultado de sucesso esperado.

| Bloco | Métrica | Artigo | Reproduzido | Veredito |
|---|---|---:|---:|---|
| Binária — baseline (BoW+SVM) | macro-F1 | 0.74 | 0.730 | **igual** (artigo usa BoW+AutoML; como o auto-sklearn é Linux-only, aqui usa-se BoW+SVM) |
| Binária — BR-BERT | macro-F1 | 0.76 | 0.765 | **igual** |
| Binária — M-BERT-BR | macro-F1 | 0.75 | 0.766 | **igual** |
| Binária — M-BERT (transfer) | macro-F1 | 0.76 | 0.752 | **igual** |
| Binária — M-BERT (zero-shot) | macro-F1 | 0.56 | 0.544 | **igual** |
| Análise de erro (Tab. 12) | padrão FN-rate | — | — | **igual** (mesmo padrão: minoritárias piores) |
| Curva de aprendizado (Fig. 3) | tendência | — | — | **igual** (mesma forma) |
| Multi-rótulo — baseline | Hamming / AP | 0.08 / 0.20 | 0.069 / 0.209 | **igual** |
| Multi-rótulo — mBERT | Hamming loss | 0.07 | 0.067 | **igual** |
| Multi-rótulo — mBERT | Average Precision (AP) | 0.19 | 0.276 | **melhor** (ver ressalva, §8.5) |

**Leitura geral:**
- **Tarefa binária (resultado central do artigo): reproduzida com sucesso** — os 5 modelos ficaram a
  ≤ 1,6 ponto de macro-F1 do artigo.
- **Baseline:** o artigo usa BoW+AutoML (0.74); como o auto-sklearn é Linux-only, o baseline
  **executável** desta reprodução é BoW+SVM (0.730), na prática idêntico (ver §6).
- **Análise de erro e curva de aprendizado: equivalentes** (mesmas conclusões).
- **Monolíngue vs multilíngue:** o artigo via o monolíngue (BR-BERT) ligeiramente à frente; nesta
  execução **BR-BERT (0.765) e M-BERT-BR (0.766) empatam dentro do ruído** — a vantagem do
  monolíngue não se confirma como direção, mas tampouco se inverte de forma significativa (§9).
- **Multi-rótulo: melhor apenas no Average Precision** (0.276 vs 0.19). **Ressalva:** não é um modelo
  uniformemente superior. Ele troca **precisão por recall** (muito mais falsos positivos) e **empata no
  Hamming loss** (0.067 vs 0.07); a diferença vem da estabilidade do *stack* de treino moderno frente
  ao apex fp16 de 2020 (ver §8.5).

**Em uma frase:** a reprodução foi **fiel** (binária, erro e curva equivalentes ao artigo); o único ponto
de divergência (multi-rótulo) saiu **melhor em AP**, mas como um *trade-off* (mais recall, mais
falsos positivos), não como superioridade geral.

---

## 1. Objetivo e escopo

Reproduzir, em ambiente local, os quatro blocos experimentais do artigo e comparar com os
números originais:

1. **Classificação binária** (tóxico vs não-tóxico) — Tabelas 7–11 e matrizes de confusão (Figura 2);
2. **Análise de erro** — taxa de falsos negativos (FN-rate) por categoria fina de toxicidade (Tabela 12);
3. **Importância do volume de dados** — curva de aprendizado de 10% a 100% (Figura 3);
4. **Classificação multi-rótulo** — mBERT vs baseline (Seção 5.3 e Figura 4).

Decisões de escopo acordadas com o usuário: reprodução **completa** dos quatro blocos; modelos
**fiéis ao artigo** (não ao código liberado — ver §3); baseline **BoW+SVM** (auto-sklearn não roda
no ambiente); **máxima fidelidade** ao protocolo (mantidos o `SVC` com kernel RBF e as 30 execuções
da curva de aprendizado — 3 repetições × 10 frações).

Fora do escopo (descritivo, não experimental): a **concordância entre anotadores** (Krippendorff
α, Tabela 5 do artigo) **não é recalculada numericamente** — o notebook `01` apenas gera os arquivos
de entrada (`*_alpha.tsv`), como no original dos autores, e documenta o comando externo
`mwetoolkit3/bin/kappa.py`, que não é executado (ver §10).

---

## 2. Ambiente de reprodução

| Item | Valor |
|---|---|
| Sistema operacional | Windows 11 Pro (10.0.26200) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB |
| Python | 3.13.9 |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| transformers | 5.9.0 |
| datasets | 4.8.5 |
| accelerate | 1.13.0 |
| scikit-learn | 1.8.0 |
| pandas / numpy / scipy / nltk | 3.0.3 / 2.4.6 / 1.17.1 / 3.9.4 |

**Ambiente virtual:** criado **fora do OneDrive**, em `%LOCALAPPDATA%\toldbr-repro-venv`, para
evitar a sincronização de milhares de arquivos de pacotes (o repositório está numa pasta
sincronizada do OneDrive). Os modelos do HuggingFace são baixados em `~/.cache/huggingface`
(também fora do OneDrive).

**Observação relevante:** as bibliotecas vieram em versões **muito mais novas** que as do artigo
(2020): `transformers 5.x` (vs ~2.x/3.x à época), `datasets 4.x`, `scikit-learn 1.8`. Apesar disso,
a API de alto nível (`Trainer`, `AutoModelForSequenceClassification`) rodou o código sem nenhuma
adaptação (ver §7.3).

**Reprodutibilidade observada:** esta execução (2026-06-21) reproduziu **dígito a dígito** uma
execução anterior feita no mesmo ambiente — sinal de que, com `seed=42` e versões fixas de
biblioteca/CUDA, o pipeline é determinístico neste hardware.

---

## 3. Artigo vs repositório: o que foi efetivamente usado

Há **divergências entre o que o artigo descreve e o que o código liberado faz**. Optou-se por
reproduzir o **artigo**:

| Aspecto | Artigo | Código liberado (`experiments/`) | Nesta reprodução |
|---|---|---|---|
| Baseline | BoW + **AutoML** (auto-sklearn) | `automl.ipynb` (auto-sklearn) + um BoW+SVM em `classification.ipynb` | **BoW + SVM** (auto-sklearn é Linux-only) |
| Modelo PT monolíngue (BR-BERT) | BERTimbau (`neuralmind/bert-base-portuguese-cased`) | **distilbert** (hardcoded) | **BERTimbau** (fiel ao artigo) |
| Modelo multilíngue (M-BERT) | mBERT (`bert-base-multilingual-cased`) | **`distilbert-base-multilingual-cased`** | **mBERT** (fiel ao artigo) |
| Framework | (não detalhado; simpletransformers no código) | `simpletransformers` + NVIDIA `apex`, Google Colab | **HuggingFace `transformers` (Trainer)** |

**Os dados do repositório batem exatamente com o artigo.** A contagem por `wc -l` é enganosa (os tweets contêm
quebras de linha internas), mas o parsing por pandas confirma:

- Splits binários (`experiments/data/1annotator.zip`, agregação "ao menos um anotador"):
  **train 16.800, validation 2.100, test 2.100** (total **21.000**, os 21K do artigo).
- Distribuição do **test = 1.128 não-tóxicos + 972 tóxicos** — idêntica aos totais das matrizes de
  confusão da Figura 2.
- `ToLD-BR.csv`: 21.000 linhas; "qualquer rótulo > 0" produz **9.255 tóxicos / 11.745 não-tóxicos**,
  exatamente a Tabela 6 ("at least one annotator").

Para os modelos BERT seguiu-se o código original: treino = **train + validation = 18.900**
exemplos; avaliação no **test = 2.100**; sem validação durante o treino (sem *early stopping*).

**Ressalva sobre o OLID (transfer / zero-shot):** o artigo diz usar o OLID "concatenando os splits
de *treino e teste*" (≈ 14.100 exemplos: 13.240 treino + 860 teste). O **código liberado**, porém,
carrega apenas o arquivo de **treino** do OLID (`olid-training-v1.0.tsv` = 13.240). Esta reprodução
segue o código (13.240, via `cardiffnlp/tweet_eval`) e, portanto, fica **860 exemplos abaixo** do que
o texto do artigo descreve. Trata-se de uma divergência *texto-do-artigo × código × reprodução* herdada do
repositório. O impacto sobre os resultados de transfer/zero-shot é pequeno, mas o ponto fica
**explicitamente registrado** (ver §6, desvio #6).

---

## 4. Por que não executar o código original diretamente

Os notebooks originais não rodam no ambiente atual sem reescrita:

- **`automl.ipynb`** usa **auto-sklearn**, que é **Linux-only** e não instala no Windows/Python 3.13.
- **`classification.ipynb`** é feito para **Google Colab de 2020**: monta o Google Drive
  (`google.colab.drive`), lê dados de `drive/My Drive/...`, compila o **NVIDIA `apex`**
  (descontinuado, não builda em CUDA/Python modernos), usa `simpletransformers` e `files.download()`.
- **Bugs nos notebooks** que travariam a execução: `AutoSklearnClassifiervb()` (typo),
  argumento `train_amount` passado a um construtor que espera `data_amount`, variável `it`
  indefinida no laço da curva de aprendizado, variável `score` indefinida no baseline multi-rótulo.
- **Divergência de modelo** (§3): rodar o código literal reproduziria *distilbert*, não o BERT
  completo descrito no artigo.

Por isso a reprodução é uma **reimplementação fiel da metodologia** (mesmos dados, splits, número
de épocas, *seed*, modelos do artigo, métricas), trocando apenas a infraestrutura por uma que
roda em 2026 (HuggingFace `transformers`).

---

## 5. Metodologia da reprodução

### 5.1 Pipeline de dados (notebooks `01` e `03`)
- Os *splits* binários são lidos de dentro do zip `experiments/data/1annotator.zip` (colunas
  `text`, `toxic`).
- O conjunto de treino dos modelos BERT = train + validation (18.900), como no código original.
- O multi-rótulo lê `ToLD-BR.csv` e binariza os 6 rótulos (`valor > 0 → 1`); split 90/10
  (`random_state=42`), test = 2.100.
- O OLID (OffensEval 2019) é baixado via `cardiffnlp/tweet_eval` (config *offensive*),
  concatenando train + validation = **13.240** exemplos — que equivalem ao conjunto de treino do OLID (rótulo
  `offensive → 1`). Ver §3 (ressalva) e §7.3.

### 5.2 Modelos e hiperparâmetros (notebooks `02` e `03`)

| Parâmetro | Valor | Origem |
|---|---|---|
| Épocas (binário / multi-rótulo) | 3 | artigo / código |
| Épocas (curva de aprendizado) | 1 | código original |
| *Seed* | 42 | artigo |
| `do_lower_case` | False | artigo |
| Learning rate | 4e-5 | default do `simpletransformers` (inferido) |
| **`warmup_ratio`** | **0.06** | default do `simpletransformers` (inferido); ver §7.1 |
| `max_seq_len` | **128** | desvio (artigo: 512) — ver §6 |
| `train_batch_size` | **32** | desvio (artigo: 50) — ver §6 |
| Precisão | fp16 na GPU | desempenho |
| Limiar multi-rótulo | 0.5 | artigo |
| Baseline | `CountVectorizer` + `SVC` (RBF) | código original |

Modelos: **BR-BERT** = `neuralmind/bert-base-portuguese-cased`; **M-BERT** =
`bert-base-multilingual-cased`.

### 5.3 Mapeamento experimento → artigo

| Notebook | Bloco do artigo | Saída |
|---|---|---|
| `03_classification.ipynb` | Binária, 5 modelos (Tab. 7–11, Fig. 2) | `results/binary_*.json`, figuras |
| `07_error_analysis.ipynb` | FN-rate por categoria (Tab. 12) | `results/error_analysis_fn_rate.json` |
| `03` + `04_learning_curve.ipynb` | Curva de aprendizado (Fig. 3) | `results/learning_curve.json`, figura |
| `03_classification.ipynb` | Multi-rótulo (Fig. 4) | `results/multilabel_*.json` |
| `02_automl_baseline.ipynb` | Baseline BoW+SVM — porte do `automl.ipynb` do autor | saídas no próprio notebook (JSONs sobrescritos pelo `03` — ver nota) |

> **Nota de proveniência dos baselines:** os notebooks `02` e `03` gravam os **mesmos arquivos**
> `results/binary_baseline_bow_svm.json` e `results/multilabel_baseline_bow_svm.json` (o
> `classification.ipynb` do autor também continha os baselines). Na ordem de execução
> `01 → 02 → 03`, os JSONs persistidos — e todos os números de baseline deste relatório e do
> artigo de reprodução — são os do **notebook 03**, cujo protocolo segue o `classification.ipynb`:
> binário treinado em train+validation (18.900) → macro-F1 0.730; multi-rótulo com split 90/10
> (`random_state=42`, teste = 2.100) → Hamming 0.069 / AP 0.209. O notebook `02` (porte do
> `automl.ipynb` do autor) usa os protocolos daquele notebook — binário treinado só em train
> (16.800) → macro-F1 0.728; multi-rótulo com split 0.9/0.5 (18.900/1.050/1.050, avaliado em
> 1.050) → Hamming 0.065 / AP 0.199 — e esses valores ficam visíveis apenas nas saídas do próprio
> notebook, pois os arquivos são sobrescritos em seguida pelo `03`.

---

## 6. Ajustes e desvios em relação ao artigo/código

| # | Desvio | Motivo | Impacto esperado |
|---|---|---|---|
| 1 | Baseline: **BoW+SVM** no lugar do BoW+AutoML | auto-sklearn é Linux-only | nenhum — SVM 0.730 ≈ artigo (0.74) |
| 2 | **`max_seq_len = 128`** (artigo: 512) | tweets têm ≤280 caracteres e cabem em 128 tokens; ~4× mais rápido | desprezível (sem truncamento relevante) |
| 3 | **`train_batch_size = 32`** (artigo: 50) | caber BERT completo + fp16 nos 16 GB | baixo |
| 4 | **`warmup_ratio = 0.06` adicionado** | default do `simpletransformers`; sem ele o *transfer* colapsou (§7.1) | **alto** — corrige instabilidade; também é *mais* fiel ao original |
| 5 | **HuggingFace `transformers`** em vez de `simpletransformers`+`apex` | stack original não roda em 2026 | metodologia idêntica; números podem variar ~1–2 pontos |
| 6 | OLID com **13.240** exemplos (treino do OLID), via `cardiffnlp/tweet_eval` | segue o **código** liberado; mas o **texto** do artigo diz treino+teste (~14.100) | baixo; divergência herdada do repositório, ver §3 |
| 7 | **Chave de texto normalizada** na análise de erro | encoding/espaços divergem entre splits e `ToLD-BR.csv` (§7.2) | melhora o alinhamento de 1.789→2.100/2.100 |

**Reprodutibilidade estocástica:** mesmo com *seed* fixa, versões diferentes de biblioteca/CUDA
movem os resultados de BERT em ~1–2 pontos de macro-F1. A meta foi reproduzir **dentro dessa
margem**, não dígito a dígito (embora, neste ambiente, o pipeline tenha se mostrado determinístico
entre execuções — §2).

**Sobre o BoW+AutoML do artigo:** o notebook `02` documenta como reproduzir o auto-sklearn real em
**Ubuntu/Docker/WSL2** (uma execução anterior deu macro-F1 0.733, best validation 0.7545). Esse caminho
via auto-sklearn é opcional e **não faz parte desta reprodução** (rodada em Windows); o baseline aqui é o BoW+SVM.

---

## 7. Problemas encontrados e soluções

### 7.1 Colapso do *transfer learning* (instabilidade de fine-tuning) — **principal ajuste**

Na primeira execução, o modelo **M-BERT (transfer)** (ToLD-Br + OLID) colapsou:
**macro-F1 = 0.349**, matriz de confusão `[[1128, 0], [972, 0]]` — previu **tudo como não-tóxico**.

**Diagnóstico:** inspecionando o log de treino, a *loss* do transfer ficou **travada em ~0.66–0.68
(≈ ln 2) do início ao fim** — o modelo nunca saiu da solução trivial. Os demais modelos (BR-BERT,
M-BERT-BR, zero-shot) treinaram normalmente (loss decrescente). Esse é o fenômeno conhecido de
**instabilidade de fine-tuning do BERT** (Mosbach et al., 2020): uma fração das execuções "colapsa" na
classe majoritária, especialmente **sem warmup** e com *learning rate* relativamente alto.

**Causa-raiz / solução:** o código original (`simpletransformers`) usa **`warmup_ratio = 0.06`
por padrão**, que havia sido omitido na reimplementação. Adicionar o warmup é, ao mesmo tempo,
**mais fiel ao original** e a correção da instabilidade. Após o ajuste, o transfer passou a treinar
normalmente (**macro-F1 = 0.752**). Por consistência, todos os 5 modelos foram (re)executados com essa mesma
configuração de warmup.

### 7.2 Alinhamento de texto na análise de erro (encoding + normalização)

A análise de erro (Tabela 12) exige cruzar as predições do M-BERT-BR no *test* com as categorias
finas de toxicidade, que só existem no `ToLD-BR.csv`. Nessa junção, o *merge* exato por texto casava apenas
**1.789 de 2.100** linhas, produzindo denominadores menores que os do artigo.

**Diagnóstico:** (i) ~10 linhas têm **corrupção de encoding gravada no arquivo** (sequência UTF-8
`EF BF BD` = U+FFFD, "�"), irrecuperável; (ii) a maioria das divergências eram **diferenças de
caixa, acentos e, principalmente, espaços/quebras de linha internas** dos tweets entre os splits
binários e o `ToLD-BR.csv`.

**Solução:** usar uma **chave de texto normalizada** no *merge* (minúsculas + remoção de acentos +
remoção de U+FFFD + colapso de espaços). Isso recuperou **2.100/2.100** alinhamentos. Os
denominadores por categoria ficaram **próximos, mas não idênticos** aos do artigo (ver §8.2): a
contagem de positivos por categoria fina depende da versão do `ToLD-BR.csv` e da normalização.

### 7.3 Outras incompatibilidades resolvidas

- **auto-sklearn (baseline do artigo):** Linux-only; substituído por BoW+SVM (desvio #1).
- **OLID indisponível pelo nome antigo:** `load_dataset("tweet_eval", ...)` falha no `datasets 4.x`
  (loaders por script foram removidos; exige `namespace/nome`). Resolvido com `cardiffnlp/tweet_eval`
  — o mesmo OLID, 13.240 exemplos (train 11.916 + validation 1.324).
- **transformers 5.x:** verificou-se que `report_to=[]`, `save_strategy="no"`, `fp16`,
  `Trainer(processing_class=None)` etc. funcionam sem alteração — o código rodou **sem desvios**.
- **Modelo pré-treinado (notebook 06):** o link do Google Drive de 2020
  (`id=1Q8MuO4...`) está **fora do ar** (`gdown` retorna `FileURLRetrievalError`). O notebook 06 é
  apenas uma *demo* e **não gera nenhum artefato de comparação**; fica marcado como **não-executável**.

---

## 8. Resultados

### 8.1 Classificação binária (Tabelas 7–11)

**Macro-F1 (métrica principal do artigo):**

| Modelo | Artigo | Reproduzido | Δ |
|---|---:|---:|---:|
| BoW+SVM (baseline executável) | 0.74 | 0.730 | −0.010 |
| BR-BERT (BERTimbau) | 0.76 | 0.765 | +0.005 |
| M-BERT-BR (mBERT) | 0.75 | 0.766 | +0.016 |
| M-BERT (transfer) | 0.76 | 0.752 | −0.008 |
| M-BERT (zero-shot) | 0.56 | 0.544 | −0.016 |

**Precision / Recall / F1 por classe** (classe 0 = não-tóxico, 1 = tóxico):

| Modelo | Classe | P (art.) | P (repr.) | R (art.) | R (repr.) | F1 (art.) | F1 (repr.) |
|---|---|---:|---:|---:|---:|---:|---:|
| BoW (AutoML art. / SVM repr.) | 0 | 0.76 | 0.725 | 0.75 | 0.817 | 0.75 | 0.769 |
| BoW (AutoML art. / SVM repr.) | 1 | 0.71 | 0.752 | 0.73 | 0.641 | 0.72 | 0.692 |
| BR-BERT | 0 | 0.77 | 0.782 | 0.80 | 0.784 | 0.79 | 0.783 |
| BR-BERT | 1 | 0.76 | 0.748 | 0.73 | 0.746 | 0.74 | 0.747 |
| M-BERT-BR | 0 | 0.81 | 0.808 | 0.69 | 0.742 | 0.75 | 0.774 |
| M-BERT-BR | 1 | 0.69 | 0.727 | 0.82 | 0.795 | 0.75 | 0.759 |
| M-BERT (transfer) | 0 | 0.80 | 0.795 | 0.74 | 0.726 | 0.77 | 0.759 |
| M-BERT (transfer) | 1 | 0.72 | 0.711 | 0.79 | 0.783 | 0.75 | 0.745 |
| M-BERT (zero-shot) | 0 | 0.59 | 0.581 | 0.83 | 0.830 | 0.69 | 0.683 |
| M-BERT (zero-shot) | 1 | 0.63 | 0.607 | 0.32 | 0.305 | 0.43 | 0.405 |

**Matrizes de confusão** `[[TN, FP], [FN, TP]]` (Figura 2):

| Modelo | Artigo | Reproduzido |
|---|---|---|
| BoW baseline | `[[843,285],[263,709]]` (AutoML) | `[[922,206],[349,623]]` (SVM) |
| BR-BERT | `[[902,226],[267,705]]` | `[[884,244],[247,725]]` |
| M-BERT-BR | `[[778,350],[179,793]]` | `[[837,291],[199,773]]` |
| M-BERT (transfer) | `[[837,291],[207,765]]` | `[[819,309],[211,761]]` |
| M-BERT (zero-shot) | `[[940,188],[657,315]]` | `[[936,192],[676,296]]` |

Figuras em `results/figures/cm_binary_*.png`. (As matrizes "Artigo" são lidas das figuras do paper;
o baseline do artigo é o **AutoML**, o reproduzido é o **SVM** — modelos diferentes.)

**Leitura** (Figuras R.1 e R.2): todos os 5 modelos reproduzem o artigo **dentro de ~1,6 ponto de macro-F1**. O padrão
qualitativo se mantém: BR-BERT e M-BERT-BR ficam no mesmo patamar (~0.76), o transfer ligeiramente abaixo e o
**zero-shot bem inferior**, com recall da classe tóxica baixíssimo (~0.30, próximo do ~0.32 do artigo) —
o que confirma a necessidade de dados na língua-alvo.

![Comparação de macro-F1 binário entre artigo e reprodução](results/figures/comparison/compare_macro_f1_bars.png)

**Figura R.1 — macro-F1 binário (Artigo × Reprodução).** Barras cinzas: valores do artigo
(Tabelas 7–11); azuis: reproduzidos. Os cinco modelos ficam a ≤ 1,6 ponto do artigo, e a diferença
entre BR-BERT e M-BERT-BR fica dentro da margem de ruído (~1–2 pontos).

![Matrizes de confusão binárias, artigo e reprodução lado a lado](results/figures/comparison/compare_fig2_binary_cm.png)

**Figura R.2 — Matrizes de confusão binárias (Figura 2 do artigo × reprodução).** Cada linha é um
modelo; à esquerda o artigo, à direita a reprodução, no formato `[[TN, FP], [FN, TP]]` e mesma escala
de cor. O baseline do artigo é o BoW+AutoML e o reproduzido é o BoW+SVM (modelos distintos na mesma
posição). A distribuição de falsos negativos é equivalente em todos os modelos; o zero-shot mantém o
alto número de falsos negativos na classe tóxica (657 no artigo, 676 reproduzido), evidenciando a
necessidade de dados na língua-alvo.

### 8.2 Análise de erro — FN-rate por categoria (Tabela 12)

FN-rate = (exemplos da categoria preditos como **não-tóxico**) / (exemplos da categoria no *test*).

| Categoria | Artigo | Reproduzido |
|---|---:|---:|
| LGBTQ+phobia | 7/35 (0.20) | 7/35 (0.20) |
| Insult | 67/448 (0.15) | 74/444 (0.17) |
| Xenophobia | 13/19 (0.68) | 9/19 (0.47) |
| Misogyny | 7/45 (0.15) | 7/44 (0.16) |
| Obscene | 117/701 (0.17) | 119/699 (0.17) |
| Racism | 8/17 (0.47) | 7/17 (0.41) |

**Leitura:** o padrão central é reproduzido — **classes minoritárias (racism 0.41, xenophobia 0.47)
têm os maiores FN-rate**, enquanto as **majoritárias (obscene 0.17, insult 0.17) têm os menores** —
confirmando que "a proporção de falsos negativos é inversamente proporcional ao número de exemplos
da classe". **Ressalva honesta:** os *denominadores* (positivos por categoria no *test*) ficam
**próximos, mas não idênticos** aos do artigo — insult 444 vs 448, misogyny 44 vs 45 e obscene 699 vs
701, enquanto LGBTQ+phobia (35), racism (17) e xenophobia (19) coincidem. Essa diferença vem da contagem de positivos
por categoria fina, dependente da normalização de texto e da versão do `ToLD-BR.csv`, e não da
metodologia.

### 8.3 Curva de aprendizado (Figura 3)

Média de 3 repetições por fração; M-BERT-BR, 1 época; avaliação no *test*. Figura em
`results/figures/learning_curve.png`.

| Fração | # treino | macro-F1 | recall+ | recall− | precision+ | precision− |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 1.890 | 0.584 | 0.409 | 0.798 | 0.641 | 0.617 |
| 20% | 3.780 | 0.679 | 0.689 | 0.673 | 0.648 | 0.717 |
| 30% | 5.670 | 0.708 | 0.698 | 0.721 | 0.685 | 0.740 |
| 40% | 7.560 | 0.720 | 0.747 | 0.698 | 0.681 | 0.762 |
| 50% | 9.450 | 0.734 | 0.776 | 0.699 | 0.689 | 0.784 |
| 60% | 11.340 | 0.645 | 0.577 | 0.769 | 0.690 | 0.716 |
| 70% | 13.230 | 0.740 | 0.791 | 0.695 | 0.691 | 0.794 |
| 80% | 15.120 | 0.736 | 0.806 | 0.676 | 0.682 | 0.802 |
| 90% | 17.010 | 0.747 | 0.799 | 0.702 | 0.698 | 0.802 |
| 100% | 18.900 | 0.749 | 0.818 | 0.689 | 0.694 | 0.816 |

**Leitura:** reproduz a conclusão do artigo. Com poucos dados (10%), o **recall da classe tóxica é
baixíssimo (0.41)** — o modelo só acerta a majoritária; conforme os dados crescem, o **recall+
sobe (→0.82)**, o **recall− cai (→0.69)** e a **precisão sobe em ambas**. A performance estabiliza
por volta de **30–40% dos dados (~5,7–7,6K exemplos)** (Figura R.3), coerente com a afirmação do artigo de que
"ao menos 6K exemplos parecem necessários". (O ponto de 60% mostra uma queda — uma das 3 repetições
degenerou fortemente rumo à classe majoritária: no log de treino, a loss dessa repetição fica estagnada
enquanto as outras duas convergem. Não foi um colapso completo — a média de precision+ persistida (0.690)
exclui uma repetição com zero positivos previstos. Resíduo da instabilidade de fine-tuning com 1 época;
o JSON guarda apenas médias por fração, não métricas por repetição.)

**Curva de aprendizado (Figura 3 do artigo × reprodução).** O notebook `04` reproduz o **mesmo código
de plotagem do autor** (`experiments/learning_curve.ipynb`, modernizado com diff mínimo) aplicado aos
nossos dados (linhas recall/precision/F1). A comparação direta com o artigo está na **Figura R.3b**
(abaixo), que sobrepõe as curvas reais.
A forma é a mesma: com poucos dados o
modelo só acerta a classe majoritária; conforme o volume cresce, o **recall da classe tóxica sobe**
(~0.41 → 0.82), o **recall da não-tóxica cai** (~0.80 → 0.69) e a **precisão sobe em ambas**,
estabilizando em torno de **5,7–7,6K exemplos**.

> **Sobre a escala do eixo-x:** nos dados publicados pelo artigo (`experiments/data/learning_curve.json`),
> os pontos de **10% a 90% são frações do mesmo pool de treino de 18.900** usado nesta reprodução
> (totais 1.890, 3.780, …, 17.010); **apenas o ponto de 100% usa o dataset completo (21.000 exemplos —
> o que inclui os 2.100 de teste, com vazamento restrito a esse ponto)**. Esta reprodução mantém todas
> as frações — inclusive 100% — sobre os 18.900 de treino, com o teste sempre disjunto (o modelo nunca
> vê o teste em fração alguma). Por isso o eixo do artigo termina em ~21.000 e o nosso em 18.900 — a
> diferença de escala restringe-se ao ponto final. A **Figura R.3b** abaixo sobrepõe as duas **curvas
> reais** com o eixo em **% dos dados de treino**, o que neutraliza essa diferença.

![Curvas de aprendizado reais sobrepostas (artigo × reprodução)](results/figures/comparison/compare_fig3b_overlay_learning_curve.png)

**Figura R.3b — Sobreposição das curvas reais (artigo × reprodução), eixo em % dos dados de treino.**
As duas curvas têm a mesma forma e **convergem a 100%** (recall da classe tóxica ≈ 0.82 em ambas). As
diferenças são de **instabilidade estocástica** do fine-tuning de 1 época: a curva do artigo **começa
mais baixa a 10%** (recall+ ≈ 0.15 — mais repetições colapsadas) e a reproduzida tem a **queda pontual
em 60%** (uma repetição fortemente degenerada — ver a leitura acima); em ambos os casos é ruído, não divergência
de tendência. (O `learning_curve.json` do autor confirma colapsos completos no artigo: há valores 0 de
F1 entre as 3 repetições.)

### 8.4 Classificação multi-rótulo (Figura 4)

Protocolo: split 90/10 (`random_state=42`) sobre `ToLD-BR.csv`; mBERT (3 épocas, limiar 0.5) comparado ao baseline
BoW+SVM (Binary Relevance, 1 SVM por rótulo). A tabela abaixo confronta as duas métricas oficiais (Hamming loss e Average Precision) com o artigo.

| Modelo | Métrica | Artigo | Reproduzido | Δ |
|---|---|---:|---:|---:|
| Baseline BoW+SVM | Hamming loss | 0.08 | 0.069 | −0.011 |
| Baseline BoW+SVM | Average Precision | 0.20 | 0.209 | +0.009 |
| mBERT | Hamming loss | 0.07 | 0.067 | −0.003 |
| mBERT | Average Precision | 0.19 | **0.276** | **+0.086** |

**Matrizes por rótulo (mBERT)** `[[TN, FP], [FN, TP]]` (Figura 4):

| Rótulo | Artigo (mBERT) | Reproduzido (mBERT) |
|---|---|---|
| LGBTQ+phobia | `[[2072,2],[25,1]]` | `[[2066,8],[12,14]]` |
| Obscene | `[[1430,38],[427,205]]` | `[[1177,291],[148,484]]` |
| Insult | `[[1635,41],[290,134]]` | `[[1528,148],[169,255]]` |
| Racism | `[[2089,0],[11,0]]` | `[[2089,0],[11,0]]` |
| Misogyny | `[[2057,0],[39,4]]` | `[[2055,2],[32,11]]` |
| Xenophobia | `[[2081,0],[19,0]]` | `[[2081,0],[18,1]]` |

**Leitura:** Hamming loss e o baseline batem com o artigo. A conclusão qualitativa se mantém —
**classes raras (racism, xenophobia, misogyny, LGBTQ+phobia) são quase totalmente classificadas
como negativas** (racism TP 0, xenophobia TP 1, misogyny TP 11, LGBTQ+phobia TP 14), enquanto
**obscene e insult acertam parte dos positivos**. Porém, o **mBERT reproduzido superou o do artigo
em AP** (0.276 vs 0.19; muito mais TPs em obscene 484 vs 205 e insult 255 vs 134). Esta é a **única
conclusão do artigo não reproduzida**: no artigo o baseline ficava ≳ BERT em AP, ao passo que aqui o BERT supera o
baseline. A investigação dessa divergência está na §8.5 (matrizes lado a lado na Figura R.4).

![Matrizes de confusão multi-rótulo, artigo e reprodução lado a lado](results/figures/comparison/compare_fig4_multilabel_cm.png)

**Figura R.4 — Matrizes de confusão multi-rótulo (Figura 4 do artigo × reprodução).** Seis rótulos;
à esquerda o artigo, à direita a reprodução (ambos mBERT). As **classes raras** (racism, xenophobia,
misogyny, LGBTQ+phobia) são quase totalmente preditas como negativas nas duas execuções; nas
**classes frequentes** (obscene, insult) o mBERT reproduzido prevê muito mais positivos verdadeiros
(obscene TP 484 vs 205; insult 255 vs 134), o que eleva o Average Precision — porém ao custo de mais
falsos positivos (obscene FP 291 vs 38), um *trade-off* de precisão por recall, não superioridade
geral (ver §8.5).

### 8.5 Por que o mBERT multi-rótulo divergiu do artigo

No multi-rótulo, o modelo é **idêntico** ao do artigo (mBERT completo), o baseline coincide
(AP 0.209 vs 0.20), e dados/limiar/warmup são os mesmos (o código multi-rótulo do artigo não passa
`warmup_ratio`, logo usa o **default 0.06 do `simpletransformers`** — o mesmo desta reprodução). O
que difere é a **dinâmica de treino**, e há evidência concreta no próprio artefato do artigo:

- O treino multi-rótulo do artigo (`experiments/classification.ipynb`) usou **apex em fp16
  (opt_level O1)** e registrou, no início, **vários *gradient overflows* que fizeram o otimizador pular passos**
  (`Gradient overflow. Skipping step, loss scaler → 32768 → 16384 → 8192`). Como cada passo pulado
  é uma atualização de peso descartada, o modelo do artigo treinou de forma **menos efetiva e
  ficou conservador** (previu quase tudo negativo: alta precisão, baixo recall). Já o **torch AMP
  moderno** desta reprodução faz o *loss scaling* de forma robusta, treina mais e, com isso, o modelo
  **aprende a prever as classes frequentes** (obscene, insult), elevando recall e AP.

**Por que a divergência aparece só no multi-rótulo (e não no binário):** o multi-rótulo tem
desbalanceamento extremo (racism ~11 positivos em 2.100; xenophobia ~19). O ótimo trivial
("prever tudo negativo") está muito próximo, com *loss* baixíssima — escapar dele e passar a prever
as classes frequentes exige um treino **efetivo**. Assim, uma diferença **sistemática** de
efetividade do treino (apex pulando passos vs AMP estável) determina em qual regime o modelo termina.
No **binário** (≈44% de positivos no treino; ≈46% no teste), longe desse ótimo trivial, o resultado é robusto a essas
diferenças — e de fato **reproduziu o artigo** nos 5 modelos.

**Nuance importante (o modelo não é uniformemente "melhor"):** ele **troca precisão por recall** —
gera muito mais falsos positivos (obscene FP 291 vs 38; insult FP 148 vs 41). O **Hamming loss
empatou** (0.067 vs 0.07): o total de erros por célula é parecido, apenas redistribuído de FN para
FP. O "supera" é **específico do Average Precision**, que premia recall; é um **ponto de operação
diferente** na curva precisão/recall, não um modelo objetivamente superior.

**Status da hipótese:** a explicação (diferença de *stack* de treino/numérico — apex fp16 com passos
pulados em 2020 vs torch AMP estável) é **bem fundamentada, não uma prova**. Fechá-la exigiria
reexecutar o stack exato de 2020 (apex + `simpletransformers` da época), fora do escopo.

> **Nota de reprodutibilidade:** uma investigação anterior havia feito uma **ablação por *seed* e por
> hiperparâmetros** (4 seeds + a configuração do artigo, batch 8 / seq 512) indicando que a divergência
> é **robusta** — não desaparece ao trocar a *seed* nem ao adotar os hiperparâmetros do artigo. Esse
> experimento **não faz parte do pipeline de notebooks** (era um script à parte) e seu artefato foi
> removido na limpeza que precedeu esta execução, de modo que **seus números não são reportados aqui**.
> Se a robustez precisar ser quantificada no texto acadêmico, a ablação pode ser **reincorporada como
> uma célula/notebook reproduzível** (ver §11, opcional).

---

## 9. Conclusões do artigo verificadas

| Conclusão do artigo | Reproduzida? | Evidência |
|---|---|---|
| BERT supera o baseline na tarefa binária | ✅ | BR-BERT 0.765 / M-BERT-BR 0.766 / transfer 0.752 vs baseline BoW+SVM 0.730 |
| Modelo monolíngue ≳ multilíngue | ≈ (empate) | BR-BERT 0.765 ≈ M-BERT-BR 0.766 — diferença (0.0016) **dentro do ruído**; a direção do artigo não se confirma nem se inverte de forma significativa |
| Transfer não melhora sobre o monolíngue | ✅ | transfer 0.752 < M-BERT-BR 0.766 / BR-BERT 0.765 |
| Zero-shot é bem inferior (dados na língua importam) | ✅ | zero-shot 0.544 (recall+ 0.305) |
| FN-rate maior nas classes minoritárias | ✅ | racism 0.41 / xenophobia 0.47 altos; obscene 0.17 / insult 0.17 baixos |
| ~6K exemplos necessários para resultados confiáveis | ✅ | estabiliza em 30–40% (~5,7–7,6K) |
| Multi-rótulo é difícil; classes raras ~ tudo negativo | ✅ | racism TP 0, xenophobia TP 1, misogyny TP 11 |
| Baseline ≳ BERT no multi-rótulo (AP) | ❌ (não reproduzido) | nosso mBERT **superou** o baseline (AP 0.276 vs 0.209) — ver §8.5 |

> **Nota sobre "monolíngue vs multilíngue":** mesmo no artigo essa ordenação é frágil — os autores
> escolhem o **M-BERT-BR** (multilíngue) como seu *melhor modelo* na análise de falsos negativos
> (Tabela 12). Nesta reprodução os dois empatam dentro do ruído, o que é **coerente** com essa
> fragilidade, não uma contradição do artigo.

---

## 10. Limitações e ameaças à validade

- **Baseline:** o artigo usa BoW+AutoML (0.74); aqui o baseline executável é **BoW+SVM (0.730)**, pois
  o auto-sklearn é Linux-only. O número do AutoML real (0.733, via Docker/WSL) está **documentado**
  no notebook 02, mas **não é um artefato desta execução**.
- **Variabilidade estocástica:** sem *seeds* múltiplas para os modelos binários (exceto a curva,
  com 3 repetições). Neste ambiente o pipeline se mostrou determinístico entre execuções, mas
  diferenças de versão de biblioteca/CUDA em outras máquinas podem mover ±1–2 pontos.
- **Hiperparâmetros não totalmente especificados no artigo** (lr, warmup, batch): adotaram-se os
  defaults do `simpletransformers` por serem o que o código original usava — mas isso é uma inferência.
- **OLID com 13.240 exemplos** (treino do OLID), seguindo o código liberado, enquanto o **texto** do
  artigo descreve treino+teste (~14.100) — divergência herdada do repositório (§3, desvio #6).
- **Análise de erro:** depende de **normalização de texto** para alinhar predições e categorias finas;
  ~10 linhas têm encoding corrompido irrecuperável, e os denominadores por categoria ficam próximos
  mas **não idênticos** aos do artigo (§8.2).
- **Concordância entre anotadores (Krippendorff α, Tabela 5):** **não reproduzida numericamente** —
  o notebook 01 gera apenas os arquivos de entrada (`*_alpha.tsv`) e documenta um cálculo externo
  (`mwetoolkit3/bin/kappa.py`) que não é executado.
- **Ablação multi-rótulo (§8.5):** o experimento de robustez por *seed*/hiperparâmetros não faz
  parte do pipeline de notebooks e seu artefato foi removido; a explicação da divergência permanece
  qualitativa (porém bem fundamentada).
- **OLID via `tweet_eval`:** texto já levemente normalizado pela Cardiff (mentions→`@user`,
  links→`http`); equivalente, mas não byte-a-byte ao `olid-training-v1.0.tsv` original.

---

## 11. Como reproduzir (passo a passo)

A reprodução roda **inteiramente pelos notebooks** em `reproduction/notebooks/`, a partir da
pasta `reproduction/`. Cada alteração em relação ao notebook original do autor está documentada e
justificada em `reproduction/notebooks/CHANGES.md` (categorias **T** tecnologia descontinuada,
**B** bug, **A** fidelidade ao artigo).

```powershell
# 1. Ambiente (venv + torch CUDA + deps) e kernel Jupyter
.\setup_env.ps1
$py = "$env:LOCALAPPDATA\toldbr-repro-venv\Scripts\python.exe"
& $py -m ipykernel install --user --name toldbr-repro

# 2. Rodar o nucleo (pipeline do artigo), na ordem (kernel "toldbr-repro"), via nbconvert:
$k = "--ExecutePreprocessor.kernel_name=toldbr-repro"
& $py -m nbconvert --to notebook --execute --inplace $k notebooks\01_generate_dataset.ipynb   # gera/valida dados + TSVs IAA
& $py -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 $k notebooks\02_automl_baseline.ipynb  # baseline BoW+SVM
& $py -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1   $k notebooks\03_classification.ipynb     # 5 modelos + curva + multi-rotulo (~1,5-2 h GPU)
& $py -m nbconvert --to notebook --execute --inplace $k notebooks\07_error_analysis.ipynb     # Tabela 12 (FN-rate); requer o 03
& $py -m nbconvert --to notebook --execute --inplace $k notebooks\08_compare_figures.ipynb    # figuras comparativas Artigo x Reproducao; requer o 03
# Complementares (fora do escopo do artigo): 04_learning_curve.ipynb (Figura 3) e
# 05_wordcloud.ipynb (nuvens de palavras). 06_example_pretrained.ipynb: demo do modelo
# pre-treinado. Link do Google Drive (2020) FORA DO AR => notebook NAO-EXECUTAVEL; nao
# gera artefato de comparacao (ver §7.3).
```

Em Linux/macOS (ou via conda), veja as instruções equivalentes em
[`reproduction/README.md`](README.md#setup).

> A **Tabela 12** (FN-rate por categoria) não tinha notebook do autor. Foi adicionado o notebook
> **novo** `07_error_analysis.ipynb` (extensão), que a reproduz a partir das predições do
> `03_classification` (`results/preds_binary_mbert_br.csv`).

Tempo total aproximado de GPU na RTX 4070 Ti SUPER: **~1,5–2 h** (dominado pelo notebook 03).

**Opcional (para fechar a §8.5):** a ablação multi-rótulo por *seed*/hiperparâmetros pode ser
reincorporada como célula reproduzível no notebook 03 (gerando `results/multilabel_ablation.json`),
se ela for necessária ao texto acadêmico.

---

## 12. Artefatos gerados

```
reproduction/
  README.md                          # visão geral (notebook-centric)
  REPORT.md                          # este relatório
  setup_env.ps1                      # cria o venv (Windows) + instala deps
  setup_env.sh                       # cria o venv (Linux/macOS) + instala deps
  environment.yml                    # ambiente conda multiplataforma
  requirements.txt                   # dependências dos notebooks
  requirements-lock.txt              # versões exatas (pip freeze) do ambiente de referência
  notebooks/                         # notebooks modernizados do autor (01..08)
    01_generate_dataset.ipynb        #   gera/valida ToLD-BR.csv + arquivos IAA
    02_automl_baseline.ipynb         #   baseline BoW+SVM (binário e multi-rótulo)
    03_classification.ipynb          #   5 modelos binários + curva + multi-rótulo
    04_learning_curve.ipynb          #   [complementar] Figura 3 (consome learning_curve.json do 03)
    05_wordcloud.ipynb               #   [complementar] nuvens de palavras por categoria + Jaccard
    06_example_pretrained.ipynb      #   [complementar] demo do modelo pré-treinado (NÃO-EXECUTÁVEL: link morto)
    07_error_analysis.ipynb          #   Tabela 12 — FN-rate por categoria (NOVO, extensão)
    08_compare_figures.ipynb         #   figuras comparativas Artigo × Reprodução (NOVO, extensão)
    CHANGES.md                       #   catálogo de TODAS as mudanças (T/B/A) por notebook
    README.md                        #   guia dos notebooks
  results/                           # TODOS os artefatos abaixo são de uma única execução (2026-06-21)
    binary_baseline_bow_svm.json     # baseline SVM + matriz de confusão (persistido pelo 03 — ver §5.3)
    binary_br_bert.json
    binary_mbert_br.json
    binary_mbert_transfer.json
    binary_mbert_zeroshot.json
    preds_binary_*.csv               # predições por modelo (a do M-BERT-BR alimenta a Tab. 12)
    error_analysis_fn_rate.json      # Tabela 12 (FN-rate)
    learning_curve.json              # curva de aprendizado (Fig. 3)
    multilabel_baseline_bow_svm.json # (persistido pelo 03 — ver §5.3)
    multilabel_mbert.json
    ToLD-BR_generated.csv            # prova de reprodução (== ToLD-BR.csv do repo)
    {categoria}_alpha.tsv            # entradas para o cálculo de IAA (externo) — como no original
    figures/
      cm_binary_*.png                # matrizes de confusão binárias (Fig. 2)
      cm_multilabel_*.png            # matrizes de confusão multi-rótulo (Fig. 4)
      learning_curve.png             # curva de aprendizado (Fig. 3) [complementar, notebook 04]
      comparison/                    # figuras comparativas Artigo × Reprodução (notebook 08)
        compare_fig2_binary_cm.png   #   matrizes binárias lado a lado
        compare_fig4_multilabel_cm.png  # matrizes multi-rótulo lado a lado
        compare_fig3b_overlay_learning_curve.png # curvas reais sobrepostas (% dos dados)
        compare_macro_f1_bars.png    #   barras de macro-F1 binário
```
