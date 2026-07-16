# Catálogo de mudanças — notebooks modernizados (ToLD-Br)

## Kit de reprodução (2026-07-15) — remoções para o pacote publicável
- **08_compare_figures:** removida a célula da Fig.3 (`compare_fig3_learning_curve.png`, não usada no artigo) — única dependência da pasta do paper e do `pymupdf`; as 4 figuras do artigo seguem geradas.
- Removidas do kit as pastas `arXiv-2010.04543/` (paper original) e `docs/` (rascunhos/spec/planos).

Cada notebook em `reproduction/notebooks/` foi modernizado a partir do original do autor
(`experiments/*.ipynb`, `model/example.ipynb`) seguindo o princípio **diff mínimo**: só foi
alterado o necessário, classificado em três categorias:

- **(T) Tecnologia descontinuada** — Google Colab, simpletransformers/apex, auto-sklearn (Linux-only), caminhos do Drive, APIs antigas.
- **(B) Bug** que impedia a execução.
- **(A) Fidelidade ao artigo** — usar os modelos/baseline do artigo (mBERT/BERTimbau, não distilbert).

## 01_generate_dataset.ipynb

Origem: experiments/generate_dataset.ipynb. Gera o ToLD-BR multi-rótulo (contagens de voto por categoria) e os TSVs por categoria para o IAA. A lógica de agregação de votos (3 anotadores -> contagens 0..3; toxic=1 se qualquer rótulo>0) foi preservada 100%.

| Nº | Célula (original) | Cat | Antes | Depois | Justificativa | Impacto |
|----|-------------------|-----|-------|--------|---------------|---------|
| 1 | leitura do CSV (`cell-2`) | T | `pd.read_csv("../data/told-br/ToLD-BR_alpha.csv")` | `pd.read_csv(ALPHA_CSV)` | Estrutura de dados achatada; notebook roda de `reproduction/notebooks/`. Arquivo de anotações vive na raiz como `ToLD-BR_alpha.csv`. | Nenhum (mesma leitura). |
| 2 | escrita do CSV (`cell-5`) | T | `to_csv("../data/told-br/ToLD-BR.csv")` | grava `RESULTS/ToLD-BR_generated.csv` + valida igualdade com `MAIN_CSV` via `assert_frame_equal` | Não sobrescrever o artefato versionado; adicionar prova de reprodução. | Nenhum na geração; +validação automática. |
| 3 | TSVs por categoria (`cell-7`) | T | `to_csv(f"{category}_alpha.tsv")` (cwd) | `RESULTS / f"{category}_alpha.tsv"` | Centralizar artefatos em `results/`; notebook roda de pasta de notebooks. | Nenhum no conteúdo; muda o diretório de saída. |
| 4 | sanidade (`cell-6`) | A | `data[...]["toxic"].value_counts()` (display) | `assert` de 21000 / 9255 / 11745 | Fidelidade ao artigo: fixar a Tabela 6 como teste reproduzível em vez de inspeção visual. | Nenhum no dataset; +verificação automática. |

Observações:
- Tecnologia descontinuada (T): caminhos da estrutura antiga `data/told-br/` substituídos pelas variáveis de path do setup padrão (`ALPHA_CSV`, `MAIN_CSV`, `RESULTS`).
- Comandos de ambiente herdados do Colab (`!pip`, `!ls`, `files.download`, `drive.mount`, etc.): este notebook não continha nenhum, então nada a remover.
- As colunas de categoria do `ToLD-BR.csv` versionado são `float64` (a agregação usa `np.sum`); a célula gerada produz os mesmos floats, logo `assert_frame_equal` (com `check_dtype=False`, defensivo) passa.
- Sanidade (Tabela 6): 21.000 linhas; 9.255 tóxicos / 11.745 não-tóxicos. Prova de reprodução: `ToLD-BR_generated.csv` == `ToLD-BR.csv` (repo).

## 02_automl_baseline.ipynb (origem: experiments/automl.ipynb)

| Nº | célula (original) | cat | Antes | Depois | justificativa | impacto |
|----|-------------------|-----|-------|--------|---------------|---------|
| #1 | 0 (installs) | T | `!apt-get install build-essential swig` + `!pip install auto-sklearn` + `!pip install liac-arff` | célula `%pip install scikit-learn pandas matplotlib` comentada | auto-sklearn 0.x não instala/roda em Windows/Python atual e exige toolchain C | nenhum no resultado; só dependências |
| #2 | 1 (carga binária + fit auto-sklearn) | T,A | `pd.read_csv("ptbr_train_1annotator.csv")` + `AutoSklearnClassifier().fit()` | `load_binary_split()` lendo `DATA_ZIP` + `CountVectorizer` + `SVC().fit()` | CSVs agora no ZIP `experiments/data/1annotator.zip`; auto-sklearn substituído por BoW+SVM (família equivalente) | macro-F1 0.728 neste notebook, treino só em train=16.800 (vs 0.733 auto-sklearn reproduzido / 0.74 artigo); o JSON persistido (0.730) é o do baseline do `03`, que sobrescreve o arquivo — ver nota em "Artefatos gerados" |
| #2b | 2 (sprint_statistics) + 3 (predict) | T | `automl.sprint_statistics()` / `automl.predict()` | `print(clf)` / `clf.predict()` | SVC não tem `sprint_statistics`; objeto renomeado `automl`->`clf` | nenhum; predição idêntica |
| #3 | 6 (imports+gdown) + 7 (download) + 8 (read+split) | T,A | `import autosklearn` + `import gdown` + `gdown.download(<gdrive>)` + `pd.read_csv` + `int(bool(v))` | leitura de `MAIN_CSV` + binarização `>0` + `train_test_split` 0.9/0.5 `random_state=42` (preservado) | gdown/Drive e auto-sklearn não se aplicam ao ambiente local; CSV já versionado | split idêntico ao original (18900/1050/1050) |
| #4 | 11 (`AutoSklearnClassifiervb()` fit) | B,A | `autosklearn...AutoSklearnClassifiervb().fit(X, y2d)` (método INEXISTENTE) | Binary Relevance: um `SVC` por rótulo (loop em `LABELS`) | (B) corrige typo que jamais rodaria; (A) SVC não aceita alvo multi-rótulo direto | Hamming ~0.065, AP ~0.199 neste notebook (teste=1.050, split 0.9/0.5 do autor; vs 0.0838 / 0.2019 do auto-sklearn original); o JSON persistido (0.069 / 0.209) é o do baseline do `03` — ver nota em "Artefatos gerados" |
| doc | (nova) | doc | — | célula markdown com números auto-sklearn do artigo (binário macro-F1 0.74 / best val 0.7546 / reproduzido 0.733; multi Hamming 0.0838, AP 0.2019) + instruções WSL/Docker | preservar o método original e torná-lo replicável | nenhum (documentação) |

**Artefatos gerados:** `reproduction/results/binary_baseline_bow_svm.json`, `reproduction/results/multilabel_baseline_bow_svm.json`.

> **Nota de proveniência:** o notebook `03` também grava esses dois arquivos, pois o
> `classification.ipynb` do autor inclui os próprios baselines (binário treinado em
> train+validation=18.900 → macro-F1 0.730; multi-rótulo 90/10, teste=2.100 → Hamming 0.069 /
> AP 0.209). Na ordem de execução `01 → 02 → 03`, os JSONs persistidos — usados no `REPORT.md`
> e no artigo de reprodução — são os do `03`; os resultados do `02` (0.728; 0.065/0.199,
> protocolos do `automl.ipynb`) ficam registrados apenas nas saídas do próprio notebook.

## 03_classification.ipynb (origem: experiments/classification.ipynb)

Reproduz os três blocos do artigo: binário (5 modelos), curva de aprendizado (M-BERT-BR, 10%..100%) e multi-rótulo (BoW+SVM Binary Relevance vs mBERT). Porta o miolo simpletransformers/Colab para HuggingFace Trainer preservando a classe `Experiment` do autor (assinaturas e métodos `__load_dataset`, `__preprocess`, `describe_data`, `train`, `eval`). Artefatos em `reproduction/results/*.json`, predições `preds_*.csv` e figuras `reproduction/results/figures/`.

| Nº | Célula original | Cat | Antes | Depois | Justificativa | Impacto |
|---|---|---|---|---|---|---|
| 1 | cell-1..4 | T | !nvidia-smi, drive.mount, %%writefile setup.sh (apex+simpletransformers), !sh setup.sh | removidos | apex/Drive/Colab não rodam em 2026 | nenhum no resultado |
| 2 | cell-5 | T | imports simpletransformers | transformers (Trainer) + datasets; sklearn/nltk/unidecode preservados | simpletransformers descontinuado | tecnologia |
| 3 | cell-7 (Experiment) | T | miolo de train()/eval() simpletransformers | HF Trainer; assinaturas (__load_dataset, __preprocess, describe_data, train, eval) preservadas; eval imprime macro-F1 (era f1 binário) | porte do treino/avaliação simpletransformers->HF Trainer; macro-F1 é a métrica do artigo | tecnologia |
| 4 | cell-7 | A | distilbert-base-multilingual-cased | mBERT (bert-base-multilingual-cased) / BERTimbau (neuralmind/bert-base-portuguese-cased) via novo param pretrained_name | fidelidade ao artigo (única (A) que altera resultado) | alto — reproduz os números do artigo |
| 5 | cell-7/13 | B/T | — | warmup_ratio=0.06 adicionado | default do simpletransformers; sem ele o transfer colapsa na majoritária | alto — transfer 0.349 -> 0.752 (execução vigente, 2026-06-21; uma execução anterior dera 0.756) |
| 6 | cell-7/13 | T | max_seq_len 512; train_batch_size 50 | max_seq_len 128; train_batch_size 32 | memória/desempenho; tweets cabem em 128 tokens | desprezível |
| 7 | cell-7 (__init__/__preprocess) | B | indentação: __preprocess do train/test estava aninhado no else do bloco de stopwords | desaninhado | bug | preprocessing aplicado quando do_preprocessing=True |
| 8 | cell-7 (__load_dataset) | T | OLID: pd.read_csv(drive/...olid-training-v1.0.tsv) | load_dataset("cardiffnlp/tweet_eval","offensive") train+validation=13240 | TSV no Drive; datasets 4.x exige repo namespaced | nenhum — mesmo OLID |
| 9 | cell-7 (download_results) | T | files.download | save_results: json (métricas) + csv (predições) em RESULTS + matriz de confusão em FIGURES | Colab indisponível | tecnologia |
| 10 | cell-10 | B | train_amount=1.0 | data_amount=1.0 na instanciação do baseline | o construtor espera data_amount (TypeError) | bug — baseline passa a rodar |
| 11 | cell-13..18 | A | 1 instância distilbert + zip/ls/load do outputs | 4 modelos BERT do artigo (BR-BERT, M-BERT-BR, transfer, zero-shot) via a classe Experiment | produzir os 5 modelos binários (Tab. 7-11) | alto — reproduz a tabela binária |
| 12 | cell-20 | B | variável `it` indefinida no laço da curva | índice sequencial run=1..30 | NameError na 1a iteração | bug — curva passa a rodar |
| 13 | cell-20 | T/A | distilbert; files.download(learning_curve.json) | M-BERT-BR; grava em RESULTS | Drive/Colab; fidelidade | reproduz a Figura 3 |
| 14 | cell-22 | T | drive/...multilabel_grouped.csv | MAIN_CSV (ToLD-BR.csv) binarizado (>0->1) | arquivo no Drive | nenhum — mesmos 6 rótulos |
| 15 | cell-24..26 | B | variável `score` indefinida no baseline multi-rótulo | removida; Binary Relevance + métricas reais (hamming/AP/cm por rótulo) salvas em RESULTS | NameError; to_csv local | bug — baseline passa a rodar |
| 16 | cell-28..43 | T | MultiLabelClassificationModel (train_batch_size 8); files.download | HF Trainer (problem_type="multi_label_classification", per_device_train_batch_size=16); RESULTS/FIGURES | porte do treino multi-rótulo simpletransformers->HF Trainer; mBERT do artigo preservado; batch 8->16 (adaptação operacional ao ambiente, análoga à #6) | tecnologia |

Nota de dependência: `Unidecode` (importável como `unidecode`) — usado no preprocessing e originalmente instalado pelo `setup.sh` do Colab — foi instalado no venv durante a validação (estava ausente); nenhuma mudança de metodologia, apenas restauração de dependência do original.

## 04_learning_curve.ipynb (origem: experiments/learning_curve.ipynb) — Figura 3

| Nº | Célula original | Cat | Antes | Depois | Justificativa | Impacto |
|----|-----------------|-----|-------|--------|---------------|---------|
| 1 | cell1/cell2/cell3 + cell5 | T | Lia experiments/data/learning_curve.json (chaves "1".."10", métricas em listas: f1_overall/precision_overall/recall_overall, *_positive/*_negative, confusion_matrix) | lê RESULTS/learning_curve.json (fallback experiments/data) com chaves de fração "0.1".."1.0" e métricas em float (macro_f1, f1_pos/f1_neg, recall_pos/neg, precision_pos/neg); itera frações ordenadas; eixo X = n_train (se houver) ou int(POOL_SIZE*frac), POOL_SIZE=18900 (= len(pool) train+validation); F1 global = macro_f1; precision/recall global = média das duas classes | Schemas de JSON incompatíveis: o notebook 03 grava médias por fração, sem *_overall nem confusion_matrix. Decisão de projeto: adaptar o parsing no 04, não mexer no 03 | Nenhum no artigo — mesma Figura 3; possível diferença numérica só por o JSON ser do 03. Eixo X passa de tamanho do teste (constante) para # exemplos de treino, como no notebook 03 |
| 2 | cell6 | T | [(2*p*r)/(p+r) for r,p in zip(precision_positive,recall_positive)] | _f1(p,r) com guarda p+r>0; f1_negative análogo adicionado | Evita 0/0 na menor fração (modelo tende a prever a classe majoritária); mantém a fórmula do autor | Nenhum — idêntico onde p+r>0 |
| 3 | cell7 | T | savefig("recall-precision-positiveclass.pdf") na pasta corrente, f1 recalculado inline | FIGURES/*.pdf + *.png, usa f1_positive (com guarda); estilo do autor preservado | Padronizar artefatos, gerar PNG e evitar recálculo inline sem guarda | Nenhum — mesma curva |
| 4 | cell8 | T | savefig("recall-precision-negativeclass.pdf"), f1 inline | FIGURES/*.pdf + *.png, usa f1_negative (com guarda) | Padronizar artefatos e gerar PNG | Nenhum — mesma curva |
| 5 | cell9 | T | FN/FP em contagem absoluta lidos de confusion_matrix por rep | FN-rate=1-recall+, FP-rate=1-precision+; FIGURES/fn-fp.pdf + .png | Novo JSON não tem confusion_matrix; derivam-se as taxas que a figura ilustra | Nenhum no comportamento qualitativo; eixo Y muda de contagem para taxa |
| 6 | cell10 | T | data["1"]["confusion_matrix"] | print defensivo (só se a chave existir) | Chave e índice "1" inexistentes no JSON do 03 | Nenhum — print é diagnóstico opcional |

Notas: cell0 (imports, incl. o import não usado sklearn.model_selection.learning_curve) e cell4 (rcParams font 25) preservados sem alteração, conforme o princípio de máxima preservação do autor; o estilo de plotagem do autor (figsize=(16,10), grid, linewidth=5, scatter, rótulos "F1 Macro"/"Precision"/"Recall") foi mantido em todas as figuras. Não havia files.download/drive.mount/!pip/!ls/!apt/!zip no original deste notebook (checagem feita) — nada a remover.

## 05_wordcloud.ipynb

Origem: experiments/wordcloud.ipynb — nuvens de palavras por categoria (DataFrame `freq`) + distâncias de Jaccard entre os top-100 termos.

| Nº | Célula | Cat | Antes | Depois | Justificativa | Impacto |
|----|--------|-----|-------|--------|---------------|---------|
| 1 | imports (cell0) | T | `import nltk` sem downloads | `nltk.download("punkt")`, `nltk.download("punkt_tab")`, `nltk.download("stopwords")` | NLTK >= 3.8 exige `punkt_tab` para `word_tokenize`; recursos não vêm por padrão fora do Colab antigo | Nenhum — apenas garante os recursos; mesmos tokens |
| 2 | leitura CSV (cell1) | T | `pd.read_csv("../data/told-br/ToLD-BR.csv")` | `pd.read_csv(MAIN_CSV)` | Caminho relativo do layout Colab/antigo inexistente no novo local; CSV localizado na raiz via `MAIN_CSV` | Nenhum — mesmo arquivo/dados |
| 3 | laço wordclouds (cell2) | T | `plt.savefig(f"../data/{CATEGORY}.png")` e `plt.savefig("../data/toxic.png")` | `plt.savefig(FIGURES / f"wordcloud_{CATEGORY}.png")` e `plt.savefig(FIGURES / "wordcloud_toxic.png")` | Pasta `../data/` do layout antigo inexistente; artefatos centralizados em `reproduction/results/figures` | Nenhum — mesmas figuras, só muda o destino |

Preservados sem alteração: DataFrame `freq` e a lógica de tokenização/filtragem/geração das nuvens (cell2), distâncias de Jaccard sobre `freq.columns[1:]` (cell3), `print(freq[:10])` (cell4) e comprimento médio do texto `data["text"].apply(len).mean()` (cell5).

## 06_example_pretrained.ipynb (origem: model/example.ipynb)

Demo do modelo binário pré-treinado: baixa o modelo do Google Drive, descompacta e roda inferência sobre `"este é um exemplo."`. Depende de download externo (id `1Q8MuO4SsND0xzDIW9TNvzfl5Fc2NGwAJ`); se o link cair, o notebook fica não-executável sem afetar os demais.

| Nº | Célula | Cat | Antes | Depois | Justificativa | Impacto |
|----|--------|-----|-------|--------|---------------|---------|
| #1 | `[1]`+`[2]` (setup.sh + `!sh setup.sh`) | T | `%%writefile setup.sh` (clone apex, `pip install simpletransformers/unidecode/gdown`) + `!sh setup.sh` | célula de `import` (`zipfile`, `torch`, `gdown`, `transformers`) | apex e simpletransformers são tecnologia descontinuada; fp16 é nativo no PyTorch e a inferência usa `transformers` puro; instalação de pacotes não pertence ao notebook | Nenhum (dependências viram pré-requisito do ambiente) |
| #2 | `[3]` (download) | T | `from google.colab import drive, files` + `gdown.download(...)` + `os.environ['modelpath']` | `gdown.download(...)` para `RESULTS/toxic_bert_model.zip`, sem `google.colab` nem env var shell | `google.colab` não existe fora do Colab; a env var só servia ao `!unzip`; mantém-se o `gdown` e o mesmo id | Nenhum (mesmo arquivo baixado) |
| #3 | `[4]` (unzip) | T | `!unzip "$modelpath" -d .` | `zipfile.ZipFile(...).extractall(RESULTS)` | `unzip` é binário Unix ausente no Windows; `zipfile` é stdlib multiplataforma | Nenhum (mesma estrutura `toxic_bert_model/`) |
| #4 | `[5]`+`[6]` (load + predict) | T | `ClassificationModel("distilbert", "toxic_bert_model")` + `model.predict([...])` | `AutoTokenizer`/`AutoModelForSequenceClassification` + inferência `torch` (logits → `argmax`) | simpletransformers é a tecnologia descontinuada substituída por HuggingFace `transformers`; o modelo já está em formato HF, arquitetura lida do `config.json` | Mesma predição (`0`) e mesmos logits (`[[1.8342592, -1.7641641]]`) |

Observações:
- Não há mudança (A) nem (B) neste notebook: o modelo distribuído já está em formato HuggingFace, então preservá-lo como está é fiel ao artigo; a única natureza de mudança aqui é tecnologia (T). Em particular, a string `"distilbert"` do original era apenas o rótulo de arquitetura exigido pelo construtor do simpletransformers — a arquitetura real do checkpoint é lida do `config.json` pelo `AutoModel*`, então NÃO há troca de modelo nem relação com a mudança (A) distilbert→BERT-completo dos notebooks de treino.
- O modelo é salvo em `reproduction/results/toxic_bert_model/` (sob `ROOT`, via variável `RESULTS`, conforme convenção #7). Não é obrigatório salvar fora do OneDrive.

## 07_error_analysis.ipynb (NOVO — extensão; sem original do autor)

**Notebook novo.** O artigo reporta a Tabela 12 (taxa de falsos negativos por categoria fina), mas o
autor não liberou um notebook para ela — este é o único notebook da pasta que **não parte de um
original**. É autossuficiente e segue o estilo dos demais notebooks
(setup de paths padrão). Não há mudanças (T/B/A) por não haver original; abaixo a
descrição do que ele faz.

| Item | Descrição |
|------|-----------|
| Entrada | `results/preds_binary_mbert_br.csv` (predições do M-BERT-BR, geradas pelo `03_classification`) + `ToLD-BR.csv` binarizado (categorias finas) |
| Alinhamento | chave de texto normalizada `_norm` (minúsculas + remoção de acentos + remoção de U+FFFD + colapso de espaços), recupera 2100/2100 — ver `REPORT.md` §7.2 |
| Cálculo | por categoria: positivos no *test*, falsos negativos (positivo predito como não-tóxico), taxa = FN/positivos |
| Saídas | `results/error_analysis_fn_rate.json` + `results/figures/tab12_fn_rate.png` (barras artigo vs reproduzido) |
| Resultado | homophobia 7/35 (0.20), insult 74/444 (0.17), xenophobia 9/19 (0.47), misogyny 7/44 (0.16), obscene 119/699 (0.17), racism 7/17 (0.41) — mesmo padrão do artigo (minoritárias com maior FN-rate) |

Pré-requisito: rodar antes o `03_classification.ipynb`. A lógica de alinhamento e cálculo
(FN-rate por categoria) segue o descrito no `REPORT.md` §7.2.

## 08_compare_figures.ipynb (NOVO — extensão; sem original do autor)

**Notebook novo.** Gera figuras que colocam **lado a lado** os resultados do artigo e os da
reprodução, no mesmo estilo, para comparação visual direta. Não há mudanças (T/B/A) por não haver
original.

| Item | Descrição |
|------|-----------|
| Entradas | `results/binary_*.json`, `results/multilabel_mbert.json`, `results/learning_curve.json` (do `03`) + `experiments/data/learning_curve.json` (curva dos autores) |
| Fig. 2 | grade 5×2 (Artigo \| Reproduzido) das matrizes de confusão binárias, re-renderizadas dos números (mesmo estilo); baseline artigo = BoW+AutoML, repro = BoW+SVM |
| Fig. 4 | grade 6×2 (Artigo \| Reproduzido) das matrizes multi-rótulo (mBERT) |
| Fig. 3 | **Removida no kit (2026-07-15)** — usava a imagem original do artigo (`recall-precision-*class.pdf`, via `pymupdf`); dependia da pasta do paper e não entrava no artigo. A curva comparada é a **Fig. 3b** abaixo. |
| Fig. 3b | **sobreposição das curvas reais**: artigo (de `experiments/data/learning_curve.json`, média das 3 reps) × reprodução, eixo-x em **% dos dados de treino** (neutraliza a diferença do ponto final: no artigo, só o ponto de 100% usa 21.000 — os pontos 10–90% já são frações de 18.900) |
| Barras | macro-F1 binário Artigo × Reproduzido |
| Saídas (`results/figures/comparison/`) | `compare_fig2_binary_cm.png`, `compare_fig4_multilabel_cm.png`, `compare_fig3b_overlay_learning_curve.png`, `compare_macro_f1_bars.png` |
| Dependência | nenhuma além de `requirements.txt` (o `pymupdf` foi removido junto com a Fig. 3) |

Os números do artigo (matrizes das Figuras 2/4 e macro-F1 das Tabelas 7–11) entram **hardcoded** como
referência, com a mesma proveniência das colunas "Artigo" do `REPORT.md`. Pré-requisito: rodar antes
o `03_classification.ipynb`.
