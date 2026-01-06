# Trabalho Final (IA) — (Adult + Bank Marketing + Credit Card)
# Alunos: Gustavo Francisco S. S. França, Mauricio Cesar Cetrangolo, Everton Ferreira de Lima

Este repositório contém **todo o código-fonte** e instruções para **reproduzir** os experimentos do Trabalho Final, incluindo:
- Treinamento de **baseline** (Regressão Logística);
- Treinamento de **modelo mitigado** (Adversarial Debiasing via Fairlearn);
- Cálculo de métricas de **desempenho** e **fairness**;
- **Explicabilidade** com SHAP e exportação de planilhas `.csv`.


## 1) Estrutura do repositório

Sugestão de organização (padrão exigido pelo enunciado):

├─ notebooks/
│  └─ Trabalho_Final_Fairness_SHAP.ipynb
├─ src/
│  └─ run_all_datasets.py
├─ outputs/
│  ├─ adult_shap_importance.csv
│  ├─ bank_shap_importance.csv
│  └─ credit_shap_importance.csv
└─ README.md
```

- `notebooks/`: notebook principal para execução no Google Colab (recomendado).
- `src/`: versão em script (Python) com o mesmo pipeline.
- `outputs/`: arquivos gerados (CSV SHAP).  
  **Nota:** você pode versionar os CSVs no repositório (para conferência), mas o principal é que eles possam ser **gerados novamente** ao executar o notebook/script.


## 2) Bases públicas utilizadas (links + forma de download)

Este trabalho utiliza **3 bases públicas**. A reprodução pode ocorrer de duas formas:

### (A) Forma recomendada (automática e reprodutível): loaders do Fairlearn
O código baixa/carrega os datasets via loaders do Fairlearn:
- `fetch_adult`
- `fetch_bank_marketing`
- `fetch_credit_card`

✅ Vantagens:
- evita download manual e erros de versão/arquivo;
- execução reprodutível (basta rodar o notebook).

### (B) Links oficiais (UCI) — referência direta
As bases correspondem às versões públicas clássicas do repositório UCI:

- **Adult (Census Income)** — UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/adult

- **Bank Marketing** — UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/bank+marketing

- **Default of Credit Card Clients** — UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

> **Observação importante:** ao usar os loaders do Fairlearn, algumas execuções podem retornar colunas **anonimizadas** (ex.: `V1..V16` ou `x1..x23`).  
> Para manter o experimento simples e reprodutível, nesses casos o atributo sensível é definido por **proxy numérico binarizado pela mediana** (detalhado na Seção 5).

---

## 3) Requisitos (ambiente)

- Python 3.10+ (Google Colab funciona)
- Bibliotecas:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `fairlearn`
  - `tensorflow`
  - `shap`

Instalação:

```bash
pip install -U numpy pandas scikit-learn fairlearn tensorflow shap
```

## 4) Como executar (reprodução no Colab)

### 4.1 Abrir o notebook
Abra `notebooks/Trabalho_Final_Fairness_SHAP.ipynb` no Colab e execute na ordem.

### 4.2 Instalar dependências
Primeira célula:

```bash
!pip -q install -U numpy pandas scikit-learn fairlearn tensorflow shap
```

### 4.3 Executar o pipeline
Execute as células seguintes. Ao final, o notebook imprime:
- métricas de desempenho (Accuracy / F1 / AUC);
- métricas de fairness (DPD / DI / EOD);
- e gera os CSVs SHAP em `outputs/` (ou no diretório de trabalho do Colab).

### 4.4 Baixar os resultados
No Colab:

```python
from google.colab import files
files.download("adult_shap_importance.csv")
files.download("bank_shap_importance.csv")
files.download("credit_shap_importance.csv")
```

## 5) Configuração dos datasets e atributos sensíveis (3 bases)

### Dataset 1 — Adult (Census Income)
- **Target (Y):** renda > 50K (classe positiva)
- **Atributo sensível (A):** `sex` (Male/Female)

### Dataset 2 — Bank Marketing
- **Target (Y):** subscrição do produto (classe positiva)
- **Atributo sensível (A):**
  - Se vier com colunas semânticas (ex.: `age`): pode-se usar `age_group`
  - **Se vier anonimizadas (`V1..V16`):** usa-se `V1_bin(mediana)` (**proxy adotado**)

### Dataset 3 — Credit Card Default
- **Target (Y):** default (classe positiva)
- **Atributo sensível (A):**
  - Se vier com colunas semânticas (ex.: `SEX`): usar `SEX`
  - **Se vier anonimizadas (`x1..x23`):** usa-se `x2_bin(mediana)` (**proxy adotado**)

> **Justificativa do proxy:** em versões anonimizadas não existe atributo semântico explícito; a binarização por mediana cria **dois grupos comparáveis** e permite calcular DPD/DI/EOD conforme exigido.

---

## 6) Modelos e métricas

### 6.1 Modelos
- **Baseline:** Regressão Logística (`LogisticRegression`)
- **Mitigação:** `AdversarialFairnessClassifier` (Fairlearn) com `constraints="demographic_parity"`

### 6.2 Métricas
- **Desempenho:** Accuracy, F1-score, AUC-ROC
- **Fairness:**  
  - DPD — Demographic Parity Difference (ideal ~ 0)  
  - DI — Demographic Parity Ratio / Disparate Impact (ideal ~ 1)  
  - EOD — Equal Opportunity Difference (ideal ~ 0)

### 6.3 SHAP (explicabilidade)
- Baseline: `LinearExplainer`
- Mitigado: `KernelExplainer` com amostragem  
CSV exportado:
- `feature`
- `mean_abs_shap_baseline`
- `mean_abs_shap_adversarial`
- `delta`

---

## 7) Saídas do experimento

Arquivos gerados:
- `adult_shap_importance.csv`
- `bank_shap_importance.csv`
- `credit_shap_importance.csv`

Além disso, o notebook imprime um resumo no console (útil para preencher o relatório):

```
=== RESUMO (para preencher tabelas) ===
--- Adult ---
BASELINE perf: {accuracy, f1, auc}
BASELINE fair: {DPD, DI, EOD}
ADV perf: {accuracy, f1, auc}
ADV fair: {DPD, DI, EOD}
...
```

---

## 8) Reprodutibilidade (boas práticas)
- `random_state=42` para divisão treino/teste;
- pipeline padronizado nos 3 datasets;
- observação técnica: o adversarial (TensorFlow) exige **matriz densa**, por isso há conversão `sparse -> dense`.

---

## 9) Problemas comuns (Colab)
- **Erro de sparse/dense:** converter com `.toarray()` antes do adversarial.
- **Target com 1 classe:** binarização robusta do target (fallback para classe menos frequente).
- **Colunas anonimizadas:** usar proxies `V1` e `x2` binarizados por mediana.

---
