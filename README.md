# Trabalho Final (IA) — (Adult + Bank Marketing + Credit Card)
# Alunos: Gustavo Francisco S. S. França, Mauricio Cesar Cetrangolo, Everton Ferreira de Lima


3 datasets públicos via **Fairlearn** (download automático):
1) Adult — alvo: renda >50K; sensível: `sex`
2) Bank Marketing — alvo: `y` (yes/no); sensível: `age_group_(>=40)` (derivado de `age`)
3) Credit Card Default — alvo: default (0/1); sensível: `SEX`

Modelos:
- Baseline: Regressão Logística
- Mitigação: AdversarialFairnessClassifier (`constraints="demographic_parity"`)

Métricas:
- AUC, F1, Accuracy
- DPD, DI, EOD

SHAP:
- Baseline: LinearExplainer (rápido)
- Debiased: KernelExplainer (amostrado)

## Instalar
```bash
pip install -U pandas scikit-learn fairlearn shap tensorflow
```

## Rodar tudo
```bash
python run_all.py
```

## Saídas
- Console: métricas baseline vs debiased por dataset
- CSVs:
  - adult_shap_importance.csv
  - bank_shap_importance.csv
  - credit_shap_importance.csv
