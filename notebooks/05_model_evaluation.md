# Avaliação de Modelo de Classificação Multiclasse

Este documento descreve o processo para avaliar um modelo de classificação multiclasse utilizando dados de teste. As métricas de avaliação incluem acurácia, relatório de classificação, matriz de confusão e AUC ROC.

## Passos

1. **Carregar Dados e Modelo**
   - Os dados de treino, validação e teste são carregados a partir de arquivos CSV.
   - O modelo treinado é carregado a partir de um arquivo `.pkl`.

2. **Separar Recursos e Rótulos**
   - As colunas de recursos e a coluna alvo (`popularity`) são separadas.

3. **Fazer Previsões**
   - As previsões e probabilidades de previsão são obtidas para os dados de teste.

4. **Remapear Previsões de Probabilidade**
   - As probabilidades são remapeadas para garantir que cada linha nas probabilidades remapeadas some 1.

5. **Avaliar o Modelo**
   - Acurácia, relatório de classificação, matriz de confusão e AUC ROC são calculados e exibidos.

6. **Plotar Resultados**
   - A matriz de confusão e a curva ROC para cada classe são plotadas.

## Código Python

O código completo está disponível no arquivo `evaluation.py`. Certifique-se de atualizar os caminhos dos arquivos de dados e modelo conforme necessário.

## Considerações
Probabilidades Normalizadas: As probabilidades são normalizadas após o remapeamento para garantir que somem 1.

Métricas Indefinidas: Para lidar com classes que não têm amostras previstas ou verdadeiras, usamos o parâmetro zero_division=0 nas funções de métrica.

Certifique-se de ter as bibliotecas necessárias instaladas:

 - pip install pandas numpy scikit-learn joblib matplotlib seaborn





```
# Conteúdo do arquivo evaluation.py
evaluation_py_content = 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
```

# Caminhos dos arquivos
data_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/splits/'
model_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/models/best_random_forest.pkl'

# Carregar dados
train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
val_data = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

# Definir a coluna de destino (target)
target_column = 'popularity'  # Substitua pelo nome correto se necessário

# Separar recursos e rótulos
X_train = train_data.drop(target_column, axis=1)
y_train = train_data[target_column]
X_val = val_data.drop(target_column, axis=1)
y_val = val_data[target_column]
X_test = test_data.drop(target_column, axis=1)
y_test = test_data[target_column]

# Carregar o modelo treinado
best_rf_model = joblib.load(model_path)

# Fazer previsões
y_test_pred = best_rf_model.predict(X_test)
y_test_proba = best_rf_model.predict_proba(X_test)

# Verificar o número de classes em y_test e y_test_proba
unique_classes_test = np.unique(y_test)
n_classes_y_test = len(unique