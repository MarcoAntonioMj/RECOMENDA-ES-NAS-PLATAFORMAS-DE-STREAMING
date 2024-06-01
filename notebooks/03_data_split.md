# 03_data_split.ipynb

## Etapa 4: Divisão de Dados

Neste notebook, vamos dividir os dados processados em três conjuntos: treinamento, validação e teste. Cada conjunto tem um propósito específico:

- **Conjunto de Treinamento**: Usado para treinar o modelo.
- **Conjunto de Validação**: Usado para ajustar hiperparâmetros e evitar overfitting.
- **Conjunto de Teste**: Usado para avaliar a performance final do modelo.

```python
.import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar os dados
.data_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/processed/songs_processed.csv'
.df = pd.read_csv(data_path)

# Dividir os dados em treinamento, validação e teste
.train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
.train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Salvar os dados divididos
.train_data.to_csv('C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/splits/train_data.csv', index=False)
.val_data.to_csv('C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/splits/val_data.csv', index=False)
.test_data.to_csv('C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/splits/test_data.csv', index=False)

# Imprimir informações sobre os conjuntos de dados
.print("Tamanho do conjunto de treinamento:", len(train_data))
.print("Tamanho do conjunto de validação:", len(val_data))
.print("Tamanho do conjunto de teste:", len(test_data))
