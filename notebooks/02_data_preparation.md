 # Preparação de Dados

 ## Objetivo
 A preparação dos dados envolve a análise exploratória e o pré-processamento dos dados, garantindo que estejam prontos para serem usados na construção e treinamento de modelos de machine learning. Neste caso, utilizamos um conjunto de dados de músicas.

## Bibliotecas Utilizadas
- `pandas`: Para manipulação e análise de dados.
- `numpy`: Para operações numéricas.
- `matplotlib` e `seaborn`: Para visualização dos dados.

## Passos Realizados

### 1. Configurações Iniciais
 Configuramos o estilo de visualização do Seaborn para melhorar a aparência dos gráficos:
 ```python
 import seaborn as sns
 sns.set(style="whitegrid")
 ```

 ### 2. Carregamento dos Dados
 Carregamos os dados brutos de um arquivo JSON para um DataFrame do Pandas:
 ```python
 import pandas as pd

raw_data_path = '../data/raw/songs.json'
df = pd.read_json(raw_data_path)
 df.head()  # Exibir as primeiras linhas do DataFrame
 ```

 ### 3. Análise Exploratória dos Dados
 Exibimos informações básicas e estatísticas descritivas dos dados:
 ```python
 df.info()  # Informações básicas
 df.describe(include='all')  # Estatísticas descritivas
 ```

 ### 4. Visualizações
 Criamos diversas visualizações para entender a distribuição dos dados:
 - **Distribuição das Músicas por Ano**:
 ```python
 import matplotlib.pyplot as plt
 import seaborn as sns

 plt.figure(figsize=(10, 6))
 sns.histplot(df['year'], bins=30, kde=True)
 plt.title('Distribuição das Músicas por Ano')
 plt.xlabel('Ano')
 plt.ylabel('Contagem')
 plt.show()
 ```
 - **Contagem de Músicas por Gênero**:
 ```python
 plt.figure(figsize=(12, 8))
 sns.countplot(y=df['genre'], order=df['genre'].value_counts().index)
 plt.title('Contagem de Músicas por Gênero')
 plt.xlabel('Contagem')
 plt.ylabel('Gênero')
 plt.show()
 ```
 - **Mapa de Calor de Correlação**:
 ```python
 numeric_df = df.select_dtypes(include=[np.number])
 plt.figure(figsize=(10, 8))
 sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
 plt.title('Mapa de Calor de Correlação')
 plt.show()
 ```

 ### 5. Tratamento de Dados Ausentes
 Verificamos e tratamos os valores ausentes preenchendo-os com a mediana:
 ```python
 missing_values = df.isnull().sum()
 print("Valores ausentes por coluna:\n", missing_values)
 df['year'].fillna(df['year'].median(), inplace=True)
 ```

 ### 6. Tratamento de Outliers
Identificamos e tratamos outliers usando o método IQR:
 ```python
 Q1 = df['year'].quantile(0.25)
 Q3 = df['year'].quantile(0.75)
 IQR = Q3 - Q1
 lower_bound = Q1 - 1.5 * IQR
 upper_bound = Q3 + 1.5 * IQR
 outliers = df[(df['year'] < lower_bound) | (df['year'] > upper_bound)]
 print("Outliers identificados:\n", outliers)
 df = df[(df['year'] >= lower_bound) & (df['year'] <= upper_bound)]
```

 ### 7. Conversão de Variáveis Categóricas
 Convertimos a coluna `genre` em variáveis dummy:
 ```python
 df = pd.get_dummies(df, columns=['genre'])
 ```

 ### 8. Salvando os Dados Processados
 Salvamos os dados processados em um arquivo CSV:
 ```python
 processed_data_path = '../data/processed/songs_processed.csv'
 df.to_csv(processed_data_path, index=False)
 print(f"Dados processados salvos em: {processed_data_path}")
 ```
