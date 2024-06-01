import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    return pd.read_json(file_path)

def data_info(df):
    print(df.head())  # Exibir as primeiras linhas do DataFrame
    print(df.info())  # Exibir informações sobre o DataFrame, como tipos de dados e valores nulos
    print(df.describe(include='all'))  # Exibir estatísticas descritivas

def plot_distribution(df):
    plt.figure(figsize=(10, 6))  # Configurar o tamanho da figura
    sns.histplot(df['year'], bins=30, kde=True)  # Criar histograma com a distribuição dos anos
    plt.title('Distribuição das Músicas por Ano')  # Definir título do gráfico
    plt.xlabel('Ano')  # Definir rótulo do eixo x
    plt.ylabel('Contagem')  # Definir rótulo do eixo y
    plt.show()  # Mostrar o gráfico

def plot_genre_count(df):
    plt.figure(figsize=(12, 8))  # Configurar o tamanho da figura
    sns.countplot(y=df['genre'], order=df['genre'].value_counts().index)  # Criar gráfico de contagem
    plt.title('Contagem de Músicas por Gênero')  # Definir título do gráfico
    plt.xlabel('Contagem')  # Definir rótulo do eixo x
    plt.ylabel('Gênero')  # Definir rótulo do eixo y
    plt.show()  # Mostrar o gráfico

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])  # Selecionar apenas colunas numéricas
    plt.figure(figsize=(10, 8))  # Configurar o tamanho da figura
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # Criar mapa de calor
    plt.title('Mapa de Calor de Correlação')  # Definir título do gráfico
    plt.show()  # Mostrar o gráfico

def handle_missing_values(df):
    missing_values = df.isnull().sum()  # Contar valores ausentes por coluna
    print("Valores ausentes por coluna:\n", missing_values)
    df['year'].fillna(df['year'].median(), inplace=True)  # Preencher valores ausentes na coluna 'year' com a mediana

def handle_outliers(df):
    Q1 = df['year'].quantile(0.25)  # Calcular o primeiro quartil (Q1)
    Q3 = df['year'].quantile(0.75)  # Calcular o terceiro quartil (Q3)
    IQR = Q3 - Q1  # Calcular o intervalo interquartil (IQR)
    lower_bound = Q1 - 1.5 * IQR  # Definir limite inferior para outliers
    upper_bound = Q3 + 1.5 * IQR  # Definir limite superior para outliers
    outliers = df[(df['year'] < lower_bound) | (df['year'] > upper_bound)]  # Identificar outliers
    print("Outliers identificados:\n", outliers)
    return df[(df['year'] >= lower_bound) & (df['year'] <= upper_bound)]  # Remover outliers do DataFrame

def convert_categorical(df):
    return pd.get_dummies(df, columns=['genre'])  # Converter coluna 'genre' em variáveis dummy

def save_data(df, file_path):
    df.to_csv(file_path, index=False)  # Salvar DataFrame em um arquivo CSV sem incluir o índice
    print(f"Dados processados salvos em: {file_path}")

if __name__ == "__main__":
    # Caminho absoluto para o arquivo de dados brutos
    raw_data_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/raw/songs.json'
    # Caminho absoluto para salvar os dados processados
    processed_data_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/processed/songs_processed.csv'
    
    df = load_data(raw_data_path)  # Carregar dados
    data_info(df)  # Exibir informações básicas sobre os dados
    plot_distribution(df)  # Plotar distribuição das músicas por ano
    plot_genre_count(df)  # Plotar contagem de músicas por gênero
    plot_correlation(df)  # Plotar mapa de calor das correlações
    handle_missing_values(df)  # Tratar valores ausentes
    df = handle_outliers(df)  # Tratar outliers
    df = convert_categorical(df)  # Converter colunas categóricas em numéricas
    save_data(df, processed_data_path)  # Salvar os dados processados
