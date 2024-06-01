import pandas as pd
from sklearn.model_selection import train_test_split

def load_processed_data(file_path):
    return pd.read_csv(file_path)

def split_data(df, test_size=0.2, validation_size=0.25, random_state=42):
    # Dividir em conjunto de treinamento + validação e conjunto de teste
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Dividir conjunto de treinamento + validação em conjunto de treinamento e conjunto de validação
    train_df, val_df = train_test_split(train_val_df, test_size=validation_size, random_state=random_state)
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, base_path):
    train_df.to_csv(f'{base_path}/train.csv', index=False)
    val_df.to_csv(f'{base_path}/val.csv', index=False)
    test_df.to_csv(f'{base_path}/test.csv', index=False)
    print(f"Dados divididos e salvos em: {base_path}")

if __name__ == "__main__":
    processed_data_path = 'C:/Users/marco/OneDrive/Área de Trabalho/A3-Dados/RECOMENDA-ES-NAS-PLATAFORMAS-DE-STREAMING/data/processed/songs_processed.csv'
    splits_path = '../data/splits'
    df = load_processed_data(processed_data_path)
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df, splits_path)
