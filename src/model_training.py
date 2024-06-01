import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    # Carregar os conjuntos de dados
    train_data = pd.read_csv('../data/splits/train_data.csv')
    val_data = pd.read_csv('../data/splits/val_data.csv')
    test_data = pd.read_csv('../data/splits/test_data.csv')

    # Definir a coluna de destino (target)
    target_column = 'popularity'  # Substitua pelo nome correto se necessário

    # Separar recursos e rótulos
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]

    # Identificar colunas categóricas e numéricas
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    numerical_columns = X_train.select_dtypes(exclude=['object']).columns

    # Criar transformadores para colunas categóricas e numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])

    # Inicializar modelos
    models = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier()
    }

    # Treinar e avaliar modelos usando um pipeline
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'{model_name} Validation Accuracy: {accuracy:.4f}')
        print(classification_report(y_val, y_pred))

    # Exemplo de ajuste de hiperparâmetros para o RandomForestClassifier
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier())]), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_val)
    print(f'Best RandomForest Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}')
    print(classification_report(y_val, y_pred))

    # Exemplo: Aplicar regularização ao Logistic Regression
    log_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression(C=0.1, max_iter=1000))])
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_val)
    print(f'Logistic Regression with Regularization Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}')
    print(classification_report(y_val, y_pred))

    # Salvar o modelo treinado
    joblib.dump(best_rf_model, '../models/best_random_forest.pkl')

if __name__ == "__main__":
    main()
