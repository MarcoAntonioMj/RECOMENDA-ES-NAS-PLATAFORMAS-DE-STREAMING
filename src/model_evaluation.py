import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
n_classes_y_test = len(unique_classes_test)
n_classes_y_proba = y_test_proba.shape[1]

# Criar um dicionário para mapear as classes presentes para os índices de probabilidade
class_to_index = {class_label: i for i, class_label in enumerate(np.unique(y_train))}

# Remapear previsões de probabilidade
y_test_proba_remapped = np.zeros((y_test_proba.shape[0], n_classes_y_test))
for i, class_label in enumerate(unique_classes_test):
    if class_label in class_to_index:
        y_test_proba_remapped[:, i] = y_test_proba[:, class_to_index[class_label]]
    else:
        # Se a classe não está nas previsões, preencher com zero
        y_test_proba_remapped[:, i] = 0

# Normalizar as probabilidades para garantir que somam 1
y_test_proba_remapped = y_test_proba_remapped / y_test_proba_remapped.sum(axis=1, keepdims=True)

# Avaliar o modelo
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred, zero_division=0)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba_remapped, multi_class='ovr', average='macro')

# Exibir resultados
print(f'Test Accuracy: {test_accuracy:.4f}')
print('Classification Report:')
print(test_classification_report)
print('Confusion Matrix:')
print(test_confusion_matrix)
print(f'ROC AUC: {test_roc_auc:.4f}')

# Função para plotar matriz de confusão
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plotar matriz de confusão
plot_confusion_matrix(test_confusion_matrix, classes=unique_classes_test)

# Função para plotar curvas ROC
def plot_roc_curve(y_true, y_proba, n_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plotar curva ROC para cada classe
plot_roc_curve(y_test, y_test_proba_remapped, n_classes=n_classes_y_test)
