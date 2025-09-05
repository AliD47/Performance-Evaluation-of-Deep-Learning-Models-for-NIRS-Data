# Importe les librairies nécessaires
import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import KFold
import optuna 
import warnings
warnings.filterwarnings('ignore')
# Pour la reproductibilité
tf.random.set_seed(47)
np.random.seed(47)

# Vérifie si un GPU est disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU disponible:", gpus)
else:
    print("Aucun GPU détecté.")

        ### == Fonctions utilitaires == ###    
# Fonction pour séparer les données en calibration, validation et test
def split_data(Var):
    # Vérifie que la variable existe bien dans les fichiers Y et M
    if Var not in Y.columns or Var not in M.columns:
        raise ValueError(f"Erreur : la variable {Var} n'a pas été trouvée.")
    
    # Crée un masque basé sur la colonne de la variable dans M
    mask = M[Var]
    
    # Sépare X et Y en utilisant le masque
    X_cal = X[mask == 'cal']
    Y_cal = Y.loc[X_cal.index, Var]
    
    X_val = X[mask == 'val']
    Y_val = Y.loc[X_val.index, Var]
    
    X_test = X[mask == 'test']
    Y_test = Y.loc[X_test.index, Var]
    
    # Retourne les ensembles de données séparés
    return (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test)

def build_model(trial):
    # Paramètres fixes du modèle
    filter_size = 700
    padding_type = 'valid'
    
    # les  hyperparamètres à tester
    num_filters = trial.suggest_int("num_filters", 1, 32)
    num_dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
    dense_units = [trial.suggest_int(f"dense_{i}_units", 8, 128, step=4) for i in range(num_dense_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.005) 
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.03, log=True)
    l2_reg = trial.suggest_float("l2_reg", 0.0, 0.1, step=5e-4) 
    batch_size = trial.suggest_int("batch_size", 32, 256, step=16)
    
    # Initialisation des poids du réseau
    initializer = tf.keras.initializers.he_normal(seed=123)  # seed pour la reproductibilité
    
    ### == Définition de l'architecture du modèle == ###
    inputs = tf.keras.Input(shape=(700,)) # Entrée avec 700 variables
    x = tf.keras.layers.Reshape((700, 1))(inputs) # Reshape pour la couche Conv1D
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=filter_size,
                               padding=padding_type,
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Activation('elu')(x) # Fonction d'activation ELU
    x = tf.keras.layers.Flatten()(x) # Aplatissement pour passer à la couche dense
    
    # Ajoute les couches denses (s'il y en a)
    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(units, kernel_initializer=initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = tf.keras.layers.Activation('elu')(x)
        if num_dense_layers > 1:
            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    # Couche de sortie
    output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer)(x) 
    model = tf.keras.Model(inputs=inputs, outputs=output) # Modèle final
    
    # Compilation du modèle (définition de l'optimiseur et fonction de perte)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]) 
    return model, batch_size

def objective(trial, X_train_data, Y_train_data):
    # Callbacks pour l'entraînement (pour éviter le surapprentissage)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=5e-2, patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=10)
    
    # Prépare la validation croisée (5 plis)
    kf = KFold(n_splits=5, shuffle=True, random_state=47)  
    rmse_scores = []
    epochs_list = []
    history_dicts = []

    # Boucle pour chaque pli de la validation croisée
    for train_idx, val_idx in kf.split(X_train_data):  
        X_train_cv, X_val_cv = X_train_data[train_idx], X_train_data[val_idx]
        Y_train_cv, Y_val_cv = Y_train_data.iloc[train_idx], Y_train_data.iloc[val_idx]
        # Construit un nouveau modèle pour chaque pli
        model, batch_size = build_model(trial)
        # Entraîne le modèle
        history = model.fit(X_train_cv, Y_train_cv,
                            validation_data=(X_val_cv, Y_val_cv),
                            epochs=150,
                            batch_size=batch_size,
                            callbacks=[es_callback, reduce_lr],
                            verbose=0)
        
        epochs_list.append(len(history.history['loss']))
        history_dicts.append(history.history)
        val_loss, val_rmse = model.evaluate(X_val_cv, Y_val_cv, verbose=0)
        rmse_scores.append(val_rmse)
    
    avg_epochs = int(np.mean(epochs_list))
    trial.set_user_attr("avg_epochs", avg_epochs)
    trial.set_user_attr("epochs_list", epochs_list)
    trial.set_user_attr("fold_histories", history_dicts)

    return np.mean(rmse_scores)

# Fonction pour construire le modèle final avec les meilleurs paramètres
def build_best_model(params):
    filter_size = 700
    padding_type = 'valid'
    num_filters = params['num_filters']
    initializer = tf.keras.initializers.he_normal(seed=123)
    
    inputs = tf.keras.Input(shape=(700,))
    x = tf.keras.layers.Reshape((700, 1))(inputs)
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=filter_size,
                               padding=padding_type,
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(params['num_dense_layers']):
        units = params[f'dense_{i}_units']
        x = tf.keras.layers.Dense(units, kernel_initializer=initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(x)
        x = tf.keras.layers.Activation('elu')(x)
        if params['num_dense_layers'] > 1:
            x = tf.keras.layers.Dropout(rate=params['dropout_rate'])(x)
    output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # Compile le modèle final
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model



# charge les fichiers CSV en DataFrames (X = prédicteurs, Y = réponses, M = partitions)
X = pd.read_csv(";;; X.csv", sep=';')
Y = pd.read_csv(";;; Y.csv", sep=';')
M = pd.read_csv(";;; M.csv", sep=';', na_values ='missing')

# Nom du modèle utilisé
Modd = "CNN-R_v1E"  # on peut changer cela au CNN-R_v1D pour l'autre version

# Boucle sur chaque variable cible dans le fichier Y
for Var in Y.columns:
    # Sépare les données pour la variable actuelle
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(Var)
    
    # Combine les données de calibration et validation pour l'entraînement
    Y_train = pd.concat([Y_cal, Y_val])
    X_train = pd.concat([X_cal, X_val])

    # Convertit les dataframes en 'array' numpy pour TensorFlow 
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Nombre de variables spectrales
    num_features = 700

    # Calcule la moyenne et l'écart-type sur les données d'entraînement
    mean_train, std_train = X_train.mean(axis=0), X_train.std(axis=0)
    
    # Normalise (standardise) les données d'entraînement et de test
    X_train_N = (X_train - mean_train) / std_train
    X_test_N = (X_test - mean_train) / std_train

    # Ajoute une dimension (pour la convolution) pour que les données soient compatibles avec le modèle CNN
    X_train_f = X_train_N[..., np.newaxis]
    X_test_f = X_test_N[..., np.newaxis]

    # --- Lancement de l'étude Optuna --- #
    print("\nLancement de l'optimisation pour la variable :", Var)
    ii = datetime.now()
    iii = str(ii.strftime("%Y-%m-%d_%Hh%M"))
    print(" ********************************** Début:", iii, "***********************************\n")
    
    # Crée ou charge une étude Optuna
    study = optuna.create_study(
        direction="minimize",
        study_name=f"/optuna/cnn_hpo_{Var}",
        load_if_exists=True
    )
    # Lance l'optimisation pour 'n_trials' essais
    study.optimize(lambda trial: objective(trial, X_train_f, Y_train), n_trials = 300) # prend jusqu'à ~ 13h avec un RTX 2060

    # Affiche les meilleurs résultats trouvés
    print("Meilleur essai :")
    print(f"  RMSE sur la validation : {study.best_value:.4f}")
    print("  Meilleurs paramètres :")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # --- Entraînement du modèle final ---

    # Récupère les meilleurs paramètres
    best_params = study.best_trial.params
    
    # Construit et entraîne le modèle final sur toutes les données d'entraînement
    final_model = build_best_model(best_params)
    final_model.fit(X_train_f, Y_train,
                    epochs=study.best_trial.user_attrs["avg_epochs"], # Utilise le nombre moyen d'époques
                    batch_size=best_params['batch_size'],
                    verbose=0)
    
    # Évalue le modèle final sur les données de test (jamais vues)
    test_loss, test_rmse = final_model.evaluate(X_test_f, Y_test, verbose=1)
    print(f"RMSE sur l'ensemble de test : {test_rmse:.4f}")

    # --- Calcul des métriques et sauvegarde des résultats ---
    # Chemin vers le fichier de résultats
    csv_path = ";;; /resultats_CNN.csv"

    # Calcule les métriques de performance
    rmse = test_rmse
    rpd = np.std(Y_test) / rmse
    relative_error = rmse / np.mean(Y_test)
    
    # Prépare une nouvelle ligne pour le fichier CSV
    new_row = pd.DataFrame({
        "Modèle": [Modd],
        "Variable": [Var],
        "RMSE": [rmse],
        "RE": [relative_error],
        "RPD": [rpd]
    })
    
    # Si le fichier existe, charge-le et ajoute la nouvelle ligne
    if os.path.exists(csv_path):
        existing_results = pd.read_csv(csv_path)
        updated_results = pd.concat([existing_results, new_row], ignore_index=True)
    else:
        # Sinon, crée un nouveau dataframe avec la nouvelle ligne
        updated_results = new_row
    
    # Sauvegarde les résultats dans le fichier CSV
    updated_results.to_csv(csv_path, index=False, sep=',')