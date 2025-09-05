# Importe les librairies nécessaires
import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import optuna 
import warnings
warnings.filterwarnings('ignore')

# Pour la reproductibilité
Seed = 47
tf.random.set_seed(Seed)
np.random.seed(Seed)

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

def custom_loss(lambda_l2, model):
    def loss_fn(y_true, y_pred):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred)) # Calcule l'erreur absolue moyenne
        
        # Ajoute la pénalité (régul L2) pour éviter le surapprentissage (seulement pour els poids des couches denses)
        l2 = tf.add_n([
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'kernel' in v.name
        ])
        # L'erreur totale = la somme de la MAE et de la pénalité L2
        return mae + lambda_l2 * l2
    return loss_fn

def objective(trial, X_train_data, Y_train_data):
    # 1. les hyperparamètres à optimiser
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    l2 = trial.suggest_loguniform("l2", 1e-5, 1e-2)

    # 2. taux d'apprentissage exponentiellement décroissant (pour le surapprentissage)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=1e-3
    )

    # 3. Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=Seed)
    rmse_scores = []
    epochs_list = []
    history_dicts = []

    # 4. Boucle pour chaque pli de la validation croisée
    for train_idx, val_idx in kf.split(X_train_data):  
        X_train_cv, X_val_cv = X_train_data[train_idx], X_train_data[val_idx]
        Y_train_cv, Y_val_cv = Y_train_data.iloc[train_idx], Y_train_data.iloc[val_idx]
    
        # Construit un nouveau modèle pour chaque pli
        ipa_model = IPA(seed_value=Seed, regularization_factor=l2)

        ipa_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=custom_loss(l2, ipa_model),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        # 5. Callbacks
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error", min_delta=5e-2, patience=16, restore_best_weights=True)
        
        # 6. Entrainer
        history = ipa_model.fit(
            X_train_cv, Y_train_cv,
            validation_data=(X_val_cv, Y_val_cv),
            epochs=150,
            batch_size=16,
            callbacks=[es_callback],
            verbose=0
        )

        # 7. Log metrics
        val_loss, val_rmse = ipa_model.evaluate(X_val_cv, Y_val_cv, verbose=0)
        rmse_scores.append(val_rmse)
        epochs_list.append(len(history.history["loss"]))
        history_dicts.append(history.history)

    # 8. Enregistrer les résultats dans l'objet trial
    avg_epochs = int(np.mean(epochs_list))
    trial.set_user_attr("avg_epochs", avg_epochs)
    trial.set_user_attr("epochs_list", epochs_list)
    trial.set_user_attr("fold_histories", history_dicts)

    return np.mean(rmse_scores)

### == Chargement des données et optimisation du modèle == ###

# charge les fichiers CSV en DataFrames (X = prédicteurs, Y = réponses, M = partitions)
X = pd.read_csv(";;; X.csv", sep=';')
Y = pd.read_csv(";;; Y.csv", sep=';')
M = pd.read_csv(";;; M.csv", sep=';', na_values ='missing')

# Importer le modèle IPA (fichier: IPA_architecture.py)
from Modèle_IPA_architeture import IPA
Modd = "IPA"

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
        study_name=f"/optuna/ipa_hpo_{Var}",
        load_if_exists=True
    )
    # Lance l'optimisation pour 'n_trials' essais
    study.optimize(lambda trial: objective(trial, X_train_f, Y_train), n_trials = 100) # prend ~ 12h avec un RTX 2060

    # Affiche les meilleurs résultats trouvés
    print("Meilleur essai :")
    print(f"  RMSE sur la validation : {study.best_value:.4f}")
    print("  Meilleurs paramètres :")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # --- Entraînement du modèle final ---

    # Récupère les meilleurs paramètres
    best_params = study.best_trial.params
    
    final_model = IPA(seed_value=Seed, regularization_factor=best_params['best_l2'])

    lr_schedule = ExponentialDecay(
        initial_learning_rate=best_params['best_lr'],
        decay_steps=10000,
        decay_rate=1e-3
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=5,
        restore_best_weights=True
    )
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=custom_loss(best_params['best_l2'], final_model),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    history = final_model.fit(
        X_train_f, Y_train,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_rmse = final_model.evaluate(X_test_f, Y_test, verbose=1)
    print(f"RMSE on test set: {test_rmse:.4f}")

    # --- Calcul des métriques et sauvegarde des résultats ---
    # Chemin vers le fichier de résultats
    csv_path = ";;; /resultats_IAA.csv"

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
