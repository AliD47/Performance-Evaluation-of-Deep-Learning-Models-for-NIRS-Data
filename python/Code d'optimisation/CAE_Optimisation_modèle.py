# Importe les librairies nécessaires
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape
from sklearn.model_selection import KFold
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Input
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

def build_cae_model(num_features, latent_dim, kernel_size=5):
    """
        num_features (tuple): La forme des données d'entrée, par exemple (700, 1).
        latent_dim (int): Le nombre de variables dans la 'bottleneck'.
        kernel_size (int): La taille des noyaux de convolution.
    """
    # --- Encodeur (comprime les données en varaibles latente) ---
    encoder_input = Input(shape=num_features, name="encoder_input")

    # Première couche de convolution et de pooling
    x = Conv1D(16, kernel_size, activation='tanh', padding='same')(encoder_input)
    x = MaxPooling1D(2, padding='same')(x) # Réduit la taille des données de moitié (max de chaque 2 valeurs)

    # Deuxième couche de convolution + pooling
    x = Conv1D(32, kernel_size, activation='tanh', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)

    # Aplatit les données pour passer à la couche suivante
    x = Flatten()(x)
    x = Dense(64, activation='tanh')(x)
    encoder_output = Dense(latent_dim, activation='tanh', name="encoder_output")(x) # le goulot d'étranglement (bottleneck)

    # Crée le modèle d'encodeur
    encoder = Model(encoder_input, encoder_output, name="encoder")

    # --- Décodeur ---
    # Prend les variables latentes et tente de recréer les données d'origine.
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")

    # Calcule la forme nécessaire pour restaurer les données avant l'aplatissement
    pre_flatten_shape_x = int(num_features[0] / 4)
    pre_flatten_shape_y = 32
    
    x = Dense(64, activation='tanh')(decoder_input)
    x = Dense(pre_flatten_shape_x * pre_flatten_shape_y, activation='tanh')(x)
    # Remet les données dans leur forme 2D
    x = Reshape((pre_flatten_shape_x, pre_flatten_shape_y))(x)

    # Première couche de déconvolution pour reconstruire les données
    x = Conv1D(32, kernel_size, activation='tanh', padding='same')(x)
    x = UpSampling1D(2)(x) # Augmente la taille des données en double (répète chaque valeur 2 fois)

    # Deuxième couche de déconvolution
    x = Conv1D(16, kernel_size, activation='tanh', padding='same')(x)
    x = UpSampling1D(2)(x)

    # Couche de sortie pour que la forme corresponde à l'entrée d'origine
    decoder_output = Conv1D(1, kernel_size, activation='tanh', padding='same', name="decoder_output")(x)

    # Crée le modèle de décodeur
    decoder = Model(decoder_input, decoder_output, name="decoder")

    # --- Autoencodeur complet ---
    # Combine l'encodeur et le décodeur dans un seul modèle
    autoencoder_input = encoder_input
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = Model(autoencoder_input, autoencoder_output, name="autoencoder")

    return encoder, autoencoder

def objective(trial, X_train_data, Y_train_data):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # --- Cross-validation Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    rmse_scores = []
    epochs_list = []
    history_dicts = []

    # Boucle pour chaque pli de la validation croisée
    for train_idx, val_idx in kf.split(X_train_data):  
        X_train_cv, X_val_cv = X_train_data[train_idx], X_train_data[val_idx]
        Y_train_cv, Y_val_cv = Y_train_data.iloc[train_idx], Y_train_data.iloc[val_idx]
        
        # --- Etape 1: Construit l'autoencodeur et obtenir les variables latentes ---
        tf.keras.backend.clear_session() # Clear previous models from memory
        num_features = (X_train_cv.shape[1], X_train_cv.shape[2])
        encoder, autoencoder = build_cae_model(num_features, kernel_size, latent_dim)
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mean_squared_error' # The goal is to reconstruct the input
        )

        # Early stopping to prevent overfitting and speed up trials
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=20, restore_best_weights=True
        )

        # Entraîne le modèle
        autoencoder.fit(
            X_train_cv, X_train_cv,
            validation_data=(X_val_cv, X_val_cv),
            epochs=150,
            batch_size=batch_size,
            callbacks=[es_callback],
            verbose=0 
        )
        
        # --- Etape 2: Régression sur variables latentes ---
        # 1. obtenir les variables latentes
        Z_train_cv = encoder.predict(X_train_cv)
        Z_val_cv = encoder.predict(X_val_cv)

        # 2. Réression linéaire
        regressor = LinearRegression()
        regressor.fit(Z_train_cv, Y_train_cv)

        # 3. prédire et calculer le RMSE
        Y_pred_cv = regressor.predict(Z_val_cv)
        fold_rmse = np.sqrt(mean_squared_error(Y_val_cv, Y_pred_cv))
        rmse_scores.append(fold_rmse)
    
    avg_epochs = int(np.mean(epochs_list))
    trial.set_user_attr("avg_epochs", avg_epochs)
    trial.set_user_attr("epochs_list", epochs_list)
    trial.set_user_attr("fold_histories", history_dicts)

    return np.mean(rmse_scores)

# charge les fichiers CSV en DataFrames (X = prédicteurs, Y = réponses, M = partitions)
X = pd.read_csv("... X.csv", sep=';')
Y = pd.read_csv("... Y.csv", sep=';')
M = pd.read_csv("... M.csv", sep=';', na_values ='missing')

# Nom du modèle utilisé
Modd = "CAE"

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
        study_name=f"/optuna/cae_hpo_{Var}",
        load_if_exists=True
    )
    # Lance l'optimisation pour 'n_trials' essais
    study.optimize(lambda trial: objective(trial, X_train_f, Y_train), n_trials = 100) # prend ~ 15h avec un RTX 2060

    # Affiche les meilleurs résultats trouvés
    print("Meilleur essai :")
    print(f"  RMSE sur la validation : {study.best_value:.4f}")
    print("  Meilleurs paramètres :")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # --- Entraînement du modèle final ---
    best_params = study.best_trial.params

    encoder, autoencoder = build_cae_model(num_features, best_params["latent_dim"], best_params["kernel_size"])
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss='mean_squared_error'
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=2, restore_best_weights=True
    )

    history = autoencoder.fit(
        X_train_f, X_train_f,
        validation_split=0.3,
        epochs=150,
        batch_size=best_params["batch_size"],
        callbacks=[es_callback],
        verbose=1
    )
    # Encoder les données
    Z_train = encoder.predict(X_train_f, verbose=0)
    Z_test = encoder.predict(X_test_f, verbose=0)

    # Régression linéaire sur les variables latentes
    regressor = LinearRegression()
    regressor.fit(Z_train, Y_train)

    # Évalue sur les données de test
    Y_pred = regressor.predict(Z_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    print(f"\nTest RMSE: {test_rmse:.4f}")

    # --- Calcul des métriques et sauvegarde des résultats ---
    # Chemin vers le fichier de résultats
    csv_path = ";;; /resultats_CAE.csv"

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

