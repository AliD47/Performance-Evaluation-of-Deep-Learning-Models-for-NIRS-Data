

# importe les paquets nécessaires
using CSV, DataFrames, Plots, LinearAlgebra, Random, Jchemo, GLMakie, CairoMakie

# charge les fichiers CSV en DataFrames (X = prédicteurs, Y = réponses, M = partitions)
X = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Xp.csv", DataFrame)
Y = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Y.csv", DataFrame)
M = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_M.csv", DataFrame)

function split_data(x, y, m, var)
    ## fonction pour séparer les jeux en cal/val/test pour la variable `var`
    
    # vérifie que la variable est présente dans Y et M
    if !(var in names(y)) || !(var in names(m))
        throw(ArgumentError("Variable '$var' not found in Y or M."))
    end
    
    # récupère la colonne de partitionnement correspondante dans M
    mask = m[:, var]
    
    # sépare les données
    X_cal = x[mask .== "cal", :]
    Y_cal = y[mask .== "cal", var]
    
    X_val = x[mask .== "val", :]
    Y_val = y[mask .== "val", var]
    
    X_test = x[mask .== "test", :]
    Y_test = y[mask .== "test", var]
    
    # retourne les trois partitions sous forme de tuples
    return (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test)
end
function save_results!(var_name::String, RMSE, optimal_nlv, R2, RE, RPD)
    ## fonction qui ajoute une ligne de résultats au DataFrame global results_df
    
    global results_df 
    # convertit les résultats en scalaires 
    RMSE = only(RMSE)
    R2 = only(R2)
    RE = only(RE)
    RPD = only(RPD)  

    # Nouvelle ligne avec les métriques
    new_row = DataFrame(Variable=[var_name], RMSE=[RMSE], Optimal_NLV=[optimal_nlv], 
                        R2=[R2], RE=[RE], RPD=[RPD])

    # ajoute la ligne au DataFrame global
    append!(results_df, new_row)
end

# initialise le DataFrame pour stocker les résultats
results_df = DataFrame(Variable=String[], RMSE=Float64[], Optimal_NLV=Int[], R2=Float64[], RE=Float64[], RPD=Float64[])


# # # # # # #   Variable to split data  # # # 
variables = ["adf", "adl", "cf", "cp", "dmdcell", "ndf"]
for Var in variables
    # split des données pour la variable courante
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, M, Var)

    # Combine cal and val 
    X_train = vcat(X_cal, X_val)
    Y_train = vcat(Y_cal, Y_val)

    # crée un modèle PLS et définit la plage de nombres de variables latentes (LVs)
    nlv_range = 0:60
    model = plskern()
    # grille de validation pour choisir le nombre optimal de LVs selon RMSEP
    res = gridscore(model, X_cal, Y_cal, X_val, Y_val; score = rmsep, pars = nothing, nlv = nlv_range, 
        verbose = false)

    # trouve le RMSEP min et construire le meilleur modèle
    u = findall(res.y1 .== minimum(res.y1))[1]
    res[u, :]
    mod = plskern(; nlv = res.nlv[u])
    fit!(mod, X_train, Y_train)
    # prédit sur l'ensemble test et récupère les prédictions
    pred = Jchemo.predict(mod, X_test).pred
    # @show rmsep(pred, Y_test)

    # calcule les métriques
    RMSE = only(rmsep(pred, Y_test))  
    optimal_nlv = res.nlv[u]  
    R2 = only(r2(pred, Y_test))
    RE = RMSE / (only(meanv(Y_test)))
    RPD = only(rpd(pred, Y_test))

    # sauvegarde les métriques pour la variable courante
    save_results!(Var, RMSE, optimal_nlv, R2, RE, RPD)
end
# affiche le DataFrame des résultats
results_df 
# écrit les résultats dans un fichier CSV
CSV.write("results_plsr.csv", results_df) 
