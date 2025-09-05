
# importe les paquets nécessaires
using CSV, DataFrames, Plots, LinearAlgebra, Random, Jchemo, GLMakie, CairoMakie, Statistics

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


# # # # # # #   Variable to use to split the data  # # # # # # # # 
variables = ["adf", "adl", "cf", "cp", "dmdcell", "ndf"]
for Var in variables
    # split des données pour la variable courante
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, M, Var)

    # Combine cal and val 
    X_train = vcat(X_cal, X_val)
    Y_train = vcat(Y_cal, Y_val)

    # Paramètres spécifiques à LWPLSR (local-weighted PLSR) 
    nlvdis = [5; 15; 25] ; metric = [:mah]
    ### nlvdis = 0 ; metric = [:eucl]
    h = [1; 1.8; 2.5; 3.5; 5] ; k = [150; 300; 500; 600; 750; 1000] 
    pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
    length(pars[1])
    nlv = 0:20
    model = lwplsr()
    # grille de validation
    res = gridscore(model, X_cal, Y_cal, X_val, Y_val; score = rmsep, pars, nlv, 
        verbose = false)
    group = string.("nlvdis=", res.nlvdis, "metric=", res.metric, "h=", res.h, " k=", res.k)
    
    ### plotgrid(res.nlv, res.y1, group; xlabel = "Nb. LVs", ylabel = "RMSEP").f
    # trouve le RMSEP min et construire le meilleur modèle
    u = findall(res.y1 .== minimum(res.y1))[1]
    res[u, :]
    model = lwplsr(; nlvdis = res.nlvdis[u], metric = res.metric[u],
          h = res.h[u], k = res.k[u], nlv = res.nlv[u])
    fit!(model, X_train, Y_train)
    # prédit sur l'ensemble test et récupère les prédictions
    pred = Jchemo.predict(model, X_test).pred
    ### @show rmsep(pred, Y_test)
    ### plotxy(vec(pred), Y_test; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f

    # calcule les métriques
    RMSE = only(rmsep(pred, Y_test))  
    optimal_nlv = res.nlv[u]  
    R2 = only(r2(pred, Y_test))
    RE = RMSE / (only(meanv(pred))) * 100
    RPD = only(rpd(pred, Y_test))

    # sauvegarde les métriques pour la variable courante
    save_results!(Var, RMSE, optimal_nlv, R2, RE, RPD)
end
# affiche le DataFrame des résultats
results_df
# écrit les résultats dans un fichier CSV
CSV.write("results_lwplsr.csv", results_df)