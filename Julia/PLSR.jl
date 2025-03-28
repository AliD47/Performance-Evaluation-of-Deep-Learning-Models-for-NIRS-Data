using CSV, DataFrames, Plots, LinearAlgebra, Random, Jchemo, GLMakie, CairoMakie

X = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Xp.csv", DataFrame)
Y = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Y.csv", DataFrame)
M = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_M.csv", DataFrame)

function split_data(x, y, m, var)
    # Load CSV files as DataFrames
    
    # Ensure the variable exists in Y and M
    if !(var in names(y)) || !(var in names(m))
        throw(ArgumentError("Variable '$var' not found in Y or M."))
    end
    
    # Extract the partition column for the target variable
    mask = m[:, var]
    
    # Split X and Y based on M values
    X_cal = x[mask .== "cal", :]
    Y_cal = y[mask .== "cal", var]
    
    X_val = x[mask .== "val", :]
    Y_val = y[mask .== "val", var]
    
    X_test = x[mask .== "test", :]
    Y_test = y[mask .== "test", var]
    
    return (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test)
end
function save_results!(var_name::String, RMSE, optimal_nlv, R2, RE, RPD)
    global results_df  

    # Ensure the values are scalars
    RMSE = only(RMSE)
    R2 = only(R2)
    RE = only(RE)
    RPD = only(RPD)  # Ensure RPD is a scalar

    # Create a new row as a DataFrame
    new_row = DataFrame(Variable=[var_name], RMSE=[RMSE], Optimal_NLV=[optimal_nlv], 
                        R2=[R2], RE=[RE], RPD=[RPD])

    # Append the new row to results_df
    append!(results_df, new_row)
end
results_df = DataFrame(Variable=String[], RMSE=Float64[], Optimal_NLV=Int[], R2=Float64[], RE=Float64[], RPD=Float64[])


# # # # # # #   Variable to split data  # # # 
variables = ["adf", "adl", "cf", "cp", "dmdcell", "ndf"]
for Var in variables
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, M, Var)
    # size(X_cal)
    # size(Y_cal)
    # size(X_val)
    # size(Y_val)

    # Combine cal and val 
    X_train = vcat(X_cal, X_val)
    Y_train = vcat(Y_cal, Y_val)

    # Define model PLS
    nlv_range = 0:50
    model = plskern()
    res = gridscore(model, X_cal, Y_cal, X_val, Y_val; score = rmsep, pars = nothing, nlv = nlv_range, 
        verbose = false)

    u = findall(res.y1 .== minimum(res.y1))[1]
    res[u, :]
    mod = plskern(; nlv = res.nlv[u])
    fit!(mod, X_train, Y_train)
    pred = Jchemo.predict(mod, X_test).pred
    # @show rmsep(pred, Y_test)

    RMSE = only(rmsep(pred, Y_test))  
    optimal_nlv = res.nlv[u]  
    R2 = only(r2(pred, Y_test))
    RE = RMSE / only(minimum(Y_test))
    RPD = only(rpd(pred, Y_test))

    # Plot RMSEP vs. Number of LVs
    # Plots.plot(res.nlv, res.y1, xlabel="Nb. LVs", ylabel="RMSEP", label="RMSEP", lw=2)
    # Plots.scatter!([optimal_nlv], [res.y1[u]], label="Optimal", markersize=6, color=:red)
    # plotxy(vec(pred), Y_test; color = (:red, .5), bisect = true, xlabel = "Prediction",
    # ylabel = "Observed").f

    save_results!(Var, RMSE, optimal_nlv, R2, RE, RPD)
end

CSV.write("results_plsr.csv", results_df)