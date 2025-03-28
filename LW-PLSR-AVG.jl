nlvdis = [5; 15; 25] ; metric = [:mah]
h = [1; 1.8; 2.5; 3.5; 5] ; k = [150; 300; 500; 600; 750; 1000] 
nlv = [0:10, 0:15, 0:20, 5:10, 5:15, 5:20]
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k, nlv = nlv)
length(pars[1])
model = lwplsravg()
zres = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, verbose = false)
u = findall(zres.y1 .== minimum(zres.y1))[1]
respar[j] = zres[u, :]    
model = lwplsravg(nlvdis = zres.nlvdis[u], metric = zres.metric[u], 
    h = zres.h[u], k = zres.k[u], nlv = zres.nlv[u], verbose = false)
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
respred[j] = [pred ytest]
@show resrmsep[j] = rmsep(pred, ytest)[1]
plotxy(ytest, vec(pred); bisect = true, color = (:blue, .3), xlabel = "Obs.", ylabel = "Pred.").f