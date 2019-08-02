# btype = "RBF"
# nbasis = 10
# testfun = "step"
testfun = ARGS[1]
btype = ARGS[2]
nbasis = parse(Int64, ARGS[3])

@info btype, nbasis
using Revise
using Test
using ADCME
using LevyNN
using PyPlot
using SpecialFunctions
using Distributions
using LinearAlgebra
using DelimitedFiles
using Random; Random.seed!(233)


sess = Session()
if btype=="NN"
    global nbasis = div(nbasis,2)
    global rbf = NN([20*ones(Int64, nbasis);1], "nn")
elseif btype=="PL"  
    global rbf = PL1D(nbasis)
elseif btype=="RBF"
    global rbf = RBF1D(nbasis)
end
x = LinRange(0,2*π,30)|>Array
x1 = LinRange(0,2π,1000)

if testfun == "x2"
    global y = x.^2 + (2π .- x).^2
    global y1 = x1.^2 + (2π .- x1).^2
elseif testfun == "step"
    global y = Float64.(π/2 .< x .< π/2*3) .+ 0.5
    global y1 = Float64.(π/2 .< x1 .< π/2*3) .+ 0.5
end

y_ = evaluate_(rbf, x) + 0.5
y2_ = evaluate_(rbf, Array(x1)) + 0.5
loss = sum((y-y_)^2)

init(sess)
out = BFGS(sess, loss, 1000)
y2 = run(sess, y2_)
plot(x1, y1, "-", label="Exact")
plot(x1, y2, "--", label="$btype$nbasis")
plot(x, y, ">", label="Data")
if btype=="PL"
    n = length(rbf.θ)
    xx = LinRange(0,2π,n+1)[1:n]|>Array
    yy = run(sess, evaluate_(rbf, xx)+ 0.5)
    plot(xx, yy, "+", label="Nodal Values")
end
legend()
title("loss=$(round(run(sess, loss), sigdigits=4))")

cdir = @__DIR__
if !isdir("$cdir/data/compare/$btype$nbasis$testfun")
    mkdir("$cdir/data/compare/$btype$nbasis$testfun")
end
savefig("$cdir/data/compare/$btype$nbasis$testfun/result.png")
writedlm("$cdir/data/compare/$btype$nbasis$testfun/x1.txt",x1)
writedlm("$cdir/data/compare/$btype$nbasis$testfun/y1.txt",y1)
writedlm("$cdir/data/compare/$btype$nbasis$testfun/y2.txt",y2)
writedlm("$cdir/data/compare/$btype$nbasis$testfun/x.txt",x)
writedlm("$cdir/data/compare/$btype$nbasis$testfun/y.txt",y)
