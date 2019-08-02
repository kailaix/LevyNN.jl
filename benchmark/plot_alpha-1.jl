
#=
domain = "Γstep"
close("all")
btype = "NN"; nbasis = 40; include("benchmark/plot_alpha-1.jl")
btype = "PL"; nbasis = 10; include("benchmark/plot_alpha-1.jl")
btype = "RBF"; nbasis = 20; include("benchmark/plot_alpha-1.jl")
ξ0 = LinRange(0,2π,1000)
plot(ξ0, ones(1000), "k-", label="reference")
legend()
xlabel("x"); ylabel("y")
=#
@info domain, btype, nbasis
using Revise
using Test
using ADCME
using LevyNN
using PyPlot
using MAT
using SpecialFunctions
using Distributions
using LinearAlgebra
using DelimitedFiles
using Random; Random.seed!(233)
reset_default_graph()
sess = Session()

# Jump = eval(Jump)()
# Jump = TruncatedNormal2D()
# Γ = Γuniform
Γ = eval(Meta.parse(domain))
nξ = 1000

Δt = 0.5
A = zeros(2,2)
b = zeros(2)
# A = [1.0 2.0;2.0 6.0]
# b = [1.0;2.0]
λ = 1.0
α = 1.5
# Jump = MvNormal(zeros(2), diagm(0=>ones(2)))
# Jump = MixedGaussian()
ls = StableSimulator(A, b, α, λ, Γ, Δt)
x0, Δx0 = simulate(ls, zeros(2), 10000)
ξ = (rand(nξ,2) .-0.5)*3
φ = evaluateECF(Δx0, ξ)
# φ = levyf(ξ, b, A, λ, Δt)
# error()
# A = Variable(diagm(0=>ones(2))); b = Variable(zeros(2))
A = constant(A); b= constant(b)
# ν = x->zeros(size(x,1))
# rbf = RBF(5.0,20); ν = x->evaluate(rbf, x)
# rbf = PL(5.0,20); ν = x->evaluate(rbf, x)
# if btype=="NN"
#     global rbf = NN([20*ones(Int64, nbasis);1], "nn")
# elseif btype=="RBF"
#     global rbf = RBF(5.0,nbasis)
# elseif btype=="PL"
#     global rbf = PL(5.0,nbasis)
# end

quad = Quadrature1D(100)

if btype=="NN"
    global rbf = NN([20*ones(Int64, div(nbasis,2));1], "nn5")
elseif btype=="PL"  
    global rbf = PL1D(nbasis)
elseif btype=="RBF"
    global rbf = RBF1D(nbasis)
elseif btype=="Delta"
    global rbf = Delta(length(quad.weights))
end

Γ_var = x->abs(evaluate1D(rbf,x))


α_var = Variable(1.0)
lcf = StableCF(A, b, Γ_var, α_var, Δt, quad)
f = evaluate(lcf, ξ)
# error()
weight = @. exp(-(ξ[:,1]^2 + ξ[:,2]^2))
loss = sum(abs(φ-f)^2 )#.* weight)
init(sess)
load(sess, "$(@__DIR__)/data/$domain$btype$nbasis/data.mat")
# err = L2error1D(sess, Γ, Γ_var)
# println("α=$(run(sess, α_var)), err=$err")
ξ0 = LinRange(0,2π,1000)
ξ = [cos.(ξ0) sin.(ξ0)]
v2 = run(sess, Γ_var(ξ))
plot(ξ0, v2, "--", label="$btype$(div(nbasis,2))")
