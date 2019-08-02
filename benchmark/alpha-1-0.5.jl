# domain = "Γstep"
# btype = "NN"
# nbasis = 9
domain = ARGS[1]
btype = ARGS[2]
nbasis = parse(Int64, ARGS[3])

@info domain, btype, nbasis
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
α = 0.5
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
@info run(sess, loss)
# error()

out = BFGS(sess, loss, 15000)
@info run(sess, α_var)
if !isdir("$(@__DIR__)/data/alpha0.5$domain$btype$nbasis")
    mkdir("$(@__DIR__)/data/alpha0.5$domain$btype$nbasis")
end
save(sess, "$(@__DIR__)/data/alpha0.5$domain$btype$nbasis/data.mat")
writedlm("$(@__DIR__)/data/alpha0.5$domain$btype$nbasis/loss.txt", out)

err = L2error1D(sess, Γ, Γ_var)
αval = run(sess, α_var)
println("α=$αval, err=$err")
writedlm("$(@__DIR__)/data/alpha0.5$domain$btype$nbasis/alpha.txt", [err αval])


close("all")
try
    global νx = run(sess, lcf.Γx)
catch
    global νx = lcf.νx
end
plot(quad.θ,νx, "--", label="learned")
plot(quad.θ,Γ(quad.points), "-", label="exact")
legend()
savefig("$(@__DIR__)/data/alpha0.5$domain$btype$nbasis/result.png")

# ADAM(sess, loss)
error()
close("all")
φ1 = run(sess, f)
scatter3D(ξ[:,1], ξ[:,2], abs.(φ1), ".", label="Quadrature")
scatter3D(ξ[:,1], ξ[:,2], abs.(φ), ".", label="Exact")
legend()

close("all")
φ1 = run(sess, f)
scatter3D(ξ[:,1], ξ[:,2], imag.(φ1), ".", label="Quadrature")
scatter3D(ξ[:,1], ξ[:,2], imag.(φ), ".", label="Exact")
legend()

close("all")
try
    global νx = run(sess, lcf.Γx)
catch
    global νx = lcf.νx
end
plot(quad.θ,νx, "--", label="learned")
plot(quad.θ,Γ(quad.points), "-", label="exact")
legend()
# ylim([0.0,1.5])


