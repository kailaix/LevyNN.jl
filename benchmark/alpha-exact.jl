domain = "Γuniform"
btype = "PL"
nbasis = 10
# domain = ARGS[1]
# btype = ARGS[2]
# nbasis = parse(Int64, ARGS[3])

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

function φStableCF_(ξ, A, b, Δt, c)
    v = 1.0im * b'*ξ - 1/2*ξ'*A*ξ + c * (sum(ξ.^2))^(0.75/2)
    # v = -π
    v = exp(Δt*v)
end

function φStableCF(ξ, A, b, Δt, c)
    g = zeros(ComplexF64, size(ξ,1))
    for i = 1:size(ξ,1)
        g[i] = φStableCF_(ξ[i,:], A, b, Δt, c)
    end
    g
end



sess = Session()

# Jump = eval(Jump)()
# Jump = TruncatedNormal2D()
# Γ = Γuniform
Γ = eval(Meta.parse(domain))
nξ = 1000

Δt = 0.5
A = zeros(2,2)
b = zeros(2)
λ = 1.0
α = 0.75

ls = StableSimulator(A, b, α, λ, Γ, Δt)
x0, Δx0 = simulate(ls, zeros(2), 10000)
ξ = (rand(nξ,2) .-0.5)*3

# φ = evaluateECF(Δx0, ξ)

A = constant(A); b= constant(b)
lcf = StableCF(A, b, Γ, α, Δt, Quadrature1D(10000))
f = evaluate(lcf, ξ)
φ = run(sess, f)
# error()

quad = Quadrature1D(100)

if btype=="NN"
    global rbf = NN([20*ones(Int64, div(nbasis,2));1], "nn4")
elseif btype=="PL"  
    global rbf = PL1D(nbasis)
elseif btype=="RBF"
    global rbf = RBF1D(nbasis)
elseif btype=="Delta"
    global rbf = Delta(length(quad.weights))
end

# TODO Change the following codes for test
Γ_var = x->abs(evaluate1D(rbf,x))
α_var = Variable(1.0)

# Γ_var = Γ
# α_var = constant(α)

lcf = StableCF(A, b, Γ_var, α_var, Δt, quad)
f = evaluate(lcf, ξ)
loss = sum(abs(φ-f)^2 )
init(sess)

@info run(sess, loss)

out = BFGS(sess, loss, 100)
@info run(sess, [α_var,loss])
# if !isdir("$(@__DIR__)/data/benchmark/$domain$btype$nbasis")
#     mkdir("$(@__DIR__)/data/benchmark/$domain$btype$nbasis")
# end
# save(sess, "$(@__DIR__)/data/benchmark/$domain$btype$nbasis/data.mat")
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/loss.txt", out)

err = L2error1D(sess, Γ, Γ_var)
αval = run(sess, α_var)
println("α=$αval, err=$err")
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/alpha.txt", [err αval])

error()
# close("all")
try
    global νx = run(sess, lcf.Γx)
catch
    global νx = lcf.νx
end
plot(quad.θ,νx, "--", label="learned")
plot(quad.θ,Γ(quad.points), "-", label="exact")
legend()

error()

# FOR DEBUG
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


