# domain = "TruncatedUniform2D"
# btype = "Delta"
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
sess = Session()

Jump = eval(Meta.parse(domain))()
nξ = 1000


function levyf_(ξ, b, A, λ, Δt)
    v = 1.0im * b'*ξ - 1/2*ξ'*A*ξ + λ*(exp(-sum(ξ.^2)/2)-1)
    return exp(Δt*v)
end

function levyf(ξ, b, A, λ, Δt)
    out = zeros(ComplexF64, size(ξ,1))
    for i = 1:size(ξ,1)
        out[i] = levyf_(ξ[i,:], b, A, λ, Δt)
    end
    out
end

Δt = 0.5
A = zeros(2,2)
b = zeros(2)
# A = [1.0 2.0;2.0 6.0]
# b = [1.0;2.0]
λ = 1.0
# Jump = MvNormal(zeros(2), diagm(0=>ones(2)))
# Jump = MixedGaussian()
ls = LevySimulator(A, b, λ, Jump, Δt)
x0, Δx0 = simulate(ls, zeros(2), 10000)
ξ = (rand(nξ,2) .-0.5)*3
φ = evaluateECF(Δx0, ξ)
# φ = levyf(ξ, b, A, λ, Δt)

# A = Variable(diagm(0=>ones(2))); b = Variable(zeros(2))
A = constant(A); b= constant(b)
# ν = x->zeros(size(x,1))
# rbf = RBF(5.0,20); ν = x->evaluate(rbf, x)
# rbf = PL(5.0,20); ν = x->evaluate(rbf, x)
quad = Quadrature2D(64, 5.0)
if btype=="NN"
    global rbf = NN([20*ones(Int64, div(nbasis,2));1], "nn")
elseif btype=="RBF"
    global rbf = RBF(5.0,nbasis)
elseif btype=="PL"
    global rbf = PL(5.0,nbasis)
elseif btype=="Delta"
    global rbf = Delta(length(quad.weights))
end
ν = x->λ*abs(evaluate(rbf, x))
lcf = LevyCF(A, b, ν, Δt, quad)
f = evaluate(lcf, ξ)

weight = @. exp(-(ξ[:,1]^2 + ξ[:,2]^2))
loss = sum(abs(φ-f)^2)# .* weight)
init(sess)
@info run(sess, loss)
# error()

out = BFGS(sess, loss, 15000)

@info run(sess, α_var)
if !isdir("$(@__DIR__)/data/2D$domain$btype$nbasis")
    mkdir("$(@__DIR__)/data/2D$domain$btype$nbasis")
end
save(sess, "$(@__DIR__)/data/2D$domain$btype$nbasis/data.mat")
writedlm("$(@__DIR__)/data/2D$domain$btype$nbasis/loss.txt", out)
close("all")
if btype=="Delta"
    try
        global νx = run(sess, lcf.νx)
    catch
        global νx = lcf.νx
    end
    scatter(quad.points[:,1],quad.points[:,2],c=νx,marker=".", linewidths=0)
    axis("equal")
    colorbar()
else
    pcolormesh(sess, ν, 1.5)
end
savefig("$(@__DIR__)/data/2D$domain$btype$nbasis/result.png")



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
    global νx = run(sess, lcf.νx)
catch
    global νx = lcf.νx
end
# scatter3D(quad.points[:,1],quad.points[:,2],νx, ".", label="learned")
# scatter(quad.points[:,1],quad.points[:,2],c=νx, linewidths=0)
pcolormesh(sess, ν, 1.5)
# v = exp.(-sum(quad.points.^2,dims=2)/2)/(2π)
# scatter3D(quad.points[:,1],quad.points[:,2],v, ".", label="exact")
legend()


