using Revise
using Test
using ADCME
using LevyNN
using PyPlot
using SpecialFunctions
using Distributions
using LinearAlgebra
sess = Session()

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
Jump = UniformDisk(1.0)
ls = LevySimulator(A, b, λ, Jump, Δt)
x0, Δx0 = simulate(ls, zeros(2), 5000)
ξ = (rand(5000,2) .-0.5)*3
φ = evaluateECF(Δx0, ξ)
# φ = levyf(ξ, b, A, λ, Δt)

# A = Variable(diagm(0=>ones(2))); b = Variable(zeros(2))
A = constant(A); b= constant(b)
# ν = x->zeros(size(x,1))
# rbf = RBF(5.0,20); ν = x->evaluate(rbf, x)
# rbf = PL(5.0,20); ν = x->evaluate(rbf, x)
quad = Quadrature2D(128, 5.0)
rbf = NN([20,20,20,1], "nn2"); ν = x->λ*evaluate(rbf, x)
# rbf = RBF(5.0,20); ν = x->λ*evaluate(rbf, x)
# rbf = PL(5.0,20); ν = x->evaluate(rbf, x)
# ν = x-> λ*exp.(-(x[:,1].^2+x[:,2].^2)/2)/2π
# ν = x-> zeros(size(x,1))
lcf = LevyCF(A, b, ν, Δt, quad)
f = evaluate(lcf, ξ)

weight = @. exp(-(ξ[:,1]^2 + ξ[:,2]^2))
loss = sum(abs(φ-f)^2 .* weight)
init(sess)
@info run(sess, loss)
# error()

BFGS(sess, loss, 2000)
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
global νx = run(sess, lcf.νx)
catch
    global νx = lcf.νx
end
scatter3D(quad.points[:,1],quad.points[:,2],abs.(νx), ".", label="learned")
v = exp.(-sum(quad.points.^2,dims=2)/2)/(2π)
# scatter3D(quad.points[:,1],quad.points[:,2],v, ".", label="exact")
legend()


