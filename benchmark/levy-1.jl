using Revise
using Test
using ADCME
using LevyNN
using PyPlot
using SpecialFunctions
using Distributions
using LinearAlgebra
sess = Session()


Δt = 0.5
# A = zeros(2,2)
# A = [1.0 2.0;2.0 6.0]
# b = [2.0;3.0]
A = zeros(2,2)
b = zeros(2)
λ = 1.0
Jump = MvNormal(zeros(2), diagm(0=>ones(2)))
ls = LevySimulator(A, b, λ, Jump, Δt)
x0, Δx0 = simulate(ls, zeros(2), 1000)

ξ = (rand(500,2) .-0.5)*2
φ = evaluateECF(Δx0, ξ)

# A = Variable(diagm(0=>ones(2))); b = Variable(zeros(2))
A = constant(zeros(2,2)); b= constant(zeros(2))
rbf = NN([20,20,20,20,20,1], "nn"); ν = x->evaluate(rbf, x)
# rbf = RBF(5.0,20); ν = x->evaluate(rbf, x)
q = Quadrature2D(50, 5.0)
# rbf = Delta(size(q.points,1)); ν = x->evaluate(rbf, x)

cf = LevyCF(A, b, ν, Δt, q)
y = evaluate(cf, ξ)

loss = sum(abs(φ-y)^2)
init(sess)
BFGS(sess, loss, 500)

# yval = run(sess, y)
# scatter3D(ξ[:,1],ξ[:,2],abs.(yval), ".", label="learned")
# scatter3D(ξ[:,1],ξ[:,2],abs.(φ), ".", label="exact")
# legend()

νx = run(sess, cf.νx)
scatter3D(q.points[:,1],q.points[:,2],abs.(νx), ".", label="learned")
v = exp.(-sum(q.points.^2,dims=2)/2)/(2π)
scatter3D(q.points[:,1],q.points[:,2],v, ".", label="exact")
legend()


