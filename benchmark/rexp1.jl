# domain = "Γuniform"
# btype = "PL"
# nbasis = 10
# nbasis = parse(Int64, ARGS[3])

# @info domain, btype, nbasis
using Revise
using ADCME
using LevyNN
using PyPlot
using SpecialFunctions
using Distributions
using LinearAlgebra
using DelimitedFiles
using Random; Random.seed!(233)

reset_default_graph()
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

Γ = Γstep
nξ = 1000
btype = "NN"
nbasis = 5

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

sess = Session()
φ = run(sess, f)

quad = Quadrature1D(100)

if btype=="NN"
    global fn = NN([20*ones(Int64, nbasis);1], "nn")
elseif btype=="PL"  
    global fn = PL1D(nbasis)
elseif btype=="RBF"
    global fn = RBF1D(nbasis)
elseif btype=="Delta"
    global fn = Delta(length(quad.weights))
end

Γ_var = x->abs(evaluate1D(fn,x))
α_var = Variable(1.0)

# Γ_var = Γ
# α_var = constant(α)

lcf = StableCF(A, b, Γ_var, α_var, Δt, quad)
f = evaluate(lcf, ξ)
loss = sum(abs(φ-f)^2 )


ξ = LinRange(0,2π,1000)
ξ = [cos.(ξ) sin.(ξ)]
v1 = Γ(ξ)
v2 = Γ_var(ξ)
err = norm(v1-v2)


init(sess)
@info run(sess, loss)

@show sum([length(x) for x in get_collection()])
# error()

cb = (vs, iter, loss)->begin 
    printstyled("[#$iter] $(vs[1]), err=$(vs[2]) loss=$loss\n", color=:green)
end
out = BFGS!(sess, loss, 15000; callback=cb, vars=[α_var, err])
@info run(sess, [α_var,loss])
error()
# if !isdir("$(@__DIR__)/data/benchmark/$domain$btype$nbasis")
#     mkdir("$(@__DIR__)/data/benchmark/$domain$btype$nbasis")
# end
# save(sess, "$(@__DIR__)/data/benchmark/$domain$btype$nbasis/data.mat")
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/loss.txt", out)

err = L2error1D(sess, Γ, Γ_var)
αval = run(sess, α_var)
println("α=$αval, err=$err")
writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/alpha.txt", [err αval])

close("all")
try
    global νx = run(sess, lcf.Γx)
catch
    global νx = lcf.νx
end
plot(quad.θ,νx, "--", label="learned")
plot(quad.θ,Γ(quad.points), "-", label="exact")
legend()
# savefig("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/result.png")
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/theta.txt", quad.θ,)
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/vx.txt", νx)
# writedlm("$(@__DIR__)/data/benchmark/$domain$btype$nbasis/exact.txt", Γ(quad.points))



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


