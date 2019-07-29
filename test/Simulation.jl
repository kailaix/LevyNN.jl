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

@testset "Levy" begin
    Δt = 0.5
    # A = zeros(2,2)
    A = diagm(0=>ones(2))
    b = zeros(2)
    λ = 1.0
    Jump = MvNormal(zeros(2), diagm(0=>ones(2)))
    ls = LevySimulator(A, b, λ, Jump, Δt)
    x0, Δx0 = simulate(ls, zeros(2), 1000)

    ξ = (rand(500,2) .-0.5)*2
    φ = evaluateECF(Δx0, ξ)
    φ2 = levyf(ξ, b, A, λ, Δt)
    close("all")
    scatter3D(ξ[:,1],ξ[:,2], abs.(φ), ".", label="simulated")
    scatter3D(ξ[:,1],ξ[:,2], abs.(φ2), ".", label="exact")
    legend()
end