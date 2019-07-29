function φLevyCF_(ξ, A, b, Δt)
    v = 1.0im * b'*ξ - 1/2*ξ'*A*ξ + π*(exp(-sum(ξ.^2)/4)-1)
    # v = -π
    v = exp(Δt*v)
end



function φLevyCF(ξ, A, b, Δt)
    g = zeros(ComplexF64, size(ξ,1))
    for i = 1:size(ξ,1)
        g[i] = φLevyCF_(ξ[i,:], A, b, Δt)
    end
    g
end

@testset "LevyCF" begin
    quad = Quadrature2D(20, 5.0)
    A = constant([3.0 2.0;2.0 4.0])
    b = constant([1.0;2.0])
    # A = constant(zeros(2,2))
    # b = constant(zeros(2))
    ν = x-> exp.(-(x[:,1].^2+x[:,2].^2))
    Δt = 0.1
    lcf = LevyCF(A, b, ν, Δt, quad)
    ξ = (rand(200,2) .-0.5)*2
    f = evaluate(lcf, ξ)
    V = run(sess, f)
    Vr = abs.(V); Vi = imag.(V)
    φ = φLevyCF(ξ, [3.0 2.0;2.0 4.0], [1.0;2.0], Δt)
    # φ = φLevyCF(ξ, zeros(2,2), zeros(2), Δt)
    close("all")
    scatter3D(ξ[:,1], ξ[:,2], Vr, ".", label="Quadrature")
    scatter3D(ξ[:,1], ξ[:,2], abs.(φ), ".", label="Exact")
    legend()

    close("all")
    scatter3D(ξ[:,1], ξ[:,2], Vi, ".", label="Quadrature")
    scatter3D(ξ[:,1], ξ[:,2], imag.(φ), ".", label="Exact")
    legend()
end


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


@testset "Stable" begin
    # g(ξ) = ∫|cos(θ)|ᵅdθ |ξ|ᵅ
    c = -(sqrt(π)* (7gamma(7/8) + 8gamma(15/8)))/(7gamma(11/8))
    
    quad = Quadrature1D(20)
    A = constant([3.0 2.0;2.0 4.0])
    b = constant([1.0;2.0])
    # A = constant(zeros(2,2))
    # b = constant(zeros(2))
    Γ = x->ones(size(x,1))
    Δt = 0.1
    cf = StableCF(A, b, Γ, 0.75, Δt, quad)
    ξ = (rand(200,2) .-0.5)*2
    f = evaluate(cf, ξ)
    V = run(sess, f)
    Vr = abs.(V); Vi = imag.(V)
    φ = φStableCF(ξ, [3.0 2.0;2.0 4.0], [1.0;2.0], Δt, c)
    # φ = φStableCF(ξ, zeros(2,2), zeros(2), Δt, c)
    close("all")
    scatter3D(ξ[:,1], ξ[:,2], Vr, ".", label="Quadrature")
    scatter3D(ξ[:,1], ξ[:,2], abs.(φ), ".", label="Exact")
    legend()

    close("all")
    scatter3D(ξ[:,1], ξ[:,2], Vi, ".", label="Quadrature")
    scatter3D(ξ[:,1], ξ[:,2], imag.(φ), ".", label="Exact")
    legend()
end
