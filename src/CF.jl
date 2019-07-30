export LevyCF, StableCF, evaluate
mutable struct LevyCF
    A::PyObject
    b::PyObject
    ν::Function
    νx::PyObject
    Δt::Float64
    quad::Quadrature
end

function LevyCF(A::Union{Array{Float64}, PyObject}, b::Union{Array{Float64}, PyObject}, ν::Function, Δt::Float64, quad::Quadrature)
    if isa(A, Array)
        A = constant(A)
    end
    if isa(b, Array)
        b = constant(b)
    end
    points = quad.points
    νx = ν(points)
    LevyCF(A,b,ν,νx,Δt,quad)
end

function evaluate(cf::LevyCF, ξ::Union{Array{Float64},PyObject})
    local v
    b = cf.b
    A = cf.A
    weights = cf.quad.weights
    νx = cast(constant(cf.νx), ComplexF64)
    if isa(ξ, Array)
        ξ = constant(ξ)
    end
    function _evaluate(ξ)
        # 1.0im*cast(sum(b.*ξ), ComplexF64) - cast(squeeze(1/2*ξ'*A*ξ), ComplexF64) + cast(constant(sum((-1.0) .* weights .*  νx)), ComplexF64)
        ξx = cast(squeeze(cf.quad.points*reshape(ξ,2,1)), ComplexF64)
        1.0im*cast(sum(b.*ξ), ComplexF64) - cast(squeeze(1/2*ξ'*A*ξ), ComplexF64) + sum((exp(1.0im*ξx)-1) .* weights .*  νx)
    end
    if length(size(ξ))==1
        v = _evaluate(ξ)
    elseif length(size(ξ))==2
        v = tf.map_fn(_evaluate, ξ, dtype=tf.complex128)
    else
        error("Size of ξ is invalid!")
    end
    exp(v*cf.Δt)
end

mutable struct StableCF
    A::PyObject
    b::PyObject
    Γ::Function
    Γx::PyObject
    α::PyObject
    Δt::Float64
    quad::Quadrature
end


function StableCF(A::Union{Array{Float64}, PyObject}, b::Union{Array{Float64}, PyObject}, Γ::Function, α::Union{Float64, PyObject}, Δt::Float64, quad::Quadrature)
    if isa(A, Array)
        A = constant(A)
    end
    if isa(b, Array)
        b = constant(b)
    end
    if isa(α, Float64)
        α = constant(α)
    end
    Γx = Γ(quad.points)
    StableCF(A,b,Γ,Γx,α,Δt,quad)
end

function evaluate(cf::StableCF, ξ::Union{Array{Float64},PyObject})
    local v
    b = cf.b
    A = cf.A
    α = cf.α
    weights = cf.quad.weights
    points = cf.quad.points
    Γx = cf.Γx
    Γx = cast(constant(Γx), ComplexF64)
    function _evaluate(ξ)
        ξx = cast(squeeze(cf.quad.points*reshape(ξ,2,1)), ComplexF64)
        s = abs(ξx)^α
        1.0im*cast(sum(b.*ξ), ComplexF64) - cast(squeeze(1/2*ξ'*A*ξ), ComplexF64) -  cast(sum(s .* weights .* Γx), ComplexF64)
    end
    if isa(ξ, Array)
        ξ = constant(ξ)
    end
    if length(size(ξ))==1
        v = _evaluate(ξ)
    elseif length(size(ξ))==2
        v = tf.map_fn(_evaluate, ξ, dtype=tf.complex128)
    else
        error("Size of ξ is invalid")
    end
    exp(v*cf.Δt)
end

