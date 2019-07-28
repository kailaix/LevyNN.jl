export Quadrature1D, Quadrature2D, supported_quadrature_order

struct Quadrature
    points::Array{Float64}
    weights::Array{Float64}
    n::Int64
end

function supported_quadrature_order()
    [Array{Int64}(1:50);64; 128; 192; 256; 320; 448; 512; 576; 640; 704; 768; 832; 896; 960; 1024; 2048]
end

function Quadrature1D(n::Int64)
    θ = LinRange(0,2π,n+1)[1:n]
    points = [cos.(θ) sin.(θ)]
    weights = 2π/n*ones(n)
    Quadrature(points, weights, length(weights))
end

function Quadrature2D(n::Int64, R::Float64=1.0, x::Float64=0.0, y::Float64=0.0)
    if !(n in supported_quadrature_order())
        error("order $n not supported\n Supported orders: $(supported_quadrature_order())")
    end
    points, weights = quadgauss(n,x,y,R)
    Quadrature(points, weights, length(weights))
end