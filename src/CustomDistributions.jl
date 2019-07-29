export CirUniform
# Distributions on a circle
struct CirUniform end
function Base.:rand(d::CirUniform)
    θ = rand()*2π
    [cos(θ);sin(θ)]
end

