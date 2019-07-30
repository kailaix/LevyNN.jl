export UniformDisk
struct UniformDisk
    r::Float64
end
function Base.:rand(uf::UniformDisk)
    r = uf.r
    # local x, y
    # ac = false
    # while !ac 
    #     x = (2*rand()-1)*r 
    #     y = (2*rand()-1)*r 
    #     if x^2 + y^2 <= r^2
    #         ac = true
    #     end
    # end
    d = Exponential()
    r = rand(d)
    θ = rand()*2π
    x = r*cos(θ); y = r*sin(θ)
    return [x; y]
end