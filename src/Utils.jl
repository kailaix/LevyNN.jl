export L2error, pcolormesh, L2error1D
function PyPlot.:pcolormesh(sess::PyObject, ν::Function, R::Float64)
    ξ = LinRange(-R,R,100)
    ξx, ξy = np.meshgrid(ξ,ξ)
    g = ν([ξx[:] ξy[:]])
    val = run(sess, g)
    val = reshape(val, 100, 100)
    pcolormesh(ξx, ξy, val)
    xlabel("x")
    ylabel("y")
    colorbar()
    val
end

function PyPlot.:pcolormesh(uf, R::Float64)
    ξ = LinRange(-R,R,100)
    ξx, ξy = np.meshgrid(ξ,ξ)
    g = pdf(uf, [ξx[:] ξy[:]])
    val = reshape(g, 100, 100)
    pcolormesh(ξx, ξy, val)
    xlabel("x")
    ylabel("y")
    colorbar()
    val
end

function L2error(sess::PyObject, ν::Function, uf, R::Float64)
    ξ = LinRange(-R,R,100)
    ξx, ξy = np.meshgrid(ξ,ξ)
    g = ν([ξx[:] ξy[:]])
    val = run(sess, g)
    val = reshape(val, 100, 100)

    g = pdf(uf, [ξx[:] ξy[:]])
    val2 = reshape(g, 100, 100)

    sqrt(mean((val-val2)^2))
end

function L2error1D(sess::PyObject, Γ, Γ_var)
    ξ = LinRange(0,2π,1000)
    ξ = [cos.(ξ) sin.(ξ)]
    v1 = Γ(ξ)
    v2 = run(sess, Γ_var(ξ))
    sqrt(mean((v1-v2).^2))
end