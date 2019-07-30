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