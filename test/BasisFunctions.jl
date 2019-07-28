@testset "RBF" begin
    rbf = RBF(2.0, 10)
    f = x->(x[1]-1)^2+x[2]^2
    x0 = 4.0*(rand(1000,2) .- 0.5)
    y = evaluate(rbf, x0)
    y0 = zeros(size(x0,1))
    for i = 1:size(x0,1)
        y0[i] = f(x0[i,:])
    end
    loss = sum((y-y0)^2)
    init(sess)
    BFGS(sess, loss, 1000)
    z0 = run(sess, y)
    close("all")
    scatter3D(x0[:,1], x0[:,2], z0, ".")
    scatter3D(x0[:,1], x0[:,2], y0, ".")
end

@testset "PL" begin
    pl = PL(2.0, 10)
    f = x->(x[1]-1)^2+x[2]^2
    x0 = 4.0*(rand(1000,2) .- 0.5)
    y = evaluate(pl, x0)
    y0 = zeros(size(x0,1))
    for i = 1:size(x0,1)
        y0[i] = f(x0[i,:])
    end
    loss = sum((y-y0)^2)
    init(sess)
    BFGS(sess, loss, 1000)
    z0 = run(sess, y)
    close("all")
    scatter3D(x0[:,1], x0[:,2], z0, ".")
    scatter3D(x0[:,1], x0[:,2], y0, ".")
end

@testset "NN" begin
    nn = NN([20,20,20,20,20,1], "test")
    f = x->(x[1]-1)^2+x[2]^2
    x0 = 4.0*(rand(1000,2) .- 0.5)
    y = evaluate(nn, x0)
    y0 = zeros(size(x0,1))
    for i = 1:size(x0,1)
        y0[i] = f(x0[i,:])
    end
    loss = sum((y-y0)^2)
    init(sess)
    BFGS(sess, loss, 1000)
    z0 = run(sess, y)
    close("all")
    scatter3D(x0[:,1], x0[:,2], z0, ".")
    scatter3D(x0[:,1], x0[:,2], y0, ".")
end