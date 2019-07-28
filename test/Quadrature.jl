@testset "Quadrature2D" begin
    for n in supported_quadrature_order()
            q = Quadrature2D(n, 2.0)
            x = q.points[:,1]
            y = q.points[:,2]
            w = q.weights
            val = abs(sum(@. w*((x-1)^2+y^2)) - 37.699111843077)
            if n>1
                @test val < 1e-8
            end
    end
end

@testset "Quadrature2D Gaussian" begin
    for n in supported_quadrature_order()
            q = Quadrature2D(n, 5.0)
            x = q.points[:,1]
            y = q.points[:,2]
            w = q.weights
            val = abs(sum(@. w*exp(-x^2-y^2))-π)
            @show n, val
    end
end


@testset "Quadrature1D" begin
    n = 10
    q = Quadrature1D(n)
    val = sum(q.points[:,1].^2 .* q.weights)
    @test val ≈ π
end