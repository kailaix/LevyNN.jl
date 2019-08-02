export BFGS, ADAM
function BFGS(sess::PyObject, loss::PyObject, max_iter=15000; kwargs...)
    __cnt = 0
    __loss = 0
    out = []
    function print_loss(l)
        if mod(__cnt,1)==0
            println("iter $__cnt, current loss=",l)
        end
        __loss = l
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ ITER $__iter ===============")
        end
        push!(out, __loss)
        __iter += 1
    end
    opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",options=Dict("maxiter"=> max_iter, "ftol"=>1e-12, "gtol"=>1e-12))
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss])
    out
end

function ADAM(sess::PyObject, loss::PyObject, max_iter=10000; kwargs...)
    opt = AdamOptimizer().minimize(loss)
    init(sess)
    for i = 1:max_iter
        l, _ = run(sess, [loss, opt])
        println("Iter $i, loss = $l")
    end
end