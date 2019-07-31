using CMake


function buildsrc(dir)
    cd(dir)
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`$cmake ..`)
    run(`make -j`)
    cd("../..")
end

buildsrc("quadrature/")
buildsrc("PL/")
buildsrc("PL1D/")
buildsrc("RBF/")
buildsrc("RBF1D/")
