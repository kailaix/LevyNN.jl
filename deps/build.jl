using ADCME

function buildsrc(dir)
    cd(dir)
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd("../..")
end

buildsrc("quadrature/")
buildsrc("PL/")
buildsrc("PL1D/")
buildsrc("RBF/")
buildsrc("RBF1D/")
