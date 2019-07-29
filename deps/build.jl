using CMake

cd("quadrature/")
if !isdir("build")
    mkdir("build")
end
cd("build")
run(`$cmake ..`)
run(`make -j`)
cd("../..")

cd("RBF/")
if !isdir("build")
    mkdir("build")
end
cd("build")
run(`$cmake ..`)
run(`make -j`)
cd("../..")


cd("PL/")
if !isdir("build")
    mkdir("build")
end
cd("build")
run(`$cmake ..`)
run(`make -j`)
cd("../..")