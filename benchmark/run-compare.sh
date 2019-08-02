# mkdir benchmark/compare
for testfun in step x2
do 
for btype in NN PL RBF
do 
for nbasis in 10 20 40
do 
julia benchmark/compare_basis.jl $testfun $btype $nbasis &
done
wait %1 %2 %3
done
done