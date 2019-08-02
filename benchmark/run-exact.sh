for domain in Γstep Γuniform Γx2
do 
for btype in NN PL RBF
do 
for nbasis in 10 20 40
do 
julia benchmark/alpha-exact.jl $domain $btype $nbasis &
done
wait %1 %2 %3
done
done