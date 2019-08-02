for domain in Γstep Γuniform
do 
for btype in PL RBF
do 
for nbasis in 10 20 40
do 
julia benchmark/alpha-1.jl $domain $btype $nbasis &
julia benchmark/alpha-1-0.5.jl $domain $btype $nbasis &
done
wait %1 %2 %3 %4 %5 %6
done
done

# julia benchmark/alpha-1.jl Γuniform NN 10 &
# julia benchmark/alpha-1.jl Γuniform NN 20 &
# julia benchmark/alpha-1.jl Γuniform NN 40 &
