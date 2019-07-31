for domain in TruncatedUniform2D MixedGaussian2D StandardNormal2D
do 
for btype in NN PL RBF
do 
for nbasis in 10 20 40
do 
echo benchmark/levy-1.jl $domain $btype $nbasis &
done
wait %1 %2 %3
done
done

for domain in Γuniform Γx2 Γstep
do 
for btype in NN PL RBF
do 
for nbasis in 10 20 40
do 
echo benchmark/alpha-1.jl $domain $btype $nbasis &
done
wait %1 %2 %3
done
done