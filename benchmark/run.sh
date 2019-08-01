for domain in TruncatedUniform2D MixedGaussian2D StandardNormal2D
do 
for btype in NN PL RBF
do 
for nbasis in 10 20 40
do 
julia benchmark/levy-1.jl $domain $btype $nbasis &
done
wait %1 %2 %3
done
done