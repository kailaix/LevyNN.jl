for uselog in 0 1 
do 
for nbasis in 5 10 20 
do 
srun julia rexp2.jl $uselog $nbasis &
srun julia rexp2.jl $uselog $nbasis &
done 
done 