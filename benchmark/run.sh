julia levy-1.jl TruncatedNormal2D NN 2 1000  &
julia levy-1.jl TruncatedNormal2D NN 5 1000  &
julia levy-1.jl TruncatedNormal2D NN 10 1000 &
wait %1 %2 %3

julia levy-1.jl TruncatedNormal2D RBF 10 1000 &
julia levy-1.jl TruncatedNormal2D RBF 20 1000 &
julia levy-1.jl TruncatedNormal2D RBF 40 1000 &
wait %1 %2 %3

julia levy-1.jl TruncatedNormal2D PL 10 1000 &
julia levy-1.jl TruncatedNormal2D PL 20 1000 &
julia levy-1.jl TruncatedNormal2D PL 40 1000 &
wait %1 %2 %3
