using POMCPOW
using Random

t = CategoricalVector((1,0.1), 1.0)
insert!(t, 2, 2.0)

println("t: ", t)
