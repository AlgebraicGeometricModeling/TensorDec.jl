using TensorDec

f = (u,v) -> 0.5*cos(0.7*pi*(u+v))+0.4*sin(4*pi*u)-0.2*cos(pi*v)
x = @polyvar x1 x2
L = monomials(x,0:5)
T = 10
mnt = (V->f(V[1]/10,V[2]/10))
sigma = series(mnt, L)

w, Xi = decompose(sigma)
