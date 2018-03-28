using TensorDec

f = (u,v) -> 0.5*cos(0.7*pi*(u+v))+0.6*sin(4*pi*u)-0.2*cos(pi*v)
x = @ring x1 x2
L = monoms(x,5)
T = 10
mnt = (V->f(V[1]/T,V[2]/T))
sigma = series(mnt, L)

decompose(sigma)
