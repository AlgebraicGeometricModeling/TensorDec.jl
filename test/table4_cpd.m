function [min,med,max,t,Ni]=table4_cpd(r,eps)
n=10;
N=100;
d1=zeros(N,1);
ts=zeros(N,1);
N1=zeros(N,1);
for s= 1:N
    U0=cpd_rnd(n,r,'Real',@randn,'Imag',@randn);
    U0=U0([1 1 1 1]);
    T0=cpdgen(U0);
    E0=randn(n,n,n,n)+1i*randn(n,n,n,n);
    E=zeros(n,n,n,n)+1i*zeros(n,n,n,n);
    for p=perms(1:ndims(T0))'
        E=E+permute(E0,p);
    end
    E=E/factorial(ndims(T0));
    E=E/frob(E);
    T=T0+eps*E;
    tic,U2=cpd_rnd(n,r,'Real',@randn,'Imag',@randn);
    %U2=cpd_gevd(T,r);
    U2=U2([1 1 1 1]);
    [sol,output]=cpd_nls(T,U2),ts(s,1)=toc;
    N1(s,1)=output.iterations;
    z=sol{1};
    T3=zeros(n,n,n,n)+1i*zeros(n,n,n,n);
    for j= 1:r
        b=outprod(z(:,j),z(:,j),z(:,j),z(:,j));
        T3=T3+b;
    end
    d1(s,1)=frob(T0-T3)/eps;
end
a1=sort(d1);
min=a1(1,1)
med=a1(N/2,1)
max=a1(N,1)
t=sum(ts)/length(ts)
Ni=sum(N1)/length(N1)
end
