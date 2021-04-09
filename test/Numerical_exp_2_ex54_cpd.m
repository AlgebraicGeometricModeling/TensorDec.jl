function [min, med, max, ts, Ns]=Numerical_exp_2_ex54_cpd(r)

n=10;
T=zeros(n,n,n);

for j= 1:n
    for j= 1:n
        for j = 1:n
            
            
            T(j,j,j)= exp(sqrt(j)+j^2*i)+(j/n)*i;
        end
    end 
end 
for j= 1:n
    for k= 1:n
        for l = 1:n
            if j == k & j~=l
                T(j,k,l)=(j/n)*i;
            end 
        end
    end 
end 
d=zeros(50,1);
t=zeros(50,1);
Ni=zeros(50,1);

for s=1:50
    U2=cpd_rnd(n,r,'Real',@randn,'Imag',@randn);
    U2=U2([1 1 1]);
    tic,[sol,output]=cpd_nls(T,U2),t(s,1)=toc;
    Ni(s,1)=output.iterations;
    z=sol{1};
    T3=zeros(n,n,n)+1i*zeros(n,n,n);
    for j= 1:r
        b=outprod(z(:,j),z(:,j),z(:,j));
        T3=T3+b;
    end
 d(s,1)=frob(T-T3);
end
a1=sort(d);
min=a1(1,1)
med=a1(25,1)
max=a1(50,1)
ts=sum(t)/length(t)
Ns=sum(Ni)/length(Ni)
end
