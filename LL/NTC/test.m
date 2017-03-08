s=1000;
nb=100;
a=ones(s,nb,1);
matlabpool(3)
tic
for k=1:10000
    parfor i=1:s
        a(i,:)=a(i,:)+1;
    end
end
toc
sum(a)

matlabpool close