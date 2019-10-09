T = 10.^( log10(.5):.1:log10(200.))
h=10
w = (2. *pi)./T
kh = qkhfs(w, h)

fid = fopen('matlab_results.txt','w')
for i=1:length(T)
   fprintf(fid,'%.8e, %.8e\n',w(i),kh(i));
end
fclose(fid)