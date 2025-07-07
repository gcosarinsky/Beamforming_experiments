clear all
close all
fid=fopen('BURBUJAS_LENTO_1_N3496_5000.bin','rb','l');
%fid=fopen('bURBUJAS_3quieto_3flujo.bin','rb','l');
%fid=fopen('bURBUJAS_saturando.bin','rb','l');
matriz=zeros(5000,1748);
matrizd=zeros(5000,1748);
matrizdf=zeros(5000,1748);
matrizq=zeros(5000,1748);
matrizqf=zeros(5000,1748);

t=0:25e-9:1747*25e-9;
f=-20e6:40e6/1748:20e6-40e6/1748;
%fs=4.04e6;
fs=4.4e6;
sref=sin(2*pi*fs*t);
srefq=cos(2*pi*fs*t);

[B,A]=butter(8,500e3/20e6);
[B2,A2]=butter(6,[0.1 0.95]);
[B3,A3]=butter(6,[2.5/20 7/20]);
%[B3,A3]=butter(6,[3.5/20 5.5/20]);

for i=1:5000
    matriz(i,:)=fread(fid,1748,'int16');
    %matriz(i,:)=filtfilt(B3,A3,matriz(i,:));
    fread(fid,3,'int8');
    matrizd(i,:)=matriz(i,:).*sref;
    %matrizdf(i,:)=filtfilt(B,A,matrizd(i,:));
    matrizq(i,:)=matriz(i,:).*srefq;
    %matrizqf(i,:)=filtfilt(B,A,matrizq(i,:));
end

% mm=zeros(5000,1748);
% for i=2:5000
%     mm(i,:)=matriz(i,:)-matriz(i-1,:);
% end
% matriz=mm;

% for i=1:5000
%     matrizd(i,:)=matriz(i,:).*sref;
%     matrizdf(i,:)=filtfilt(B,A,matrizd(i,:));
%     matrizq(i,:)=matriz(i,:).*srefq;
%     matrizqf(i,:)=filtfilt(B,A,matrizq(i,:));
% end

close all

figure
imagesc(abs(((matriz'))))
%imagesc(matriz')

figure
plot(matriz(1:500:5000,:)')

figure
plot(matrizd(1:500:5000,:)')

% figure
% plot(matrizdf(1:100:5000,:)')



% x=sum(matrizd(:,350:500)');
% y=sum(matrizq(:,350:500)');
x=mean(matrizd');
y=mean(matrizq');
figure
plot(f,fftshift(abs(fft(matriz(1000,:)))))
ff=real(hilbert(y))+imag(hilbert(x));
rr=real(hilbert(x))+imag(hilbert(y));
figure
subplot(2,1,1)
plot(ff)
subplot(2,1,2)
plot(rr)
    

FF=abs(specgram(ff,128,1000,128,115));
RR=abs(specgram(rr,128,1000,128,115));

[a,b]=size(FF);
imagen=zeros(2*a,b);
imagen(1:a,:)=flipud(20*log10(FF));
imagen(a+1:2*a,:)=20*log10(RR);
figure
imagesc(imagen);

[a,b]=size(FF);
imagen=zeros(2*a,b);
imagen(1:a,:)=flipud((FF));
imagen(a+1:2*a,:)=(RR);


%frecuenicia media por FFT
fmediaFFT=zeros(1,b);
f=-500:500/a:500-500/a;
for i=1:b
    fmediaFFT(i)=sum(fliplr(f).*imagen(:,i)')/sum(imagen(:,i));
end

%frecuencia media por ACORR
nrep=10;
prf=1000;
fmediaACOR=zeros(1,fix(length(x)/nrep));
varACOR=zeros(1,fix(length(x)/nrep));
stdACOR=zeros(1,fix(length(x)/nrep));
k=1;
for i=1:nrep:length(x)-nrep
    xx=x(i:i+nrep-1);
    yy=y(i:i+nrep-1);
    z=xx+j*yy;
    zc=xx-j*yy;
    zcT=[0 zc(1:length(zc)-1)];
    z1=z.*zcT;
    z1r=sum(real(z1));
    z1i=sum(imag(z1));
    fmediaACOR(k)=angle(z1r+j*z1i)*prf/(2*pi);
    %varianza
    RT=z1r+j*z1i;
    R0=sum(xx.^2+yy.^2);
    varACOR(k)=prf^2*(1-abs(RT)/R0);
    stdACOR(k)=sqrt(varACOR(k))/2/pi;
    k=k+1;
end


tFFT=64:4859/length(fmediaFFT):5000-77-4859/length(fmediaFFT);
tACOR=0:5000/length(fmediaACOR):5000-5000/length(fmediaACOR);

figure
imagesc([64 5000-77],[500 -500],imagen);
set(gca,'Ydir','normal')
hold on
plot(tACOR,fmediaACOR,'r',tACOR,fmediaACOR+stdACOR/3*2,'y:',tACOR,fmediaACOR-stdACOR/3*2,'y:')
hold off

figure
imagesc([64 5000-77],[500 -500],imagen);
set(gca,'Ydir','normal')
hold on
plot(tFFT,fmediaFFT,'r')
hold off

figure
plot(tFFT,fmediaFFT,tACOR,fmediaACOR)
ax=axis;
axis([ax(1) ax(2) -500 500])


fclose(fid);
