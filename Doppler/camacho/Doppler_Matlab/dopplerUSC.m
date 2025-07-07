clear all
close all

load dop_burbujas.mat               %Cargo los datos adquiridos
[Nrep,Nmu]=size(dop_burbujas);      %Cantidad de trazas y muestras por traza
fs=40e6;                            %Frecuencia de muestreo
Ts=1/fs;                            %Periodo de muestreo
fe=4.4e6;                           %Frecuencia emitida
prf=1e3;                            %Frecuencia de repetición
Tprf=1/prf;                         %Periodo de repetición

t=0:Ts:Nmu*Ts-Ts;                       %Vector de tiempo por traza
f=-fs/2:fs/Nmu:fs/2-fs/Nmu;             %Vector de frecuencia por traza
tdop=0:Tprf:Nrep*Tprf-Tprf;             %Vector de tiempo doppler
fdop=-prf/2:prf/Nrep:prf/2-prf/Nrep;    %Vector de frecuencia doppler

%BSCAN
figure(1)
Bscan=zeros(Nrep,Nmu);
Bscan=20*log10(abs(hilbert(dop_burbujas))+10^(-10/20));
imagesc([0 tdop(end)],[0 t(end)],Bscan')
title('BSCAN en escala logaritmica')
xlabel('Tiempo en 1/PRF  (s)')
ylabel('Tiempo en 1/FS   (s)')


%DEMODULACION
rd=sin(2*pi*fe*t);  %Señal de referencia en fase
rq=cos(2*pi*fe*t);  %Señal de referencia en cuadratura
d=zeros(Nrep,Nmu);
q=zeros(Nrep,Nmu);
for i=1:Nrep        %Demoldulo cada señal
    d(i,:)=dop_burbujas(i,:).*rd;
    q(i,:)=dop_burbujas(i,:).*rq;
end

%Elijo una señal como ejemplo
iej=1700;
traza=dop_burbujas(iej,:);
trazad=d(iej,:);
trazaq=q(iej,:);

figure(2)
subplot(3,1,1)
plot(t,traza)
title('Señal original')
xlabel('Tiempo (s)')
subplot(3,1,2)
plot(t,trazad)
title('Demodulacion en Fase')
xlabel('Tiempo (s)')
subplot(3,1,3)
plot(t,trazaq)
title('Demodulacion en Cuadratura')
xlabel('Tiempo (s)')

figure(3)
subplot(3,1,1)
plot(f,fftshift(abs(fft(traza))));
title('FFT de la señal original')
xlabel('Frecuencia (Hz)')
subplot(3,1,2)
plot(f,fftshift(abs(fft(trazad))));
title('FFT de la señal en fase')
xlabel('Frecuencia (Hz)')
subplot(3,1,3)
plot(f,fftshift(abs(fft(trazaq))));
title('FFT de la señal en cuadratura')
xlabel('Frecuencia (Hz)')

%PROMEDIADO Y SEÑALES DOPPLER
D=mean(d');                     %Promedio sobre toda la ventana para obtener cada muestra doppler
Q=mean(q');

figure(4)
subplot(2,1,1)
plot(tdop,D)
title('Señal doppler en fase')
xlabel('Tiempo (s)')
subplot(2,1,2)
plot(tdop,Q)
title('Señal doppler en cuadratura')
xlabel('Tiempo (s)')

%FLUJO DIRECTO E INVERSO
F=real(hilbert(Q))+imag(hilbert(D));    %Calculo de los flujos directo e inverso mediante la transf. de Hilbert.
R=real(hilbert(D))+imag(hilbert(Q));

figure(5)
subplot(2,1,1)
plot(tdop,F)
title('Señal doppler de flujo directo')
xlabel('Tiempo (s)')
subplot(2,1,2)
plot(tdop,R)
title('Señal doppler de flujo inverso')
xlabel('Tiempo (s)')

%ANALISIS MEDIANTE STFT
FF=abs(specgram(F,128,1000,128,115));
RR=abs(specgram(R,128,1000,128,115));

[Nfft,Nint]=size(FF);                 %Obtengo el tamaño de la STFT
STFT=zeros(2*Nfft,Nint);              %Defino la matriz de imagen
STFT(1:Nfft,:)=flipud(FF);            %Ordeno los términos directo e inverso
STFT(Nfft+1:2*Nfft,:)=RR;

%Genero un vector de tiempos para la STFT
tFFT=64:(Nrep-64-77)/Nint:Nrep-77-4859/Nint;

%Grafico la SFTF en dB
figure(6)
imagesc([64 Nrep-77],[prf/2 -prf/2],20*log10(STFT));    %Grafico la STFT
set(gca,'Ydir','normal')
title('STFT en decibelios')
xlabel('Tiempo (s)')
ylabel('Frecuencia (Hz)')

%CALCULO DE MEDIA Y VARIANZA POR STFT
fmediaFFT=zeros(1,Nint);                        %Defino el vector
varFFT=zeros(1,Nint);
f_stft=fliplr(-prf/2:prf/2/Nfft:prf/2-prf/2/Nfft);      %Defino un vector de frecuencias con la cantidad de intervalos de la STFT
for i=1:Nint                                            %Calculo la media y la varianza
    fmediaFFT(i)=sum(f_stft.*STFT(:,i)')/sum(STFT(:,i));
    varFFT(i)=sum((f_stft-fmediaFFT(i)).^2.*STFT(:,i)')/sum(STFT(:,i));
end
stdFFT=sqrt(varFFT);

%grafico la media y vanianza por STFT
figure(7)
imagesc([64 Nrep-77],[prf/2 -prf/2],STFT);
set(gca,'Ydir','normal')
%colormap('gray');
hold on
plot(tFFT,fmediaFFT,'r','LineWidth',2)
plot(tFFT,fmediaFFT+stdFFT/3*2,'y:',tFFT,fmediaFFT-stdFFT/3*2,'y:')
hold off
title('Frecuencia media y Desv. estándar a partir de la FFT')
xlabel('Tiempo (s)')
ylabel('Frecuencia (Hz)')


%FRECUENCIA MEDIA Y VARIANZA POR AUTO-CORRELACION M=8

Ncorr=8;                                 %Cantidad de muestras utilizadas para el cálculo
fmediaACOR=zeros(1,fix(Nrep/Ncorr));     %Defino los vectores
varACOR=zeros(1,fix(Nrep/Ncorr));
stdACOR=zeros(1,fix(Nrep/Ncorr));
k=1;
for i=1:Ncorr:Nrep-Ncorr
    D1=D(i:i+Ncorr-1);                  %Obtengo las muestras a procesar
    Q1=Q(i:i+Ncorr-1);
    z=D1+j*Q1;                          %Genero el vector complejo
    zc=D1-j*Q1;                         %Genero el complejo conjugado
    zcT=[0 zc(1:length(zc)-1)];         %Retardo el complejo conjugado
    z1=z.*zcT;                          %Multiplico
    z1r=sum(real(z1));                  %Obtengo parte real a imaginaria de la autocorrelacion
    z1i=sum(imag(z1));
    fmediaACOR(k)=angle(z1r+j*z1i)*prf/(2*pi);  %Calculo la frecuencia media
    RT=z1r+j*z1i;
    R0=sum(D1.^2+Q1.^2);                %Obtengo la correlación en tau=0
    varACOR(k)=prf^2*(1-abs(RT)/R0);    %Obtengo la varianza
    stdACOR(k)=sqrt(varACOR(k))/2/pi;   %Obtengo la desviación estándar
    k=k+1;
end

%Genero un vector de tiempos para la frecuencia media por ACORR
tACOR=0:Nrep/length(fmediaACOR):Nrep-Nrep/length(fmediaACOR);

figure(8)
imagesc([64 Nrep-77],[prf/2 -prf/2],STFT);
set(gca,'Ydir','normal')
%colormap('gray');
hold on
plot(tACOR,fmediaACOR,'r','LineWidth',2)
plot(tACOR,fmediaACOR+stdACOR,'y:',tACOR,fmediaACOR-stdACOR,'y:')
hold off
title('Frecuencia media y Desviación estándar por AUTOCORRELACION con M=8')
xlabel('Tiempo (s)')
ylabel('Frecuencia (Hz)')

%ANALISIS DOPPLER COLOREADO
nventanas=80;                   %nº de sub-ventanas
nmuv=floor(Nmu/nventanas);      %cantindad de muestras por subventana
Dm=zeros(nventanas,Nrep);            %defino las matrices
Qm=zeros(nventanas,Nrep);
dtemp=zeros(Nrep,nventanas);
qtemp=zeros(Nrep,nventanas);
k=1;
for i=1:nmuv:Nmu-nmuv
    dtemp=d(:,i:i+nmuv-1);      %Defino la submatriz de cada intevalo
    Dm(k,:)=mean(dtemp');       %Promedio sobre cada sub-ventana
    qtemp=q(:,i:i+nmuv-1);
    Qm(k,:)=mean(qtemp');    
    k=k+1;
end

Ncorr=4;                                 %Cantidad de muestras utilizadas para el cálculo
fmediaACOR=zeros(nventanas,fix(Nrep/Ncorr));     %Defino los vectores
varACOR=zeros(nventanas,fix(Nrep/Ncorr));
stdACOR=zeros(nventanas,fix(Nrep/Ncorr));
k=1;
for i=1:Ncorr:Nrep-Ncorr
    for m=1:nventanas
        D1=Dm(m,i:i+Ncorr-1);                  %Obtengo las muestras a procesar
        Q1=Qm(m,i:i+Ncorr-1);
        z=D1+j*Q1;                          %Genero el vector complejo
        zc=D1-j*Q1;                         %Genero el complejo conjugado
        zcT=[0 zc(1:length(zc)-1)];         %Retardo el complejo conjugado
        z1=z.*zcT;                          %Multiplico
        z1r=sum(real(z1));                  %Obtengo parte real a imaginaria de la autocorrelacion
        z1i=sum(imag(z1));
        if mean(abs(D1))>2 & mean(abs(Q1))>2
            fmediaACOR(m,k)=angle(z1r+j*z1i)*prf/(2*pi);  %Calculo la frecuencia media
        else
            fmediaACOR(m,k)=0;
        end
        RT=z1r+j*z1i;
        R0=sum(D1.^2+Q1.^2);                %Obtengo la correlación en tau=0
        varACOR(m,k)=prf^2*(1-abs(RT)/R0);    %Obtengo la varianza
        stdACOR(m,k)=sqrt(varACOR(k))/2/pi;   %Obtengo la desviación estándar
    end
    k=k+1;
end

figure(9)
imagesc([0 tdop(end)],[0 t(end)],(Bscan)')
colormap('gray')
title('BSCAN en escala logaritmica')
xlabel('Tiempo en 1/PRF  (s)')
ylabel('Tiempo en 1/FS   (s)')

figure(10)
imagesc([0 tdop(end)],[0 t(end)],(fmediaACOR))
%defino el colormap
cm=zeros(64,3);
dc=[0.20:0.80/30:1];
cm(1:31,1)=flipud(dc');
cm(34:64,3)=dc';
colormap(cm)
title('Imagen de Doppler Coloreado')
xlabel('Tiempo en 1/PRF  (s)')
ylabel('Tiempo en 1/FS   (s)')

figure(11)
imagesc([64 Nrep-77],[prf/2 -prf/2],20*log10(flipud(STFT)));    %Grafico la STFT
set(gca,'Ydir','normal')
title('STFT en decibelios')
xlabel('Tiempo (s)')
ylabel('Frecuencia (Hz)')