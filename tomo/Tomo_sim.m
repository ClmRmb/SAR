clear all, clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAR TOMOGRAPHY DATA SIMULATION AND FOCUSING
% THIS SCRIPT IS INTENDED TO BE USED FOR THE TRAINING COURSE SAR TOMOGRAPHY
% TRAINING, HELD IN BEIJING IN FEBRUARY 2015 BY LAURENT FERRO-FAMIL AND
% STEFANO TEBALDINI.
% THIS SCRIPT WAS DEVELOPED BY STEFANO TEBALDINI.
% YOU ARE WELCOME TO ADDRESS ME QUESTIONS/COMMENTS/CORRECTIONS AT
% stefano.tebaldini@polimi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RADAR PARAMETERS
B = 40e6;   % pulse bandwdith [Hz]
f0 = 1e9;  % carrier frequency [Hz]
c = 3e8;    % light velocity [m/s]
lambda = c/f0 % wavelength [m]
pho_r = 10*c/2/B % range resolution [m]
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ACQUISITION GEOMETRY
N = 20;     % number of flights
H = 1000;   % mean flight altitude
%Sz = H + zeros(1,N); % vertical sensor position
%Sy = [0:N-1]*3;        % horizontal sensor position

Sz = H + [0:N-1]*15; 
Sy = [0:N-1]*50;  

dy = pho_r/3;   % ground range bin spacing
%dy = pho_r/10;   % ground range bin spacing
y_ax = 5000 + [2000:dy:3000]; % ground range axis
%y_ax = [1000:dy:5000];

z_true = zeros(1,numel(y_ax));%20*hamming(length(y_ax))'; % true topography of the scene

figure
plot(Sy,Sz,'<r',y_ax,z_true,'k','LineWidth',2), grid
hold on, 
plot([Sy(1) y_ax(1)],[Sz(1) z_true(1)],'k')
plot([Sy(1) y_ax(end)],[Sz(1) z_true(end)],'k')
xlabel('ground range [m]'), ylabel('height [m]')
title('Acquisition geometry')
legend('Sensors','Scene topography')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAR GEOMETRY

% Reference topography for SAR processing
if 1 % true topgraphy
    z_ref = z_true;
else % flat terrain
    z_ref = 0*z_true;
   % z_ref(1600 - 50 : 1600 + 50) = z_ref(1600 - 50 : 1600 + 50) + 40;
   % z_ref(1600 - 90 : 1600 - 51) = z_ref(1600 - 90 : 1600 - 51) + (0:39);
   % z_ref(1600 + 51 : 1600 + 90) = z_ref(1600 + 51 : 1600 + 90) + (39:-1:0);
end
% Master
Master = 5
%Distances to reference topography (ground range)
R_ref = [];
for n = 1:N
    R_ref(n,:) = sqrt( (Sy(n)-y_ax).^2 + (Sz(n)-z_ref).^2);
end
% Slant range axis
r_min = min(R_ref(Master,:));
r_max = max(R_ref(Master,:));
%dr = (R_ref(Master,2)) -(R_ref(Master,1)) ;
dr = pho_r;
rg_ax = [r_min:dr:r_max];

%rg_ax = R_ref(Master,:);


Nr = length(rg_ax);


z_ref_of_r = interp1(R_ref(Master,:),z_ref,rg_ax);
y_ref_of_r = interp1(R_ref(Master,:),y_ax,rg_ax);

dy_radar = y_ref_of_r(12) - y_ref_of_r(11)

figure, 
subplot(2,1,1), plot(y_ax,z_ref,y_ref_of_r,z_ref_of_r), grid
xlabel('ground range [m]'), ylabel('height'), title('Topography - ground coordinates [m]')
subplot(2,1,2), plot(rg_ax,z_ref_of_r), grid
xlabel('slant range [m]'), ylabel('height'), title('Topography - SAR coordinates [m]')

%%

% Distances to reference topography (ground range)
clear R_ref
for n = 1:N
    R_ref(n,:) = sqrt( (Sy(n)-y_ref_of_r).^2 + (Sz(n)-z_ref_of_r).^2);
    Teta_ref(n,:) = acos ( (Sz(n)-z_ref_of_r)./R_ref(n,:) );
end
%% Phase to height conversion factors

r_ref = 20;
Kz = [];
for n = 1:N
    dteta = Teta_ref(n,:) - Teta_ref(Master,:);
    Kz(n,:) = -4*pi/lambda*dteta./sin(Teta_ref(n,:));
    %Kz(n,:) = -4*pi/lambda*dteta.*sin(Teta_ref(Master,:));
    Ky(n,:) = 4*pi/lambda*dteta.*cos(Teta_ref(Master,:));
    Kfe(n,:) = 4*pi/lambda*dteta./tan(Teta_ref(Master,:));
end
figure, subplot(2,1,1), plot(rg_ax,Kz), grid
xlabel('slant range [m]'), title('Phase to height conversion factor [rad/m]')
ylim([-0.5 0.5])
% return
% vertical resolution
dKz_max = max(Kz(:)) - min(Kz(:));
min_pho_z = 2*pi/dKz_max

% height of ambiguity
dKz = Kz(2,:)-Kz(1,:);
z_amb = 2*pi/max(abs(dKz))
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TARGETS
scenario = 5;
switch scenario
    case 1 % one point target
        y = mean(y_ax);
        z = interp1(y_ax,z_true,y) + 20;
    case 2 % two point targets
        y = mean(y_ax)*[1 1];
        z = interp1(y_ax,z_true,y) + [0 40];
    case 3 % surface
        y = y_ax;
        z = z_true;
    case 4 % two surfaces
        y = [y_ax y_ax];
        z = [z_true z_true + 10];
    case 5
        y = zeros(100,1);
        dy_true = dy_radar/2;
        z = zeros(100,1);
        dz_true = 3;
        
        [~,ly] = min(abs(y_ref_of_r - mean(y_ref_of_r)));
        ly = ly;
        hwall = 20;
        lroof = 10;
        
        y(1) = y_ref_of_r(1);
        z(1) = z_ref_of_r(1);
        tmp = 0;
        l = 2;
        while tmp < y_ref_of_r(ly)
            y(l) = y(l-1) + dy_true;
            z(l) = z(l-1); 
            tmp = y(l);
            l = l + 1;
        end
        tmp = 0;
       while tmp < z_ref_of_r(ly) + hwall
            y(l) = y(l-1);
            z(l) = z(l-1) + dz_true;
            tmp = z(l);
            l = l + 1;
       end
       tm = 0;
       while tmp < y_ref_of_r(ly + lroof)
            y(l) = y(l-1) + dy_true;
            z(l) = z(l-1);
            tmp = y(l);
            l = l + 1;
       end
        
    otherwise
end
Np = length(y);
% Complex reflectivity
%s = randn(1,Np) + 1i*randn(1,Np);
s = ones(1,Np).*exp(1i*rand(1,Np)) + 1e-1*(randn(1,Np) + 1i*randn(1,Np));%randn(1,Np) + 1i*randn(1,Np);
%s(1:2:end) = -1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLC DATA
I = zeros(N,Nr);
Ic = zeros(N,Nr);
% figure,
for n = 1:N
    for p = 1:Np

        Rn = sqrt( (Sy(n)-y(p)).^2 + (Sz(n)-z(p)).^2);
        I(n,:) = I(n,:) + s(p)*tripuls( (rg_ax-Rn)/pho_r )*exp(-1i*4*pi/lambda*Rn);
        
        RM = sqrt( (Sy(Master)-y(p)).^2 + (Sz(Master)-z(p)).^2);
        [~,ind] = min(abs(RM-rg_ax));
        Ic(n,:) = Ic(n,:) + s(p)*tripuls( (rg_ax-RM)/pho_r )*exp(-1i*4*pi/lambda*Rn);
        %Ic(n,ind) = Ic(n,ind) + s(p)*exp(-1i*4*pi/lambda*Rn);
        
        
%         subplot(2,1,1), imagesc(abs(I)), title(sprintf('%f\%',p/Np*100)),
%         subplot(2,1,2), plot(abs(s(p)*tripuls((rg_ax-Rn)/pho_r))), title(Rn),
%         drawnow();
    end
end

%%
I0 = Ic.*exp(1i*4*pi/lambda*R_ref(:,:));

I1 = I0.*repmat(exp(-1i*angle(Ic(Master,:))),N,1);

%I1 = I1.*repmat(exp(-1i*angle(I1(:,20))),1,Nr);

I2 = I1.*exp(1i*Kfe.*repmat(rg_ax - rg_ax(20),N,1));
%I2 = I1.*exp(1i*Ky.*repmat(y_ref_of_r - y_ref_of_r(20),N,1));

Fe = 1/dr;
df = Fe/Nr;

Frange = [-Fe/2:df:Fe/2];

% for n = 1 : N
%     [~,idx_fe] = max(abs(fftshift(fft(I1(n,:)))));
%     Ffe = Frange(idx_fe);
%     I2(n,:) = I1(n,:).*exp(-1i*2*pi*Ffe*rg_ax);
% end
Ic(isnan(Ic)) = 0;


I0 = Ic.*exp(1i*4*pi/lambda*R_ref);

I1 = I0.*repmat(exp(-1i*angle(I0(Master,:))),N,1);

figure, 
subplot(3,2,1),imagesc(abs(I)), 
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Intensity')
subplot(3,2,2), imagesc(angle(I))
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Phase')
subplot(3,2,3),imagesc(abs(Ic)), 
xlabel('slant range [m]'), ylabel('flight'), title('Coregistered SLC - Intensity')
subplot(3,2,4), imagesc(angle(Ic))
xlabel('slant range [m]'), ylabel('flight'), title('Coregistered SLC - Phase')
subplot(3,2,5),imagesc(abs(I1)), 
xlabel('slant range [m]'), ylabel('flight'), title('Coregistered and Phase Flattened SLC - Intensity')
subplot(3,2,6), imagesc(angle(I1))
xlabel('slant range [m]'), ylabel('flight'), title('Coregistered and Phase Flattened SLC - Phase')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TOMOSAR
% height axis w.r.t. reference DEM
%dz = min_pho_z/8
dz = .1;
dy = dy_radar;
z_ax_rel = [-5:dz:35];
y_ax_rel = y_ref_of_r(1) : dy : y_ref_of_r(end);
tomo_sar = [];
for r = 1:Nr
    %nu =  cos(Teta_ref(Master,idx))*(y_ref_of_r(r) - y_ref_of_r(1)) + sin(Teta_ref(Master,idx))*(z_ax_rel - z_ref_of_r(1));%z_ax_rel./sin(Teta_ref(Master,idx)) + (r)./cos(Teta_ref(Master,idx));%cos(Teta_ref(Master,idx))*(rg_ax(r)) - sin(Teta_ref(Master,idx))*z_ax_rel;
    
    W = exp(1i*Kz(:,r)*z_ax_rel);% - 1i*repmat(Kfe(:,r).*(rg_ax(r) - rg_ax(20)),1,numel(z_ax_rel)))  ; % steering matrix
    
    
    %W = exp(-1i*Kz(:,r)*z_ax_rel) - 1i*repmat(Ky(:,r).*(y_ref_of_r(r) - y_ref_of_r(20)),1,numel(z_ax_rel)))  ; % steering matrix
    tomo_sar(:,r) = W'*I1(:,r);
end

figure, imagesc(y_ref_of_r,z_ax_rel, exp(log(abs(tomo_sar))/1)), axis xy
xlabel('slant range [m]'), ylabel('height [m]'), 
title('TomoSAR vertical section - SAR geometry')
%figure, imagesc(angle(tomo_sar));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% filter along x
Lx = round(2);
filter_x = hamming(2*Lx+1);

% sub-sampling along x
r_sub = Lx+1:max(round(Lx/2),1):Nr-Lx;

z_sub = linspace(-20,100,length(r_sub));

z_sub = z_ax_rel;

Nr_sub = length(r_sub);

Cov = ones(Nr_sub,N,N);
for n = 1:N
    In = I1(n,:); % n-th image
    % second-order moment
    Cnn = filter_and_sub_sample(In.*conj(In),filter_x,r_sub);
    for m = n:N
        Im = I1(m,:);
        Cmm = filter_and_sub_sample(Im.*conj(Im),filter_x,r_sub);
        Cnm = filter_and_sub_sample(Im.*conj(In),filter_x,r_sub);
        % coherence
        coe = Cnm./sqrt(Cnn.*Cmm);
        Cov(:,n,m) = coe;
        Cov(:,m,n) = conj(coe);
        %imagesc(abs(coe));
        %drawnow; pause(0.5);
    end
end

A = [];
Kz(isnan(Kz)) = 0;
Sp_est = [];    
for r = 1:Nr_sub
    A = exp(-1i*Kz(:,r_sub(r))*z_sub);
    %
    cov = squeeze(Cov(r,:,:));
    cov(isnan(cov)) = 0;    
    switch 2
      case 2, %CAPON
        cov_i = inv(cov + 1e-2*eye(N));
        Si = real(diag(A'*cov_i*A));
        Sp_est(:,r) = 1./Si;
      case 1,  % Fourier
        M =  A'*cov*A;
        Sp_est(:,r) = real(diag(A'*cov*A));
      case 3,  % MUSIC at order 2 
        [V,D]=eig(cov);
        [dum,ind]=sort(diag(D));
        Vn=V(:,ind(1:N-3));
        cov_i=Vn*Vn'+1e-2*eye(N);
        Si = real(diag(A'*cov_i*A));
        Sp_est(:,r) = 1./Si;
    end
end

Sp_est_i = interp1(rg_ax(r_sub),Sp_est',rg_ax);

figure

imagesc(rg_ax,z_sub,Sp_est_i'), axis xy
xlabel('range [m]'), ylabel('height [m]')
%hold on, plot(rg_ax,(DEM(:,a0)-DEM_avg),'k')






%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GEOCODING
% height axis
z_ax = z_ax_rel ;[-20:dz:60];
% ground range, height coordinate grids
[Za,Ya] = ndgrid(z_ax,y_ax);
% slant range position of each pixel
R = sqrt( (Sy(Master)-Ya).^2 + (Sz(Master)-Za).^2 );
% reference dem to be substracted at each pixel
Z_ref = interp1(rg_ax,z_ref_of_r,R,'linear','extrap');
% height w.r.t. reference dem
Z = Za;

R(isnan(R)) = 0;
Z(isnan(Z)) = 0;
tomo_geo = interp2(rg_ax,z_ax_rel,abs(tomo_sar),R,Z);
%tomo_geo = interp2(rg_ax,z_ax_rel,exp(log(abs(Sp_est_i'))/1e1),R,Z);

figure, imagesc(y_ax,z_ax,tomo_geo), axis xy
xlabel('slant range [m]'), ylabel('height [m]'), 
title('TomoSAR vertical section - Ground geometry')
hold on, plot(y,z,'ko','color','r','markers',1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Construction de la matrice A permettant de passer de l'image des réflectivités complexes aux images radar SLC
%  (images SLC recalées sur Master)

%A = sparse(N*Nr,Nbz*Nby); %sparse([],[],[],N*Nr,Nbz*Nby,N*Nbz*Nby);

% RM = sqrt( (Sy(Master)-y_ax(1)).^2 + (Sz(Master)-z_ax(1)).^2);
% S = tripuls( (rg_ax-RM)/pho_r );
% idx = find(S~=0);
% 
% Ai = zeros(1,numel(z_ax)*numel(y_ax)*N*numel(idx));
% Aj = zeros(1,numel(z_ax)*numel(y_ax)*N*numel(idx));
% V = zeros(1,numel(z_ax)*numel(y_ax)*N*numel(idx));
% 
% k = 1;

Ai = [];
Aj = [];
V = [];
Norma_v = [];
Normai_v = [];
k = 1;
l = 1;

im_ref = sqrt(squeeze(mean(abs(I1.^2),1)));
im_ref(isnan(im_ref)) = 0;

dz_ech = .25;
z_ech = -20:dz_ech:40;
dy_ech = dy_radar/6;
y_ech = y_ref_of_r(1):dy_ech:y_ref_of_r(end);
Nbz = numel(z_ech);
Nby = numel(y_ech);
Nbx = 1;

%figure,
for iz = 1:numel(z_ech)
    iz
    zp = z_ech(iz);
    for iy = 1:numel(y_ech)
        yp = y_ech(iy);
        
        RM = sqrt( (Sy(Master)-yp).^2 + (Sz(Master)-zp).^2);
        [~,idx] = min(abs(rg_ax - RM));
        Norma_v(l) = min(1,1/abs(im_ref(idx)));
        Normai_v(l) = max(1,abs(im_ref(idx)));

        [~,ind_p] = min((y - yp).^2 + (z - zp).^2);
        l = l + 1;
        for antenna = 1:N
            Rn = sqrt( (Sy(antenna)-yp).^2 + (Sz(antenna)-zp).^2);
            S = tripuls( (rg_ax-Rn)/pho_r )*exp(-1i*4*pi/lambda*Rn);
            %S = exp(-1i*4*pi/lambda*Rn);
            %nu = cos(Teta_ref(Master,idx))*((yp-5e3)*5e-2) - sin(Teta_ref(Master,idx))*zp;
            %S = exp(1i*Kz(antenna,idx)*zp);
            %S = S*(abs(rg_ax(idx)-RM) < .5*pho_r);
            idx = find(S~=0);
            %plot((rg_ax),tripuls( (rg_ax-RM)/pho_r )),
            %drawnow();
            %A((antenna-1)*Nr+idx,(iz-1)*numel(y_ax)+iy) = S(idx);
            Ai(k:k+numel(idx)-1) = (antenna-1)*Nr+idx;
            Aj(k:k+numel(idx)-1) = (iz-1)*numel(y_ech)+iy;
            V(k:k+numel(idx)-1) = S(idx);
            %Ai = [Ai, (antenna-1)*Nr+idx];
            %Aj = [Aj, repmat((iz-1)*numel(y_ax)+iy,1,numel(idx))];
            %V = [V , S(idx)];
            k = k+numel(idx);
        end
    end
end

Nout = numel(Norma_v(1:l-1));

A = sparse(Ai(1:k-1),Aj(1:k-1),V(1:k-1),N*Nr,Nbz*Nby);
Norma = sparse(1:Nout,1:Nout,Norma_v(1:l-1),Nout,Nout);
Norma_i = sparse(1:Nout,1:Nout,Normai_v(1:l-1),Nout,Nout);



%% construction de l'image de réflectivité "vérité terrain":
I_ground_truth = zeros(numel(z_ax),numel(y_ax));
for k=1:numel(y)
    yp = y(k);
    zp = z(k);
    sp = s(k);
    [~,iz_nearest] = min(abs(z_ax-zp));
    [~,iy_nearest] = min(abs(y_ax-yp));
    %iz_nearest
    %iy_nearest
    I_ground_truth(iz_nearest,iy_nearest) = sp;
end


figure, imagesc(y_ax,z_ax,abs(I_ground_truth)), axis xy
xlabel('slant range [m]'), ylabel('height [m]'), 
title('Ground truth')
hold on, plot(y,z,'ko','color','r','markers',1)

%%
truth = transpose(I_ground_truth);
%truth = transpose(I_ground_truth)*0;
%y = mean(y_ax);
%z = interp1(y_ax,z_true,y) + 20;
%[~,iz_nearest] = min(abs(z_ax-z));
%[~,iy_nearest] = min(abs(y_ax-y));
%truth(iy_nearest,iz_nearest) = 1;

%% simulation des mesures

Mes = A*truth(:);
%Mes = reshape(Mes,N,Nr);
Mes = reshape(Mes,Nr,N);
Mes = transpose(Mes);

figure, 
subplot(2,2,1),imagesc(abs(Ic)), 
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Intensity (ref)')
subplot(2,2,2), imagesc(angle(Ic))
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Phase (ref)')
subplot(2,2,3),imagesc(abs(Mes)), 
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Intensity (us)')
subplot(2,2,4), imagesc(angle(Mes))
xlabel('slant range [m]'), ylabel('flight'), title('SLC - Phase (us)')

%
I0_ = Mes.*exp(1i*4*pi/lambda*R_ref);
I1_ = I0_.*repmat(exp(-1i*angle(I0_(Master,:))),N,1);

%%
% TOMOSAR
% height axis w.r.t. reference DEM
%dz = min_pho_z/8
z_ax_rel = [-20:dz:60];
for r = 1:Nr
    W = exp(-1i*Kz(:,r)*z_ax_rel); % steering matrix
    tomo_sar(:,r) = W'*I1_(:,r);
end

figure, imagesc(rg_ax,z_ax_rel,abs(tomo_sar)), axis xy
xlabel('slant range [m]'), ylabel('height [m]'), 
title('TomoSAR vertical section - SAR geometry')


%% Inversion régularisée (Tikhonov)
% Ar = real(A);
% Ai = imag(A);
% %Abig = [Ar -Ai;Ai Ar];
% ind_real_3d = 1:Nbz*Nby;
% ind_imag_3d = Nbz*Nby+ind_real_3d;
% ind_real = 1:N*Nr;
% ind_imag = N*Nr+ind_real;
% Abig = @(u) [Ar*u(ind_real_3d)-Ai*u(ind_imag_3d); Ai*u(ind_real_3d)+Ar*u(ind_imag_3d)];
% Abig_transpose = @(u) [transpose(Ar)*u(ind_real)+transpose(Ai)*u(ind_imag); -transpose(Ai)*u(ind_real)+transpose(Ar)*u(ind_imag)];
% rho1 = 1;%E2;
% Hessian = @(u) Abig_transpose(Abig(u))+rho1*u;
% u0 = zeros(2*Nbz*Nby,1);
% vectorize = @(u) u(:);
% v = [vectorize(transpose(real(Ic))); vectorize(transpose(imag(Ic)))];
% %v = [vectorize(transpose(real(Mes))); vectorize(transpose(imag(Mes)))];
% b = Abig_transpose(v)+rho1*u0;
% 
% rec = pcg(Hessian,b,1E-12,1000);
% rec = rec(ind_real_3d)+1i*rec(ind_imag_3d);
% 
% reshape_data = @(v) transpose(reshape(v,Nr,N));
% reshape_unknowns = @(u) transpose(reshape(u,numel(y_ax),numel(z_ax)));
% 
% 
% Irec = reshape_unknowns(rec);%reshape(transpose(rec),numel(z_ax),numel(y_ax));
% figure, imagesc(y_ax,z_ax,abs(Irec)), axis xy
% xlabel('slant range [m]'), ylabel('height [m]'), 
% title('Regularized reconstruction')
% hold on, plot(y,z,'ko','color','r','markers',1);
% 
% figure;
% subplot(1,2,1);
% imagesc(abs(Mes));
% title('mesures');
% subplot(1,2,2);
% imagesc(reshape_data(abs(A*rec)));
% title('pseudo-mesures');



 %% Inversion régularisée: lissage quadratique sur le log du module (ADMM double splitting)
data = I;
mu_y = 0;1e1;%1E3;
mu_z = 0;1e1;%1E1;
epsilon = 1E-5;
beta1 = 1e2;%1E1;%1E6;%1E-4;
beta2 = 1e2;%1E1;%beta1/(var(log(abs(data(:)).^2+epsilon^2))/var(abs(data(:)).^2));
ADMM_iter = 50;
nb_y = numel(y_ech);
nb_z = numel(z_ech);
nb_ant = N;
nb_range = Nr;
%rec_regul = regularized_inversion_ADMM(data,mu_regul_horz,mu_regul_vert,beta1,beta2,A,ADMM_iter,epsilon,nb_y,nb_z,nb_ant,nb_range);
outer_iter = 1;
inner_iter = 20;
mu_l1 = 3;



init = zeros(Nby,Nbz,Nbx);
data_init = permute(init,[1 3 2]);
mu_x = 0;

Norma_Id = sparse(1:Nout,1:Nout,1,Nout,Nout);

%profile on
%rec_regul = regularized_inversion_ALBHO(data,mu_y,mu_z,mu_L1,beta1,beta2,A,outer_iter,inner_iter,nb_y,nb_z,nb_ant,nb_range);
rec_regul = regularized_inversion_ALBHO(data,data_init,mu_x,mu_y,mu_z,mu_l1,beta1,beta2,A,Norma_Id,Norma_i,outer_iter,inner_iter,Nbx,Nby,Nbz,N,Nr);
%profile viewer

%%
reshape_data = @(v) transpose(reshape(v,Nr,N));
reshape_unknowns = @(u) transpose(reshape(u,numel(y_ech),numel(z_ech)));
Irec = reshape_unknowns(rec_regul);%reshape(transpose(rec),numel(z_ax),numel(y_ax));
figure, 
subplot(311), imagesc(y_ech,z_ech,exp(log(abs(Irec))/1)), axis xy, colormap jet
xlabel('slant range [m]'), ylabel('height [m]'), 
title(sprintf('Regularized reconstruction with mu_{l1} = %f',mu_l1));
% subplot(312), imagesc(y_ax,z_ax,abs(Irec)), axis xy, colormap jet
% xlabel('slant range [m]'), ylabel('height [m]'), 
% title(sprintf('Regularized reconstruction with mu_{l1} = %f',mu_l1))
% hold on, plot(y,z,'ko','color','r','markers',1);
iz = [];
iy = [];
l = 1;
while l <= numel(z)
    [~,iz(l)] = min(abs(z_ech - z(l)));
    [~,iy(l)] = min(abs(y_ech - y(l)));
    l = l+1;
end

im_true = zeros(size(Irec));
im_true(sub2ind(size(Irec),iz,iy)) = abs(s);
subplot(312), imagesc(y_ech,z_ech,(im_true)), axis xy, colormap jet
xlabel('slant range [m]'), ylabel('height [m]'), 
%title(sprintf('Regularized reconstruction with mu_{l1} = %f',mu_l1))

%amp(:,nb) = abs(Irec(sub2ind(size(Irec),iz,iy)));
subplot(313),% plot(y_ech,sum(abs(Irec)>1e-3,1)./sum(abs(Irec)>1e-3,1),'x',y_ech(iy),ones(l-1,1),'.');

imagesc(y_ech,z_ech,exp(log(abs(Irec))/3)), axis xy, colormap jet
hold on
plot(y,z,'x','color','w');

%%

nb = nb + 1;

%% test de la régularisation spatiale
% Dhorz = build_Dhorz(nb_y,nb_z);
% Dvert = build_Dvert(nb_y,nb_z);
% epsilon = 1E-5;
% mu_regul = 1E2;
% Hessian_regul = @(u) u+mu_regul*((Dhorz')*(Dhorz*u)+(Dvert')*(Dvert*u));
% f = [vectorize(transpose(real(truth'))); vectorize(transpose(imag(truth')))];
% figure; subplot(1,2,1); imagesc(reshape_unknowns(abs(f(ind_real_3d)+1i*f(ind_imag_3d)))); colorbar
% conj_grad_tol = 1E-14;
% max_conj_grad_iter = 1000;
% b = (log(f(ind_real_3d).^2+f(ind_imag_3d).^2+epsilon^2));
% wopt = pcg(Hessian_regul,b,conj_grad_tol,max_conj_grad_iter);
% subplot(1,2,2); imagesc(reshape_unknowns(wopt)); colorbar;


%%

