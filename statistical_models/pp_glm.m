%% Point Process-GLM Model

close all;clear;clc; 
%%  Load data 

% Change this for your machine and data 

lin_pos = load('Data/3-2-7-4/linear_position.mat');
spike = load('Data/3-2-7-4/spike.mat');
direction = load('Data/3-2-7-4/direction.mat');
speed = load('Data/3-2-7-4/speed.mat');
time = load('Data/3-2-7-4/time.mat');
arm = load('Data/3-2-7-4/arm_name.mat');
phi = load('Data/3-2-7-4/phase.mat');

% Data preprocessing
time = time.struct.time;arm = arm.struct.arm_name';
lin_pos = lin_pos.struct.linear_position2';
speed = speed.struct.speed';

direction = direction.struct.head_direction';
spike = spike.struct.is_spike';
phi = phi';

lin_pos(isnan(lin_pos))=0;speed(isnan(speed))=0;arm(isnan(arm))=0;
direction(isnan(direction))=0;spike = double(spike);
%% Null Model
mtx_nll = ones(size(spike,1),1);
[b_n,dev_n,stat_n] = glmfit(mtx_nll,spike,'poisson','constant','off');
yhat_nl = exp(b_n.*mtx_nll);
%% History Component 

% Basis function (Spline) parameters
c_pt = 0:20:200; % Control points
s = 0.4; % Tension parameter
lag = 200; 

% Construct spline matrix
HistSpl = ModifiedCardinalSpline(lag,c_pt,s);

% Design matrix for multiplicative history model
Hist = zeros(length(spike)-lag,lag);
for i=1:lag
   Hist(:,i) = spike(lag-i+1:end-i);
end
y = spike(lag+1:end);
mtx_hist = Hist*HistSpl;

% GLM fit 
[b_hist ,dev_hist, stat_hist] = glmfit(mtx_hist,y,'poisson');
[yhat_hist1,ylo_hist1,yhi_hist1] = glmval(b_hist,HistSpl,'log',stat_hist);
%% Linearized Position

% Divide linear position for center, right, and left arms in the W-track
pos_c=lin_pos(arm==0);pos_r=lin_pos(arm==1);pos_l=lin_pos(arm==-1);
arm_c=(arm==0); arm_r=(arm==1); arm_l=(arm==-1);

% Spline parameters 
c_pt_posc=[0,10,30,50,max(pos_c)];
c_pt_posr=[min(pos_r(pos_r~=0)),133,max(pos_r)+2];
c_pt_posl=[min(pos_l(pos_l~=0)),198,max(pos_l)+2];
num_c_ptsc = length(c_pt_posc);num_c_ptsr = length(c_pt_posr);
num_c_ptsl = length(c_pt_posl);s = 0.4; 

idx_c=find(arm==0);idx_r=find(arm==1);idx_l=find(arm==-1);

% Design matrix for lin-positon
mtx_posc = zeros(size(lin_pos,1),num_c_ptsc);
mtx_posc(idx_c,:) = ModifiedCardinalSpline_pos(lin_pos(arm==0),c_pt_posc,s);

mtx_posr = zeros(size(lin_pos,1),num_c_ptsr);
mtx_posr(idx_r,:) = ModifiedCardinalSpline_pos(lin_pos(arm==1),c_pt_posr,s);

mtx_posl = zeros(size(lin_pos,1),num_c_ptsl);
mtx_posl(idx_l,:) = ModifiedCardinalSpline_pos(lin_pos(arm==-1),c_pt_posl,s);

mtx_pos = [mtx_posc ,mtx_posl , mtx_posr];

% GLM fit
[b_pos,dev_pos,stat_pos] = glmfit(mtx_pos,spike,'poisson');
[yhat_pos,ylo_pos,yhi_pos] = glmval(b_pos,mtx_pos,'log',stat_pos);
%% Lin-Position, Speed

% Design matrix for lin-positon and speed
thresholdSpd = 2;idx = find(speed>=thresholdSpd); % Setting threshold for speed
mtx_posspd = mtx_pos(idx,:); 

% GLM fit 
[b_posspd,dev_posspd,stat_posspd] = glmfit(mtx_posspd,spike(idx),'poisson');
[yhat_posspd,ylo_posspd,yhi_posspd] = glmval(b_posspd,mtx_posspd,'log',stat_posspd);
%% Lin-Position, Direction

% Defining direction on the linearized track 
direction = diff(lin_pos);
direction = [direction<0 , direction>=0];

spikee = spike(2:end);
mtx_posdir = [mtx_pos(2:end,:).*direction(:,1),mtx_pos(2:end,:).*direction(:,2)];

% GLM Fit
[b_posdir,dev_posdir,stat_posdir] = glmfit(mtx_posdir,spikee,'poisson');
[yhat_posdir,ylo_posdir,yhi_posdir] = glmval(b_posdir,mtx_posdir,'log',stat_posdir);
%% Lin-Position,Phase 

% Indicator fundtion on theta phase 
ind_phase = [];numIndicators=5;
phi_prt = linspace(min(phi),max(phi)+0.01,numIndicators);
for i=1:numIndicators-1
   ind_phase(:,i) = [phi>=phi_prt(i) & phi<phi_prt(i+1)];   
end

% Design matrix for lin-pos and theta precession 
mtx_posphi = [];
for i=1:numIndicators-1
   mtx_posphi = [mtx_posphi,mtx_pos.*ind_phase(:,i)];
end

% GLM fit 
[b_posphi,dev_posphi,stat_posphi] = glmfit(mtx_posphi,spike,'poisson');
[yhat_posphi,ylo_posphi,yhi_posphi] = glmval(b_posphi,mtx_posphi,'log',stat_posphi);
%% Lin-Position, Phase, Speed

% Design matrix for lin-pos, theta precession, and speed
thresholdSpd = 2; idx = find(speed>=thresholdSpd); % Setting threshold for speed
mtx_posphispd = mtx_posphi(idx,:);

% GLM fit
[b_posphispd,dev_posphispd,stat_posphispd] = glmfit(mtx_posphispd,spike(idx),'poisson');
[yhat_posphispd,ylo_posphispd,yhi_posphispd] = glmval(b_posphispd,mtx_posphispd,'log',stat_posphispd);
%% Lin-Position, Phase, Speed, Direction

% Defining direction on the linearized track 
direction = diff(lin_pos);
direction = [direction<0 , direction>=0];

spd = speed(2:end); 
spikee = spike(2:end); 

idxx = find(spd>thresholdSpd);
dir = direction(idxx,:);

% Design matrix for lin-pos, theta precession, speed, and direction
mtx_posphispddir = [mtx_posphispd.*dir(:,1),mtx_posphispd.*dir(:,2)];

% GLM fit
[b_posphispddir,dev_posphispddir,stat_posphispddir] = glmfit(mtx_posphispddir,spikee(idxx),'poisson');
[yhat_posphispddir,ylo_posphispddir,yhi_posphispddir] = glmval(b_posphispddir,mtx_posphispddir,'log',stat_posphispddir);
%% Full: Lin-Position, Phase, Speed, Direction, History


spdd = speed(lag+2:end);
spk = spike(lag+2:end);

% Design matrix for the full model
mtx_posphidir = [mtx_posphi(2:end,:).*direction(:,1),mtx_posphi(2:end,:).*direction(:,2)];
mtx_posphidirhist = [mtx_posphidir(lag+1:end,:),mtx_hist(2:end,:)];
mtx_full = mtx_posphidirhist(spdd>=thresholdSpd,:);

% GLM fit
[b_full,dev_full,stat_full] = glmfit(mtx_full,spk(spdd>=thresholdSpd),'poisson');
[yhat_full,ylo_full,yhi_full] = glmval(b_full,mtx_full,'log',stat_full);
%% Chi-Squre Test
 
chi_h = dev_n-dev_hist; p_hist = 1-chi2cdf(chi_h,length(b_hist)-1);
chi_p = dev_n-dev_pos; p_pos = 1-chi2cdf(chi_p,length(b_pos)-1);
chi_pd = dev_pos-dev_posdir; p_posdir = 1-chi2cdf(chi_pd,length(b_posdir)-length(b_pos));
chi_pp = dev_pos-dev_posphi; p_posphi = 1-chi2cdf(chi_pp,length(b_posphi)-length(b_pos));
chi_ps = dev_pos-dev_posspd; p_posspd = 1-chi2cdf(chi_ps,length(b_posspd)-length(b_pos));