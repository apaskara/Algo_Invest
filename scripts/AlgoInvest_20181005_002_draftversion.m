%% Algo_Invest Script File
% Authors: AB Paskaramoorthy
% Version: AlgoInvest-20170904-004aa

% # Problem Specification:  ...
% # Data Specification: EPC_HML_SMB_J203_FMP_workspace_R2007a
% # Configuration Control:
%        userpath/MATLAB/Algo_Invest
%        userpath/MATLAB/Algo_Invest/data       
%        userpath/MATLAB/Algo_Invest/scripts     
%        userpath/MATLAB/Algo_Invest/functions   
%        userpath/MATLAB/Algo_Invest/html        
%        userpath/MATLAB/Algo_Invest/test_code
%        userpath/MATLAB/Algo_Invest/junk
%        userpath/MATLAB/Algo_Invest/workspace   
% # Version Control: ... (SVN/Tortoise SVN/CVS)
% # References: ...
% # Current Situation: 
%   Split into initialization and update sections
%   Split components into different scripts
%   took out the index application to the betas
%
%   
%                       
% # Future Situation: 
% EWMA characteristic payoffs
%
% Uses: 
%       
%% Notes:
% 1. FIXME -       
% 2. TBDL -          
%       1. Low pass filter on raw data to smooth out spikes (and errors)
%       2. Definition and Application of Entropy Measures
%       3. Robust transformation on loss functions
%       4. Can move FMP construction to data prep - alternatively, create
%       FMP function
%       5. Check forecast accuracy corresponds to loss portfolio performance
%       6. Beta and R_0 intialisation 
%       
% 4. Use a startup.m file where possible to manage your paths/java libraries


%% 1. Data Description
% fts
% 1 DY          -   dividend yield
% 2 MV          -   market capitalisation (public float?)
% 3 NT          -   ??
% 4 PE          -   price-to-earnings ratio (earnings reporting frequency?)
% 5 PTBV        -   price to book value (book value reporting frequency?)
% 6 RI          -   monthly prices of each ticker
% 7 VO          -   volume traded
% rfr           - risk-free rate proxy - (90-day NCD)
            
%% 3. Clear workspace
close all;
clear;
clc;
warning off;

%% 4. Set Paths (implement configuration control)
% Specify the path directory containing the data files
% addpath c:\Users\user\Documents\MATLAB
% use the userpathstr.
%
%  # Project Path:              userpath/MATLAB/Algo_Invest
%  # Data Path:                 userpath/MATLAB/Algo_Invest/data       
%  # Script Files:              userpath/MATLAB/Algo_Invest/scripts     
%  # Classes and Function:      userpath/MATLAB/Algo_Invest/functions
%  # Test Code:                 userpath/MATLAB/Algo_Invest/test_code
%  # Published Output:          userpath/MATLAB/Algo_Invest/html        
%  # Legacy Code:               ...
%

userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
% Project Paths:
% -- Modify this line to be your prefered project path ----->
projectpath = 'Algo_Invest';
% <----------------------------------------------------------
addpath(fullfile(userpathstr,projectpath,'data'));
addpath(fullfile(userpathstr,projectpath,'functions'));
addpath(fullfile(userpathstr,projectpath,'scripts'));
addpath(fullfile(userpathstr,projectpath,'test_code'));
addpath(fullfile(userpathstr,projectpath,'html'));
addpath(fullfile(userpathstr,projectpath,'plots'));
addpath(fullfile(userpathstr,projectpath,'workspace'));

%% 1. Load Data
% prep data in Data_prep_002aa

% load data
load('procdata1_w');

% indexes
indexes     = {'top40', 'top100', 'top160'};
nindex      = [40 100 160];

% choose index
% for ii = 1:numel(indexes)
% uni=indexes{ii};
uni = 'top160';
findex = db_procdata.indexes.(uni);
index = double(fts2mat(db_procdata.indexes.(uni)));
index(index==0) = NaN;

%% split data into initialization and update
% end of in-sample period
insample = datenum('28-Dec-2007');
insample = find(findex.dates==insample);

% split into initialisation and update (one-third / two-third)
ini = [1, round(insample/3) ];
updt = [round(insample/3) + 1, insample];

%% Sort companies by characteristics into quantiles
% index has not been lagged (findex has)
% characteristics have been lagged by 4 weeks

% we lag variables from published financial records.
clear nb nm nm3;
% create temp local characteristic variables
mv = lagts(db_procdata.char_data{2},4); %mv
bvtp = lagts(db_procdata.char_data{4},4); %bvtp

% quintile sorts - value
nb = ntile(transpose((index .* fts2mat(bvtp))),3,'descend'); 
db_quintile.sorts.nb = fints(bvtp.dates,transpose(nb),fieldnames(bvtp,1),bvtp.freq,bvtp.desc);
% quintile sort - size - 2 quintiles
nm = ntile(transpose(index .* fts2mat(mv)),2,'descend'); 
db_quintile.sorts.nm = fints(mv.dates,transpose(nm),fieldnames(mv,1),mv.freq,mv.desc);
% quintile sort - size - 3 quintiles
nm3 = ntile(transpose(index .* fts2mat(mv)),3,'descend');
db_quintile.sorts.nm3 = fints(mv.dates,transpose(nm3),fieldnames(mv,1),mv.freq,mv.desc);
% delete temp variables
clear mv bvtp;
clear nb nm nm3; 


%% Create Fama-French Factor portfolios
% lag portfolios to apply return
% eg. at time t, intersection portfolio return: HS(t-1)*ret(t)

nbx = fts2mat(db_quintile.sorts.nb);
nmx = fts2mat(db_quintile.sorts.nm);
% nm3x = fts2mat(db_quintile.sorts.nm3);
% create the index for the different bins
small  = (nmx==2);
big    = (nmx==1);
high   = (nbx==1);
medium = (nbx==2);
low    = (nbx==3);
% create LS (low small) intersection of nbvtp and nmv equally weighted
db_quintile.intersection.LS = low & small;
db_quintile.intersection.MS = medium & small;
db_quintile.intersection.HS = high & small;
db_quintile.intersection.LB = low & big;
db_quintile.intersection.MB = medium & big;
db_quintile.intersection.HB = high & big;

% Compute the normalizations
db_quintile.norms.normLS = transpose(sum(transpose(db_quintile.intersection.LS)));
db_quintile.norms.normMS = transpose(sum(transpose(db_quintile.intersection.MS)));
db_quintile.norms.normHS = transpose(sum(transpose(db_quintile.intersection.HS)));
db_quintile.norms.normLB = transpose(sum(transpose(db_quintile.intersection.LB)));
db_quintile.norms.normMB = transpose(sum(transpose(db_quintile.intersection.MB)));
db_quintile.norms.normHB = transpose(sum(transpose(db_quintile.intersection.HB)));

% clear local variables
clear nbx nmx nm3x small big high medium low;
 
%% Create the indices for the intersection portfolios
% retrieve stock returns
ret = db_procdata.returns.ret;
retx = fts2mat(db_procdata.returns.ret);

%create temp vars
tmp_inputs1 = {'LS', 'MS', 'HS', 'LB', 'MB', 'HB'};
[LS, MS, HS, LB, MB, HB] = create_vars(db_quintile.intersection, tmp_inputs1);
tmp_inputs2 = {'normLS', 'normMS', 'normHS', 'normLB', 'normMB', 'normHB'};
[normLS, normMS, normHS, normLB, normMB, normHB] = create_vars(db_quintile.norms, tmp_inputs2);

% compute the returns - equal  weighting
db_quintile.intersection.rLS = transpose(nansum(transpose(retx .* LS))) ./ normLS; 
db_quintile.intersection.rMS = transpose(nansum(transpose(retx .* MS))) ./ normMS;
db_quintile.intersection.rHS = transpose(nansum(transpose(retx .* HS))) ./ normHS;
db_quintile.intersection.rLB = transpose(nansum(transpose(retx .* LB))) ./ normLB;
db_quintile.intersection.rMB = transpose(nansum(transpose(retx .* MB))) ./ normMB;
db_quintile.intersection.rHB = transpose(nansum(transpose(retx .* HB))) ./ normHB;
form_returns = [db_quintile.intersection.rLS, db_quintile.intersection.rMS, ...
    db_quintile.intersection.rHS, db_quintile.intersection.rLB, ... 
    db_quintile.intersection.rMB, db_quintile.intersection.rHB];
db_quintile.intersection.retfts = fints(ret.dates,form_returns,{'LS','MS','HS','LB','MB','HB'},ret.freq,'Returns');

% clear temp variables
clear retx form_returns ret; 
clear tmp_inputs1 LS MS HS LB MB HB;
clear tmp_inputs2 normLS normMS normHS normLB normMB normHB;


%% Create Fama-French Factor Replicating Portfolios

% local temp variables
tmp_inputs = {'rHB','rHS','rMB','rMS','rLB','rLS'};
[rHB, rHS, rMB, rMS, rLB, rLS] = create_vars(db_quintile.intersection, tmp_inputs);
% Create the quarterly sorts of H(igh)-M(inus)-L(ow) Book-to-Market
rHML = (1/2)*((rHB + rHS) - (rLB + rLS));
% Create the quarterly sorts of S(mall)-M(inus)-B(ig) on Log-size
rSMB = (1/3)*((rHS + rMS + rLS) - (rHB + rMB +rLB));
% Create a market index (market cap weighted - balanced portfolio)
mvx = exp(index .* fts2mat(lagts(db_procdata.char_data{2})));
retx = fts2mat(db_procdata.returns.ret);
normMV = transpose(nansum(transpose(mvx)));
% normND = min(nansum(index,2), nansum(~isnan(retx),2)); % min of index or populated returns
normND = nansum(~isnan(index.*retx),2);

% rMKT = (transpose(nansum(transpose(retx .* mvx))) ./ normMV) - fts2mat(db_procdata.returns.rfr);
rMKT = (transpose(nansum(transpose(retx .* mvx))) ./ normMV) - fts2mat(db_procdata.returns.rfr);
% naive diversification
rIND = nansum(index.*retx,2)./normND;
% naive weights
ndw = index ./ normND;
% naive turonver
ndTO = turnover2(ndw, retx);
% net return assuming 50 basis point proportional transaction costs
c = 0.005;
netIND = rIND + log(1-c*ndTO);

% cap weighted
rCAP = (transpose(nansum(transpose(retx .* mvx))) ./ normMV);
% cap weights
capw = mvx ./ normMV;
% cap weights turnover - will be zero if we consider whole market, but
% there are rebalancing costs associated with change in the index
% constituents
capTO = turnover2(capw, retx);
% net return
netCAP = rCAP + log(1-c*capTO);

% store returns struct
db_factors.returns.rHML = rHML;
db_factors.returns.rSMB = rSMB;
db_factors.returns.rMKT = rMKT;
db_factors.returns.rIND = rIND;
db_factors.returns.rCAP = rCAP;
db_factors.returns.rIND = rIND;
db_factors.returns.netIND = netIND;
db_factors.returns.netCAP = netCAP;

% clear vars
clear mvx retx normMV;

% risk-free rate
rfr = fts2mat(db_procdata.returns.rfr);

% Store factor returns in separate fints structure
fmpfts = fints(db_procdata.returns.ret.dates,[rHML, rSMB, rMKT],{'HML','SMB','MKT'},db_procdata.returns.ret.freq,'Returns');
indpft = fints(db_procdata.returns.ret.dates,[rIND,rCAP, netIND, netCAP, rfr ],{'ND','CAP', 'netND', 'netCAP', 'rfr'},db_procdata.returns.ret.freq,'Benchmark Portfolio');
prcfmp = fillts(fmpfts(12:end),0);
prcfmp(1) = 0;
prcfmp = exp(cumsum(prcfmp));

prcind = fillts(indpft(12:end),0);
prcind(1) = 0;
prcind = exp(cumsum(prcind));

% store vars
db_factors.returns.fmpfts = fmpfts;
db_factors.returns.prcfmp = prcfmp;
db_factors.returns.indpft = indpft;
db_factors.returns.prcind = prcind;
% store turnover
db_factors.returns.ndTO =  fints(db_procdata.returns.ret.dates, ndTO ,{'ND'},db_procdata.returns.ret.freq,'Turnover');
db_factors.returns.capTO = fints(db_procdata.returns.ret.dates, capTO ,{'CAP'},db_procdata.returns.ret.freq,'Turnover');

%clear variables
clear tmp_inputs rHB rHS rMB rMS rLB rLS rHML rSMB rMKT fmpfts prcfmp;
clear capTO capw ndTO ndw netCAP netIND normND rCAP rfr rIND prcind indpft;

%% Specify Factor Models
mdl_type = 'ol';
db_mdlindx.(mdl_type){1,1} = {[3]}; 
db_mdlindx.(mdl_type){1,2} = {'MKT'};
db_mdlindx.(mdl_type){2,1} = {[1 2 3]};
db_mdlindx.(mdl_type){2,2} = {'HML','SMB','MKT'};
clear mdl_type;


%% Choose Lambda for factor models

% iniisation is done by estimating factor models and taking mean \
% median of betas at final date of iniisation
% 0.91 is best lambda

% Initialisation - choose lambda
% calculate MSE for different lambdas 
%{
% forgetting factor
lambda = 0.88:0.01:1;
% outputs and inputs
ydata = fts2mat(db_procdata.returns.eret(ini(1): ini(2)));
fdata = fts2mat(db_factors.returns.fmpfts(ini(1): ini(2)));
% fit for different lambdas
fit = NaN*ones(length(lambda),3);
% factor index
fct_ind = [1 2 3];

for i = 1:length(lambda)
    model = online_ts_reg(fdata(:,fct_ind),ydata, 'y', {'lambda',lambda(i)}, {'weight', 'huber'});
    fit(i,1) = lambda(i);
    fit(i,2) = model.MSE;
    fit(i,3) = model.MFE;
end
save('FMfit_lambda', 'fit' );

% clear fit model ydata fdata lamba fct_ind;

%}

%% Initialise Factor models

% estimate factor models 
% forgetting factor
lambda = 0.98;
% outputs and inputs (in-sample)
ydata = fts2mat(db_procdata.returns.eret(ini(1): ini(2)));
fdata = fts2mat(db_factors.returns.fmpfts(ini(1): ini(2)));
% define mdl_type
mdl_type = 'ol';
% choose weight function
% wfunc = {'huber', 'std'};
wfunc = {'huber'};
% models definedeted by factor selection - see db_mdlindx.(mdl_type)
for k = 1:size(db_mdlindx.(mdl_type),1)
% for k = 2
    fct_ind = cell2mat(db_mdlindx.(mdl_type){k,1});    % select factors
%     db_models.rw{1,k} = rolling_ts_reg(fdata(:,fct_ind),ydata,mdof); %rolling
    for j = 1:length(wfunc)
        db_models.ol{j,k} = online_ts_reg(fdata(:,fct_ind),ydata, 'y', {'lambda',lambda}, {'weight', wfunc{j}}); %online
    end
end

clear fct_ind ydata fdata k mdl_type wfunc;

%% store beta in fints to keep track of dates
% set temp vars
ret = db_procdata.returns.ret(ini(1): ini(2)) ;
% choose model type
mdl_type = 'ol'; % rw - rolling window; ol - online
all_factors = {'bBIAS', 'bHML', 'bSMB', 'bMKT'};
% index over iniisation period
Iindex = index(ini(1): ini(2),:);
% Iindex = ones(size(index(ini(1): ini(2),:))); single indexing

for k = 1:size(db_mdlindx.(mdl_type),1)
% for k = 2
    clear fct_ind;
    fct_ind = cell2mat(db_mdlindx.(mdl_type){k,1}); % choose factors
    fct_ind = [1 (fct_ind + 1)]; % include bias
    for i = 1:length(fct_ind) % for each factor
        clear fname;
        fname = all_factors{fct_ind(i)}; %fints title
        for j = 1:size(db_models.(mdl_type),1) % include in model index?
            db_models.(mdl_type){j,k}.beta(:,:,i) = db_models.(mdl_type){j,k}.beta(:,:,i) ;
            beta = squeeze(db_models.(mdl_type){j,k}.beta(:,:,i)); %beta matrix
            db_models.(mdl_type){j,k}.fbeta{i} = fints(ret.dates, beta, fieldnames(ret,1), ret.freq, fname); %store as fints 
            % mean beta
            db_models.(mdl_type){j,k}.b_mean(i) = nanmean(Iindex(end,:) .* beta(end,:));
            % median beta 
            db_models.(mdl_type){j,k}.b_median(i) = nanmedian(Iindex(end,:) .* beta(end,:));
        end
    end
end
% clear vars
clear ret all_factors fct_ind fname beta k i;

%% store betas for cross-sectional regressions


% online CAPM beta
db_procdata.beta_data{1} = db_models.ol{1}.fbeta{2};
db_procdata.beta_index{1} = 'online CAPM MKT beta';
% online FF3F HML beta
db_procdata.beta_data{2} = db_models.ol{2}.fbeta{2};
db_procdata.beta_index{2} = 'online FF3F HML beta';
% online FF3F SMB beta
db_procdata.beta_data{3} = db_models.ol{2}.fbeta{3};
db_procdata.beta_index{3} = 'online FF3F SMB beta';
% online FF3F MKT beta
db_procdata.beta_data{4} = db_models.ol{2}.fbeta{4};
db_procdata.beta_index{4} = 'online FF3F MKT beta';
% online FF3F MKT alpha 
db_procdata.beta_data{5} = db_models.ol{2}.fbeta{1};
db_procdata.beta_index{5} = 'online FF3F alpha';

% save to disk
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas');
ol_capm_beta = db_models.ol{1}.fbeta{2};
ol_ff_alpha = db_models.ol{2}.fbeta{1};
save(fullfile(filepath,strcat('ol_capm_beta_',uni)), 'ol_capm_beta');
save(fullfile(filepath, strcat('ol_ff_alpha_',uni)), 'ol_ff_alpha');

clear filepath ol_capm_beta ol_ff_alpha;

%% Factor Realisations means and covariances

% lambda
lambda = 0.99;
% initial means and covariance
fretx = fts2mat(db_factors.returns.fmpfts(ini(1):ini(2)));
[fcov0,~,fmu0] = ewma_cov(lambda,fretx, {'mu0',  nanmean(fretx)}, {'COV_0', nancov(fretx)});

% initial R matrix (hessian matrix (search direction?))
for i = 1:size(db_mdlindx.ol,1)
    clear fct_ind inputs;
    fct_ind = cell2mat(db_mdlindx.ol{i,1});
    inputs = [ones(size(fretx,1),1) fretx(:, fct_ind)];
    R0{i} = ewma_autocorr(lambda, inputs, {'COV_0', inputs'*inputs});
end
clear fretx fct_ind inputs;

%% Initialisation Systematic Model Forecasts (expected returns only)

% factor model initial forecast
lambda = 0.98;
% Uindex = index(ini(1):ini(2),:);
ret = db_procdata.returns.eret(ini(1):ini(2));
rfr = db_procdata.returns.rfr(ini(1):ini(2));
mdl_type = 'ol';
% index over initialisation period
Iindex = index(ini(1): ini(2),:);
% Iindex = ones(size(index(ini(1): ini(2),:))); single indexing


for i = 2 %% FF3F
    % factor selection
    fct_ind = cell2mat(db_mdlindx.ol{i,1});
    for j = 1:size(db_models.ol,1)
        % stock factor beta 
        beta = db_models.ol{j,i}.beta;
        % pre-allocation
        exp_ret = NaN*ones(size(beta,1)+1,size(beta,2));    
        sys_error = NaN*ones(size(beta,1)+1,size(beta,2));  
        % factor premiums
        fact_prem = fts2mat(db_factors.returns.fmpfts(ini(1):ini(2)+1));
        fact_prem = fact_prem(:,fct_ind); % select factors
        [fcov,~,fmu] = ewma_cov(0.99, fact_prem);
        % for each time period
        for t = 1:size(beta,1)
            % time t index
            index_t = Iindex(t,:);              
            % time t beta - exclude intercept for factor expected return
            beta_t = reshape(squeeze(beta(t,:,2:end)), size(index_t,2),[]); 
            % expected return for t+1
            exp_ret(t+1,:) = (beta_t * fmu(t,:)')' + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
            % factor innovation at t+1 (not sure about the index_t, just copying from previous line) - also fact_prem t+1 and f_mu are approximations
            sys_error(t+1,:) = (beta_t * (fact_prem(t+1,:) - fmu(t,:))')';
        end
        % store as fints
        db_models.(mdl_type){j,i}.fexp = lagts(fints(ret.dates, exp_ret(2:end,:), fieldnames(ret,1), ret.freq),1,NaN) + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
        db_models.(mdl_type){j,i}.sys_error = fints(ret.dates, sys_error(1:end-1,:), fieldnames(ret,1), ret.freq);      
    end
end

clear t fct_ind fact_prem ret mdl_type beta beta_t exp_ret index_t;

%% Factor Model Update Phase

% forgetting factor
lambda = 0.98;
% outputs and inputs
% ydata = fts2mat(db_procdata.returns.eret(updt(1):updt(2)));
ydata = fts2mat(db_procdata.returns.eret(updt(1):end));
% fdata = fts2mat(db_factors.returns.fmpfts(updt(1):updt(2)));
fdata = fts2mat(db_factors.returns.fmpfts(updt(1):end));
% choose weight function
wfunc = {'huber'};
% models defined by factor selection - see db_mdlindx.(mdl_type)
for k = 1:size(db_mdlindx.ol,1)
    fct_ind = cell2mat(db_mdlindx.ol{k,1}); 
    for j = 1:length(wfunc)
        % betas from initialisation
        B0 = squeeze(db_models.ol{j,k}.beta(end,:,:))';
        % find where no betas
        bnan = isnan(B0);
        % use median beta
        Bmed = repmat(db_models.ol{j,k}.b_median',1, size(B0,2)) ;
        % intial beta
        B0(bnan) = Bmed(bnan);
        % estimate model
        db_models.olu{j,k} = online_ts_reg(fdata(:,fct_ind), ydata, 'y', {'lambda',lambda}, {'weight', wfunc{j}}, {'B0',B0 }, {'invR0', inv(R0{k}) }    ); %online
    end
end

clear fct_ind ydata fdata k mdl_type wfunc bnan Bmed B0;

%% store beta in fints to keep track of dates 
% set temp vars
% ret = db_procdata.returns.eret(updt(1): updt(2)) ;
ret = db_procdata.returns.eret(updt(1): end) ;
% choose model type
mdl_type = 'olu'; % rw - rolling window; ol - online
all_factors = {'bBIAS', 'bHML', 'bSMB', 'bMKT'};
% index over update period
% Uindex = index(updt(1): updt(2),:);

for k = 1:size(db_mdlindx.ol,1)
% for k = 2
    clear fct_ind;
    fct_ind = cell2mat(db_mdlindx.ol{k,1}); % choose factors
    fct_ind = [1 (fct_ind + 1)]; % include bias
    for i = 1:length(fct_ind) % for each factor
        clear fname;
        fname = all_factors{fct_ind(i)}; %fints title
        for j = 1:size(db_models.(mdl_type),1) % include in model index?
            db_models.(mdl_type){j,k}.beta(:,:,i) = db_models.(mdl_type){j,k}.beta(:,:,i) ; % set betas not in unverse to NaN
            beta = squeeze(db_models.(mdl_type){j,k}.beta(:,:,i)); % beta matrix for factor i 
            db_models.(mdl_type){j,k}.fbeta{i} = fints(ret.dates, beta, fieldnames(ret,1), ret.freq, fname); %store as fints 
            % mean beta of stocks in i
            db_models.(mdl_type){j,k}.b_mean(:,i) = nanmean(beta,2);
            % median beta 
            db_models.(mdl_type){j,k}.b_median(:,i) = nanmedian(beta,2);
        end
    end
end
% clear vars
clear ret all_factors fct_ind fname  k i;

%% store for characteristics models

% online CAPM beta
db_procdata.beta_data{1} = db_models.olu{1}.fbeta{2};
db_procdata.beta_index{1} = 'online CAPM MKT beta';
% online FF3F HML beta
db_procdata.beta_data{2} = db_models.olu{2}.fbeta{2};
db_procdata.beta_index{2} = 'online FF3F HML beta';
% online FF3F SMB beta
db_procdata.beta_data{3} = db_models.olu{2}.fbeta{3};
db_procdata.beta_index{3} = 'online FF3F SMB beta';
% online FF3F MKT beta
db_procdata.beta_data{4} = db_models.olu{2}.fbeta{4};
db_procdata.beta_index{4} = 'online FF3F MKT beta';
% online FF3F MKT alpha 
db_procdata.beta_data{5} = db_models.olu{2}.fbeta{1};
db_procdata.beta_index{5} = 'online FF3F alpha';

% save to disk
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas');
olu_capm_beta = db_models.olu{1}.fbeta{2};
olu_ff_alpha = db_models.olu{2}.fbeta{1};
save(fullfile(filepath,strcat('olu_capm_beta_',uni)), 'olu_capm_beta');
save(fullfile(filepath, strcat('olu_ff_alpha_',uni)), 'olu_ff_alpha');

clear olu_capm_beta olu_ff_alpha;

%% Factor Models expected return forecast 
% applied time t index here - check if time t+1 index is applied at
% portfolio construction stage


flambda = 0.99; % lambda for factor realisations
% Uindex = index(updt(1):updt(2),:);
Uindex = index(updt(1):end,:);
% ret = db_procdata.returns.ret(updt(1):updt(2));
ret = db_procdata.returns.ret(updt(1):end);
% rfr = db_procdata.returns.rfr(updt(1):updt(2));
rfr = db_procdata.returns.rfr(updt(1):end);
mdl_type = 'olu';

for i = 2 % FF3F
    clear sys_error;
    % factor selection
    fct_ind = cell2mat(db_mdlindx.ol{i,1});
    % factor premiums
%     fact_prem = fts2mat(db_factors.returns.fmpfts(updt(1):updt(2)+1));
    fact_prem = fts2mat(db_factors.returns.fmpfts(updt(1):end));
    fact_prem = fact_prem(:,fct_ind); % select factors
    [fcov,fact_prem_cov,fmu] = ewma_cov(flambda, fact_prem, {'mu0',  fmu0(end,:)}, {'COV_0', fcov0});
    % for each weight function (eachg filter type)
    for j = 1:size(db_models.ol,1)
        clear temp tempRet
        % stock factor beta 
        beta = db_models.olu{j,i}.beta;
        % pre-allocation
        exp_ret = NaN*ones(size(beta,1)+1,size(beta,2));        
        % for each time period
        for t = 1:size(beta,1)
            % time t index
            index_t = Uindex(t,:);  
            % NaN to zero (needed to form index matrix)
            index_t(isnan(index_t)) = 0;
            % covariance index at t
            %{
            INDEX_t = index_t'*index_t;
            INDEX_t(INDEX_t==0) = NaN;
            index_t(index_t==0) = NaN;
            %}
            % time t beta - exclude intercept for factor expected return
            beta_t = reshape(squeeze(beta(t,:,2:end)), size(index_t,2),[]); 
            % expected return for t+1
            exp_ret(t+1,:) = (beta_t * fmu(t,:)')' + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
            % systematic risk innovation at t+1
%             sys_error(t,:) = index_t.*(beta_t * (fact_prem(t,:) - fmu(t,:))')';
            % risk model for t+1
%             db_models.(mdl_type){j,i}.risk{t+1} = INDEX_t.*(beta_t(:,:)*fact_prem_cov{t}*beta_t(:,:)'); % exclude bias term
            db_models.(mdl_type){j,i}.risk{t+1} = (beta_t(:,:)*fact_prem_cov{t}*beta_t(:,:)'); % exclude bias term
        end
        % store as fints
        db_models.(mdl_type){j,i}.fexp = lagts(fints(ret.dates, exp_ret(2:end,:), fieldnames(ret,1), ret.freq),1,NaN) + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
%         db_models.(mdl_type){j,i}.sys_error = fints(ret.dates, sys_error(1:end,:), fieldnames(ret,1), ret.freq);
        
        % remove latest risk model so that size / dates coincide with fexp
        db_models.(mdl_type){j,i}.risk(end) = [];        
        % install first risk model from intialisation data
        
        % save risk model to disk
        SysRisk = db_models.(mdl_type){j,i}.risk; 
        filepath = fullfile(userpathstr,'Algo_Invest/workspace/Factor_Risk');
        save(fullfile(filepath,strcat(mdl_type,num2str(i),'_','Risk_',uni)), 'SysRisk', '-v7.3');   
        
        % save expected returns to disk
        SysRet = db_models.(mdl_type){j,i}.fexp; 
        filepath = fullfile(userpathstr,'Algo_Invest/workspace/Expected_Returns');
        save(fullfile(filepath,strcat(mdl_type,num2str(i),'_','ER_',uni)), 'SysRet');   
        
    end
end

clear t fct_ind fact_prem ret mdl_type beta beta_t index_t INDEX_t sys_error fcov fmu fact_prem_cov temp SysRet SysRisk exp_ret;

%% Out-of-sample forecast accuracy 
%
% specify models
all_mdls = {'ol', 'olu'};
% set forgetting factor
lambda = 0.98;

% for each model
for i = 1:length(all_mdls)
    % select model
    mdl_type = all_mdls{i};
    % specify models for each model type
    switch mdl_type
    case {'ol','olu'}
        mdl_indx = [2];
    case 'ch'
        mdl_indx = [6];
    case 'adch'
        mdl_indx = [2 5 6];
    end
    for j = 1:numel(mdl_indx)
%         expected returns - excess or absolute?
        clear fexp temp;
        fexp = db_models.(mdl_type){mdl_indx(j)}.fexp;
        % calculate Theil Stats
        if strcmp(mdl_type(end), 'u') % if update
            % systematic return innovations (FF3F)
%             sys_error = db_models.olu{2}.sys_error;
            % actual returns minus systematic return innovations
%             ret = db_procdata.returns.ret(updt(1):updt(2)) - sys_error;

%             ret = db_procdata.returns.ret(updt(1):updt(2));
              ret = db_procdata.returns.ret(updt(1):end);

            % set initial values
            mu0 = db_models.(mdl_type(1:2)){mdl_indx(j)}.Theil.MFE_T(end,:);
            COV_0 = db_models.(mdl_type(1:2)){mdl_indx(j)}.Theil.COV_FE;
            % clear initial Theil from memory
            db_models.(mdl_type(1:2)){mdl_indx(j)}.Theil = [];
            % calculate Theil Stats 
%             db_models.(mdl_type){mdl_indx(j)}.Theil = fewma_theil2(lambda, fexp, ret, {'mu0', mu0 }, {'COV_0', COV_0} , {'cov_type', 'updt'});
            % save to disk
            SysTheil = fewma_theil2(lambda, fexp, ret, {'mu0', mu0 }, {'COV_0', COV_0} , {'cov_type', 'updt'});
%             SysTheil = db_models.(mdl_type){mdl_indx(j)}.Theil; 
            filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
            save(fullfile(filepath,strcat(mdl_type,num2str(mdl_indx(j)),'_Theil_', uni)), 'SysTheil', '-v7.3');   
            clear SysTheil;
        else % if initial
            % systematic return innovations (FF3F)
%             sys_error = db_models.ol{2}.sys_error;
            % actual returns minus systematic return innovations
            ret = db_procdata.returns.ret(ini(1):ini(2));          
            % calculate Theil stats
            db_models.(mdl_type){mdl_indx(j)}.Theil = fewma_theil2(lambda, fexp, ret, {'cov_type', 'init'});
            % save to disk
%             SysTheil = fewma_theil2(lambda, fexp, ret, {'cov_type', 'init'});
            SysTheil = db_models.(mdl_type){mdl_indx(j)}.Theil; 
            filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
            save(fullfile(filepath,strcat(mdl_type,num2str(mdl_indx(j)),'_Theil_',uni)), 'SysTheil');
            clear SysTheil;
            
        end
    end
end

clear all_mdls lambda ret i mdl_type j fexp mu0 COV_0 AyaTheil;

%% characteristics models

% create characteristic data matrix
clear char_data char_data_ind;
for i = 1:size(db_procdata.char_data,2)
    ini_char_data = fts2mat(db_procdata.char_data{1,i});
    char_data(:,:,i) = ini_char_data(ini(1):ini(2),:);
    char_data_ind{i} = db_procdata.char_index{i};
end
clear ini_char_data;
% add CAPM beta to characteristic data

%{
% online CAPM beta
db_procdata.beta_data{1} = db_models.ol{1}.fbeta{2};
db_procdata.beta_index{1} = 'online CAPM MKT beta';
% online FF3F HML beta
db_procdata.beta_data{2} = db_models.ol{2}.fbeta{2};
db_procdata.beta_index{2} = 'online FF3F HML beta';
% online FF3F SMB beta
db_procdata.beta_data{3} = db_models.ol{2}.fbeta{3};
db_procdata.beta_index{3} = 'online FF3F SMB beta';
% online FF3F MKT beta
db_procdata.beta_data{4} = db_models.ol{2}.fbeta{4};
db_procdata.beta_index{4} = 'online FF3F MKT beta';
% online FF3F MKT alpha 
db_procdata.beta_data{5} = db_models.ol{2}.fbeta{1};
db_procdata.beta_index{5} = 'online FF3F alpha';
%}

% char_data(:,:,size(db_procdata.char_data,2)+1) = db_models.(mdl_type){1}.beta(:,:,2);
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas'); 
load(fullfile(filepath,strcat('ol_capm_beta_',uni)));
char_data(:,:,8) = fts2mat(ol_capm_beta);
char_data_ind{8} = 'CAPM \beta';
clear mdl_type i;

% add FF3F alpha to characteristic data
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas'); 
load(fullfile(filepath,strcat('ol_ff_alpha_',uni)));
char_data(:,:,9) = fts2mat(ol_ff_alpha);
char_data_ind{9} = 'FF3F \alpha';

clear ol_capm_beta ol_ff_alpha;

%% characteristic models

db_mdlindx.ch{1,1} = {[4 2]};         % bvtp & mv
db_mdlindx.ch{2,1} = {[4 2 8]};       % bvtp & mv % capm beta 
db_mdlindx.ch{3,1} = {[4 2 6]};       % bvtp & mv & moml
db_mdlindx.ch{4,1} = {[4 2 6 8]};     % bvtp & mv & moml & capm
db_mdlindx.ch{5,1} = {[4 2 6 7]};     % bvtp & mv & moml & moms
db_mdlindx.ch{6,1} = {[4 2 6 7 8]};   % bvtp & mv & capm beta & moml & moms
db_mdlindx.ch{7,1} = {[4 2 6 7 8 9]}; % bvtp & mv & capm beta & moml & moms & FF3F alpha
db_mdlindx.ch{8,1} = {[4 2 6 7 8 1]}; % bvtp & mv & capm beta & moml & moms & dy
db_mdlindx.ch{9,1} = {[4 2 6 7 8 9 1]}; % bvtp & mv & capm beta & moml & moms & FF3F alpha & dy

% variable descriptions
for i = 1:length(db_mdlindx.ch)
    db_mdlindx.ch{i,2} = char_data_ind(db_mdlindx.ch{i,1}{1});
end
clear i;

%% estimate characteristic models (initisalisation)
% we do not consider different lags

% db_models.ch = [];
ret = db_procdata.returns.eret(ini(1): ini(2));
ydata = fts2mat(db_procdata.returns.eret);
% ini data
ydata = ydata(ini(1):ini(2),:);
% choose model
% for k = 1:size(db_mdlindx.ch,1)
lag = 4;
% for k = 1:9
for k = 8
    ch_ind = cell2mat(db_mdlindx.ch{k,1});
    xdata = char_data(:,:,ch_ind);
    % choose stock universe
    for i = 1:length(ch_ind)
        xdata(:,:,i) = Iindex.*xdata(:,:,i);
    end 
    % cross-sectional regressions
    db_models.ch{k} = cs_reg_001ab(xdata,Iindex.*ydata, 'robustfit', lag);
    % store payoffs in fints
    db_models.ch{k}.payoffs = fints(ret.dates, db_models.ch{k}.beta, {} ,ret.freq, 'Payoffs');
end
% clear temp vars
clear ydata ch_ind xdata k i;

%% Expected Return for Characteristic Models

rfr = db_procdata.returns.rfr(ini(1):ini(2));
ret = db_procdata.returns.eret(ini(1):ini(2));
lambda = 0.97;
% count number of observations
% sumnan = nansum(~isnan(fts2mat(Iindex.*ret)),1);
% for i = 1:length(db_models.ch)
for i = 8
    clear exp_ret
    exp_ret(:,:) = NaN*ones(size(ret,1)+1, size(ret,2));
    % smooth payoffs
    db_models.ch{i}.EWpayoffs = fewma_mean(lambda, db_models.ch{i}.payoffs);
    for t = lag+1:length(db_models.ch{i}.payoffs)
        clear char_mtx;
        % index at t
%         index_t = Uindex(t,:);
        % matrix of firm characteristics (predictors)
        char_mtx = db_models.ch{i}.X(:,:,t-lag+1);
        % expected return smoothed payoffs
        exp_ret(t+1,:) = (char_mtx * fts2mat(db_models.ch{i}.EWpayoffs(t))')';
    end
    % store in fints
    db_models.ch{i}.fexperet = lagts(fints(ret.dates, exp_ret(2:end,:), fieldnames(ret,1), ret.freq),1,NaN);
    % forecast (includes risk-free rate)
    db_models.ch{i}.fexpret = db_models.ch{i}.fexperet + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
    % forecast error
    fc_err_raw = fts2mat(ret - db_models.ch{i}.fexperet);
    % count number of forecasts for each company
    raw_sumnan = nansum(~isnan(fc_err_raw),1);
    % mean forecast error per model
    db_models.ch{i}.MFE_raw =  nansum((nanmean(fc_err_raw.*fc_err_raw,1).*raw_sumnan))/nansum(raw_sumnan);
end

clear fc_err_raw fc_err_smooth raw_sumnan smooth_sumnan ;

%% characteristic data for update phase 

clear char_data char_data_ind;
for i = 1:size(db_procdata.char_data,2)
    updt_char_data = fts2mat(db_procdata.char_data{1,i});
%     char_data(:,:,i) = updt_char_data(updt(1):updt(2),:);
    char_data(:,:,i) = updt_char_data(updt(1):end,:);
    char_data_ind{i} = db_procdata.char_index{i};
end
clear updt_char_data;
% add CAPM beta to characteristic data
% mdl_type = 'ol';

%{
% online CAPM beta
db_procdata.beta_data{1} = db_models.olu{1}.fbeta{2};
db_procdata.beta_index{1} = 'online CAPM MKT beta';
% online FF3F HML beta
db_procdata.beta_data{2} = db_models.olu{2}.fbeta{2};
db_procdata.beta_index{2} = 'online FF3F HML beta';
% online FF3F SMB beta
db_procdata.beta_data{3} = db_models.olu{2}.fbeta{3};
db_procdata.beta_index{3} = 'online FF3F SMB beta';
% online FF3F MKT beta
db_procdata.beta_data{4} = db_models.olu{2}.fbeta{4};
db_procdata.beta_index{4} = 'online FF3F MKT beta';
% online FF3F MKT alpha 
db_procdata.beta_data{5} = db_models.olu{2}.fbeta{1};
db_procdata.beta_index{5} = 'online FF3F alpha';
%}

% char_data(:,:,size(db_procdata.char_data,2)+1) = db_models.(mdl_type){1}.beta(:,:,2);
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas'); 
load(fullfile(filepath,strcat('olu_capm_beta_',uni)));
char_data(:,:,8) = fts2mat(olu_capm_beta);
% char_data(:,:,8) = fts2mat(olu_capm_beta(1:updt(2)-updt(1)+1));
char_data_ind{8} = 'CAPM \beta';
clear mdl_type i;

% add FF3F alpha to characteristic data
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Betas'); 
load(fullfile(filepath,strcat('olu_ff_alpha_',uni)));
char_data(:,:,9) = fts2mat(olu_ff_alpha);
% char_data(:,:,9) = fts2mat(olu_ff_alpha(1:updt(2)-updt(1)+1));
char_data_ind{9} = 'FF3F \alpha';

clear ol_capm_beta ol_ff_alpha;
%% estimate Characteristics models (updt)

% db_models.chu = [];
%{
Uindex = index(updt(1):updt(2));
ret = db_procdata.returns.eret(updt(1): updt(2));
ydata = fts2mat(db_procdata.returns.eret(updt(1):updt(2)));
%}

Uindex = index(updt(1):end,:);
ret = db_procdata.returns.eret(updt(1): end);
ydata = fts2mat(db_procdata.returns.eret(updt(1):end));


% choose model
lag = 4;
ind = 1;
% for k = 1:size(db_mdlindx.ch,1)
for k = 8
    ch_ind = cell2mat(db_mdlindx.ch{k,1});
    xdata = char_data(:,:,ch_ind);
    % choose stock universe
    for i = 1:length(ch_ind)
        xdata(:,:,i) = Uindex.*xdata(:,:,i);
    end 
    % cross-sectional regressions
    db_models.chu{k} = cs_reg_001ab(xdata,Uindex.*ydata, 'robustfit', lag);
    % store payoffs in fints
    db_models.chu{k}.payoffs = fints(ret.dates, db_models.chu{k}.beta, {} ,ret.freq, 'Payoffs');
end

clear xdata ydata;

%% Expected Excess Returns Characteristics Models

% ret = db_procdata.returns.eret(updt(1):updt(2));
% rfr = db_procdata.returns.rfr(updt(1):updt(2));

lambda = 0.97;

ret = db_procdata.returns.eret(updt(1):end);
rfr = db_procdata.returns.rfr(updt(1):end);

% count number of observations
% sumnan = nansum(~isnan(fts2mat(Iindex.*ret)),1);
% for i = 1:length(db_models.chu)

% lagged characteristics by one month 
lag = 4;

for i = 8
    
    clear exp_ret CharRet
    exp_ret(:,:) = NaN*ones(size(ret,1)+1, size(ret,2));
    % smooth payoffs
    db_models.chu{i}.EWpayoffs = fewma_mean(lambda, db_models.chu{i}.payoffs);
    for t = lag+1:length(db_models.chu{i}.payoffs)
        clear char_mtx;
        % index at t
%         index_t = Uindex(t,:);
        % matrix of firm characteristics (predictors)
        char_mtx = db_models.chu{i}.X(:,:,t-lag+1);
        % expected return smoothed payoffs
        exp_ret(t+1,:) = (char_mtx * fts2mat(db_models.chu{i}.EWpayoffs(t))')';
    end
    % store in fints
    db_models.chu{i}.fexperet = lagts(fints(ret.dates, exp_ret(2:end,:), fieldnames(ret,1), ret.freq),1,NaN);
    % forecast (includes risk-free rate)
    db_models.chu{i}.fexpret = db_models.chu{i}.fexperet + repmat(fts2mat(rfr(t)),1,size(exp_ret,2));
   
    % save expected return model
    CharRet = db_models.chu{i}.fexpret;
    filepath = fullfile(userpathstr,'Algo_Invest/workspace/Expected_Returns');
    save(fullfile(filepath,strcat('chu',num2str(i),'_ER_',uni)), 'CharRet');   
    
    % forecast error
    fc_err_raw = fts2mat(ret - db_models.chu{i}.fexperet);
    % count number of forecasts for each company
    raw_sumnan = nansum(~isnan(fc_err_raw),1);
    % mean forecast error per model
    db_models.chu{i}.MFE_raw =  nansum((nanmean(fc_err_raw.*fc_err_raw,1).*raw_sumnan))/nansum(raw_sumnan);
    
end

clear index_t char_mtx exp_ret fc_err_raw fc_err_smooth raw_sumnan smooth_sumnan;

%% Out-of-sample forecast accuracy 
%
% specify models
all_mdls = {'ch', 'chu'};
% set forgetting factor
lambda = 0.97;

% for each model
for i = 1:length(all_mdls)
    % select model
    mdl_type = all_mdls{i};
    % specify models for each model type
    switch mdl_type
    case {'ol','olu'}
        mdl_indx = [2];
    case {'ch','chu'}
        mdl_indx = [8];
    case 'adch'
        mdl_indx = [2 5 6];
    end
    for j = 1:numel(mdl_indx)
%         expected returns - excess or absolute?
        clear fexp CharTheil;
        fexp = db_models.(mdl_type){mdl_indx(j)}.fexpret;
        % calculate Theil Stats
        if strcmp(mdl_type(end), 'u') % if update
            % systematic return innovations (FF3F)
%             sys_error = db_models.olu{2}.sys_error;
            % actual returns minus systematic return innovations
%             ret = db_procdata.returns.ret(updt(1):updt(2)) - sys_error;
%             ret = db_procdata.returns.ret(updt(1):updt(2));
            ret = db_procdata.returns.ret(updt(1):end);
            % set initial values
            filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
            load(fullfile(filepath,strcat(mdl_type(1:2),num2str(mdl_indx(j)),'_Theil_', uni )));
            mu0 = CharTheil.MFE_T(end,:); 
            COV_0 = CharTheil.COV_FE;
            clear CharTheil
%             mu0 = db_models.(mdl_type(1:2)){mdl_indx(j)}.Theil.MFE_T(end,:);
%             COV_0 = db_models.(mdl_type(1:2)){mdl_indx(j)}.Theil.COV_FE;
            % calculate Theil Stats 
%             db_models.(mdl_type){mdl_indx(j)}.Theil = fewma_theil2(lambda, fexp, ret, {'mu0', mu0 }, {'COV_0', COV_0} , {'cov_type', 'updt'});
            % save to disk
%             CharTheil = db_models.(mdl_type){mdl_indx(j)}.Theil; 
            CharTheil = fewma_theil2(lambda, fexp, ret, {'mu0', mu0 }, {'COV_0', COV_0} , {'cov_type', 'updt'});
            
            save(fullfile(filepath,strcat(mdl_type,num2str(mdl_indx(j)),'_Theil_', uni)), 'CharTheil', '-v7.3'); 
            
        else % if initial
            % systematic return innovations (FF3F)
%             sys_error = db_models.ol{2}.sys_error;
            % actual returns minus systematic return innovations
%             ret = db_procdata.returns.ret(ini(1):ini(2)) - sys_error;  
            ret = db_procdata.returns.ret(ini(1):ini(2));
            % calculate Theil stats
%             db_models.(mdl_type){mdl_indx(j)}.Theil = fewma_theil2(lambda, fexp, ret, {'cov_type', 'init'});
            % save to disk
%             CharTheil = db_models.(mdl_type){mdl_indx(j)}.Theil; 
            CharTheil = fewma_theil2(lambda, fexp, ret, {'cov_type', 'init'});
            filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
            save(fullfile(filepath,strcat(mdl_type,num2str(mdl_indx(j)),'_Theil_', uni)), 'CharTheil', '-v7.3'); 
            
        end
    end
end
clear all_mdls lambda ret i mdl_type j fexp CharTheil sys_error;




%% GMV Portfolio Construction

% absolute returns
% ret = db_procdata.returns.ret(updt(1):updt(2)).*Uindex;

%{
% ret = db_procdata.returns.ret(updt(1):end).*Uindex;
% standard GMV (double index)

clear GMV;
GMV = cell(1,11);
for i = 0:0.1:1
    GMV{round(10*i+1)} = GMV_panel001aa(ret, db_models.olu{2}.risk, i );
end

filepath = fullfile(userpathstr,'Algo_Invest/workspace/portfolios');
cd(filepath);
save(strcat('GMV_',uni)'GMV');
save(strcat('STRAT_',uni), 'STRAT');
%}


%% STRAT Portfolio Construction

clearvars -except index db_procdata updt userpathstr uni;
% in-sample
%{
Uindex = index(updt(1):updt(2),:);
ret = db_procdata.returns.ret(updt(1):updt(2)).*Uindex;
rfr = db_procdata.returns.rfr(updt(1):updt(2));
%}

% out-sample
Uindex = index(updt(1):end,:);
ret = db_procdata.returns.ret(updt(1):end).*Uindex;
rfr = db_procdata.returns.rfr(updt(1):end);

% risk
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Factor_Risk');
load(fullfile(filepath,strcat('olu2_Risk_',uni))) ;
% SysRisk = temp(:) ;
% clear temp;

% return
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Expected_Returns');
load(fullfile(filepath,strcat('olu2_ER_',uni))) ;
% SysRet = tempRet ;
% clear temp;

% uncertainty
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
load(fullfile(filepath,strcat('olu2_Theil_',uni))) ;
% SysTheil = temp(:) ;
% clear temp;

%{
% in-sample portfolio
% STRAT = cell(11,15);
STRAT = cell(1,11);
% including tuning parameter
for k = 0:0.1:1 % shrinkage intensity
    j = 1; % tuning parameter
%     for j = 0.1:0.1:1.5 % tuning parameter
        STRAT{round(10*k+1)} = STRAT_panel001ab(ret, rfr, SysRisk, SysTheil.COV_FE, fts2mat(SysRet), k, j); 
%     end
end

filepath = fullfile(userpathstr,'Algo_Invest/workspace/portfolios');
cd(filepath);
save(strcat('STRAT_',uni), 'STRAT');
%}

% out-of-sample portfolio
k = 3;
j = 1;
STRAT_final = STRAT_panel001ab(ret, rfr, SysRisk, SysTheil.COV_FE, fts2mat(SysRet), k, j);
filepath = fullfile(userpathstr,'Algo_Invest/workspace/portfolios');
cd(filepath);
save(strcat('STRAT_final_',uni), 'STRAT_final');

%% TACT 


% absolute returns
%{
Uindex = index(updt(1):updt(2),:);
ret = db_procdata.returns.ret(updt(1):updt(2)).*Uindex;
rfr = db_procdata.returns.rfr(updt(1):updt(2));
%}

Uindex = index(updt(1):end,:);
ret = db_procdata.returns.ret(updt(1):end).*Uindex;
rfr = db_procdata.returns.rfr(updt(1):end);

clear db_procdata;

% Factor risk
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Factor_Risk');
load(fullfile(filepath,strcat('olu2_Risk_',uni))) ;
% SysRisk = temp(:) ;
% clear temp;

% Systematic Uncertainty
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
load(fullfile(filepath,strcat('olu2_Theil_',uni))) ;
% SysTheil = temp;
% clear temp;

% Systematic return
filepath = fullfile(userpathstr,'Algo_Invest/workspace/Expected_Returns');
load(fullfile(filepath,strcat('olu2_ER_',uni))) ;
% SysRet = tempRet ;
% clear tempRet;

% Characteristic Uncertainty
for i = 8
    filepath = fullfile(userpathstr,'Algo_Invest/workspace/Theil_matrices');
    load(fullfile(filepath,strcat('chu', num2str(i) ,'_Theil_',uni))) ;

    % Characteristic return
    filepath = fullfile(userpathstr,'Algo_Invest/workspace/Expected_Returns');
    load(fullfile(filepath,strcat('chu', num2str(i) ,'_ER_',uni))) ;

    % shrinkage intensity
%     TACT = cell(1,11);
    for k = 2
%         TACT{round(10*k+1)}  = TACT_panel001aa(ret, rfr,  SysRisk, SysTheil.COV_FE, CharTheil.COV_FE, fts2mat(SysRet), fts2mat(CharRet), k, 1 );
        TACT  = TACT_panel001aa(ret, rfr,  SysRisk, SysTheil.COV_FE, CharTheil.COV_FE, fts2mat(SysRet), fts2mat(CharRet), k, 1 );
    end

    filepath = fullfile(userpathstr,'Algo_Invest/workspace/portfolios');
    cd(filepath)
%     save(strcat('TACT_',num2str(i),'_',uni), 'TACT');
    save(strcat('TACT_',num2str(i),'_final_',uni), 'TACT');
end
