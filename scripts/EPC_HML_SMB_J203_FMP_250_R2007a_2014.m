%% Characteristic and Factor Models on South African Stocks
% R2007a
% See Also : HOLIDAYS, NTILE
%
% Following the approach for Fama & French / Daniel & Titman
%
% HML : (3-tile on BVTP)
%
% SMB : (2-tile on MV)
%
% Authors : Tim Gebbie 

% $Revision: 1.23 $ $Date: 2007/07/08 12:05:05 $ $Author: tgebbie $

%% Load the Data
clear all; clc;
warning('off');
% >--------------- NUMBER OF STOCKS ----------------------------
nindex = 250; % the number of stocks to check each formation date
% <-------------------------------------------------------------
% epoch_range = daterange('31-Jan-1994',today,'M');
% combination of all 'JSE','J' and 'J203' codes
% all_codes = 'AAA,ABI,ABL,ABO,ABT,ACC,ACD,ACH,ACN,ACP,ACT,ACY,ADH,ADL,ADO,ADR,ADW,AEC,AEG,AER,AET,AFB,AFE,AFG,AFI,AFL,AFO,AFR,AFT,AFX,AGI,AGL,AGS,AHH,AHV,AIN,ALA,ALC,ALD,ALE,ALJ,ALN,ALT,ALX,ALY,AMA,AMB,AME,AMS,ANG,AOD,AON,AOO,APA,APB,APE,APK,APL,APN,APP,APS,AQL,AQP,AQU,ARC,ARI,ARL,ARP,ARQ,ART,ASA,ASG,ASO,ASR,ATN,ATR,ATS,AVA,AVG,AVI,AVS,AWT,AXC,BAT,BAW,BCF,BCX,BDE,BDM,BDS,BEE,BEG,BEL,BFS,BIC,BIL,BJM,BNT,BOE,BPL,BRC,BRM,BRN,BRT,BRY,BSB,BSR,BTG,BUR,BVT,CAE,CAL,CAN,CAT,CBS,CCG,CCL,CCT,CDT,CDZ,CEL,CEN,CHE,CKS,CLE,CLH,CLO,CMA,CMG,CMH,CML,CMO,CMT,CNC,CND,CNL,CNX,COM,CPA,CPI,CPL,CPT,CPX,CRG,CRH,CRM,CRN,CRW,CRX,CSB,CSH,CSL,CTP,CUL,CVS,CXT,CYB,CYD,DAW,DCT,DDT,DEC,DEL,DGC,DIV,DLG,DLV,DMR,DNA,DON,DRD,DST,DSY,DTC,DTP,DUR,DYM,ECH,ECO,ELD,ELH,ELR,ELX,EMI,ENL,ENR,ENV,EOH,EPL,ERM,ERP,ESR,ETH,EUR,EXL,EXO,EXX,FBR,FLC,FOM,FOS,FOZ,FRE,FRO,FRT,FSH,FSP,FSR,FUM,FVT,FWX,GBG,GBL,GDA,GDC,GDF,GDH,GDN,GFI,GIJ,GLB,GLE,GLL,GLT,GMB,GMF,GND,GNK,GNN,GRF,GRT,GRY,GUB,GVM,HAL,HAR,HCH,HCI,HCL,HDC,HPA,HPB,HVL,HVN,HWA,HWN,HYP,ICC,ICT,IDI,IDQ,IFC,IFH,IFR,IFW,IIS,ILA,ILT,ILV,IMP,IMR,IMU,IND,INL,INM,INP,INS,INT,IPL,IPR,IPS,ISA,ISC,IST,ITE,ITG,ITR,ITV,IVT,JBL,JCD,JCG,JCM,JDG,JDH,JGS,JNC,JSC,JSE,KAP,KEL,KER,KGM,KIO,KIR,KLG,KMB,KNG,KWV,LAB,LAF,LAN,LAR,LBH,LBT,LEW,LGL,LNF,LON,LSR,LST,MAF,MAS,MBN,MBT,MCE,MCP,MCU,MDC,MDN,MEC,MEL,MES,MET,MFL,MGX,MHH,MKL,MKX,MLA,MLB,MLL,MMG,MMH,MML,MNY,MOB,MOZ,MPC,MPL,MRB,MRF,MRN,MSM,MSS,MST,MTA,MTC,MTE,MTL,MTN,MTO,MTP,MTX,MTZ,MUM,MUR,MVG,MVL,MYD,NAC,NAI,NAM,NAN,NBK,NCA,NCL,NCS,NED,NEI,NFH,NHM,NIB,NIN,NMS,NPK,NPN,NTC,NWL,OAI,OAO,OAS,OCE,OCT,OLG,OML,OMN,ORE,OSI,OTK,OTR,OZZ,PAC,PAL,PAM,PAP,PCN,PEP,PET,PFN,PGD,PGH,PGR,PHM,PIK,PIM,PMA,PMG,PML,PMM,PMN,PMV,PNC,PON,PPC,PPE,PPR,PRA,PRM,PRO,PSC,PSG,PSV,PTC,PTG,PWK,PZG,QUY,RAG,RAH,RAR,RBV,RBW,RBX,RCH,RCO,RDF,REM,RES,RHW,RLO,RLY,RMH,RNG,RNT,RPR,RTN,RTO,SAB,SAC,SAE,SAL,SAM,SAN,SAP,SBG,SBK,SBL,SBN,SBO,SBV,SCE,SCN,SER,SFA,SFN,SFT,SGG,SHF,SHP,SIC,SIF,SIM,SIR,SIS,SJL,SKJ,SLL,SLM,SLO,SLU,SMC,SMR,SNG,SNT,SNV,SOH,SOL,SOV,SPA,SPE,SPG,SPP,SPS,SQE,SRL,SRN,SSA,STA,STE,STI,STK,STO,SUI,SUL,SUM,SUR,SVB,SVN,SWL,SXR,SYA,SYC,SZA,TAL,TAS,TBS,TBX,TDH,TDK,TEL,TFX,TGN,TIW,TKG,TLM,TMT,TNT,TPC,TRE,TRF,TRT,TRU,TRX,TSC,TSO,TSX,UCS,UNG,USV,UTR,V$A,V0A,V2A,V3A,V7B,V9B,VAJ,VBB,VCR,VDA,VER,VIL,VJA,VKE,VKG,VLE,VLT,VMK,VNA,VNF,VST,VTL,VTR,VXA,VZA,WAN,WAR,WBH,WBO,WCA,WEA,WES,WET,WEZ,WGR,WHL,WKF,WLL,WLN,WLO,WNH,WPH,WSL,XAN,YBA,YHK,YRK,ZCI,ZED,ZLT,ZPT,ZRR,ZZA';
% items = {'PTBV','MV','RI','VO','NT','PE','DY'};

%% Load the data (there should be at least 400 codes)
load EPC_HML_SMB_J203_FMP_workspace_R2007a;

%% Money Market
% check this NACQ (90-day NCD) to create a monthly price index to proxy
% cash using 3 month NCD's PV = MV / ( 1 + diff/365(ytm/100)) [Approx]
rfi  = log(1+(30./365)*(rfr.('CLOSE')./100));
% initialise the first data point
rfi(1) = 0;
% compute compounded continuous returns as a price index
rfi = exp(filter(1,[1 -1],fillts(rfi,0)));

%% the Characteristics (@FINTS)
% fts{1}.desc=DY
% fts{2}.desc=MV
% fts{3}.desc=NT
% fts{4}.desc=PE
% fts{5}.desc=PTBV
% fts{6}.desc=RI
% fts{7}.desc=VO
%
bvtp = 1 ./ fts{5}; fts{5}.desc 
bvtp.desc = 'BVTP'; % BVTP = 1/PTBV
mv   = inf2nan(log(fts{2})); fts{2}.desc % SIZE = LOG(MV)
ey   = 1 ./ fts{4}; fts{4}.desc % EY = 1/PE 
ey.desc = 'EY';
ri   = fts{6}; fts{6}.desc % RI
vol  = fts{7}; fts{7}.desc % VOL
dy   = fts{1}; fts{1}.desc % DY % check zero dividend yield
dy   = zero2nan(dy); % zero dividends are NaN
dy   = fillts(dy,'z'); % use zero order hold

%% Set the sampling frequency
freq = 'M';
bvtp = convertto(bvtp,freq);
mv   = convertto(mv,freq);
ri   = convertto(ri,freq);
ey   = convertto(ey,freq);
dy   = convertto(dy,freq);
moml = tsmom(log(ri),freqsamplerate(freq)); moml.desc = 'MOML'; % twice the sampling frequency (1YR)
% acl  = tsaccel(log(ri),12); acl.desc = 'ACL'; % twice the sampling
moms = tsmom(log(ri),freqsamplerate(freq)/4); moms.desc = 'MOMS';  % (1 Q)
vol  = tsmom(log(convertto(vol,freq)),freqsamplerate(freq)/4); % (1 Q)
rfi  = convertto(rfi,freq);

%% Compute the log returns
ret = filter([1 -1],1,log(ri));  % a y = b x + c => 1 y = 1 x(t) + (-1) x(t-1) + 0
rfr2 = filter([1 -1],1,log(rfi)); % synthetic money market index

%% Process data
% zero prices are set to NaN
x = fts2mat(ri);
x(x==0)=NaN;
ri = fints(ri.dates,x,fieldnames(ri,1),ri.freq,ri.desc);

%% Sort into the top 100 on size each 
[a]=size(mv);
index = false(a);
x = fts2mat(mv);
x(isnan(x))=0;
for i=1:a(1)
    [x0, ind] = sort(x(i,:),'descend');
    index(i,ind(1:nindex)) = true;
end;
index = double(index);
index(index==0)=NaN;

%% lead the BVTP by sampling frequency 
%           MV      BVTP*   RI    RET       * is lead 1 month
% Date #2   C1      A1      I2    I2/I1
% Date #3   C2      A2      I3    I3/I2
bvtp = lagts(bvtp,freqsamplerate(freq)/4); % lag by 1Q -> HML & SMB
mv   = lagts(mv,freqsamplerate(freq)/4);   % lag by 1Q -> HML & SMB
ey   = lagts(ey,freqsamplerate(freq)/4);   % lag by 1Q
dy   = lagts(dy,freqsamplerate(freq)/4);   % lag by 1Q
moms = lagts(moms,1);
moml = lagts(moml,1);
% acl  = lagts(acl,1);
vol  = lagts(vol,1);

%% Sort into n-tiles on the characteristics QUARTERLY
nb = ntile(transpose(index .* fts2mat(bvtp)),3,'descend');
nb = fints(bvtp.dates,transpose(nb),fieldnames(bvtp,1),bvtp.freq,bvtp.desc);
nm = ntile(transpose(index .* fts2mat(mv)),2,'descend');
nm = fints(mv.dates,transpose(nm),fieldnames(mv,1),mv.freq,mv.desc);
nm3 = ntile(transpose(index .* fts2mat(mv)),3,'descend');
nm3 = fints(mv.dates,transpose(nm3),fieldnames(mv,1),mv.freq,mv.desc);

%% Drop the leading data points and last data points
n = 2;
nb    = nb   (n:end-n+1);
nm    = nm   (n:end-n+1);
nm3   = nm3  (n:end-n+1);
ri    = ri   (n:end-n+1);
ret   = ret  (n:end-n+1);
mv    = mv   (n:end-n+1);
bvtp  = bvtp (n:end-n+1);
index = index(n:end-n+1,:);
rfr   = rfr  (n:end-n+1);
moml  = moml (n:end-n+1);
moms  = moms (n:end-n+1);
% acl   = acl(n:end-n+1);
vol   = vol  (n:end-n+1);
ey    = ey   (n:end-n+1);
dy    = dy   (n:end-n+1);

%% Tsallis entropy
mvx = index .* fts2mat(mv);
% Fernholz diversity
iDIV  = ((nansum(mvx.^(0.75)')).^(1/0.75))';
% Shannon entropy (extensive)
iSHA  = - nansum((mvx' .* log(mvx)'))';
%---- TSALLIS ENTROPY (non-extensive) ------
q = 2;
iLOC  = -((nansum(mvx'.^q) - 1)/(q-1)); % Entropy Factor
iLOC(1:3)=NaN;
iLOC = iLOC(:);
dLOC = diff(iLOC);
%-------------------------------------------
% create the entropy objects
entropy = fints(mv.dates,[iLOC(:) iDIV(:) iSHA(:)],{'LOC','DIV','SHA'},mv.freq,'Entropy');

%% Six Factor Mimicing Portfolios (FMP's)
nbx = fts2mat(nb);
nmx = fts2mat(nm);
nm3x = fts2mat(nm3);
% create the index for the different bins
small  = (nmx==2);
big    = (nmx==1);
high   = (nbx==1);
medium = (nbx==2);
low    = (nbx==3);
% create LS (low small) intersection of nbvtp and nmv equally weighted
LS = low & small;
% create MS (medium small)
MS = medium & small;
% create HS (high small)
HS = high & small;
% create LB (low big)
LB = low & big;
% create MB (medium big)
MB = medium & big;
% create HB (high big)
HB = high & big;

%% niave normalizations - sum of rows
normH = transpose(sum(transpose(high)));
normM = transpose(sum(transpose(medium)));
normL = transpose(sum(transpose(low)));
normB = transpose(sum(transpose(big)));
normS = transpose(sum(transpose(small)));

%% Compute the normalizations
normLS = transpose(sum(transpose(LS)))
normMS = transpose(sum(transpose(MS)));
normHS = transpose(sum(transpose(HS)));
normLB = transpose(sum(transpose(LB)));
normMB = transpose(sum(transpose(MB)));
normHB = transpose(sum(transpose(HB)));

%% Create the indices
retx = fts2mat(ret);
% remove inf values
retx((abs(retx)==Inf))=NaN;
% remove the initial value
retx(1,:) = 0;
% compute the returns - equal  weighting
rLS = transpose(nansum(transpose(retx .* LS))) ./ normLS; 
rMS = transpose(nansum(transpose(retx .* MS))) ./ normMS;
rHS = transpose(nansum(transpose(retx .* HS))) ./ normHS;
rLB = transpose(nansum(transpose(retx .* LB))) ./ normLB;
rMB = transpose(nansum(transpose(retx .* MB))) ./ normMB;
rHB = transpose(nansum(transpose(retx .* HB))) ./ normHB;
% niave FMPS
rH = transpose(nansum(transpose(retx .* high))) ./ normH;
rL = transpose(nansum(transpose(retx .* low))) ./ normL;
rB = transpose(nansum(transpose(retx .* big))) ./ normB;
rS = transpose(nansum(transpose(retx .* small))) ./ normS;

%% Price Index for 6 formation portfolios
h=figure;
retfts = fints(nb.dates,[rLS, rMS, rHS, rLB, rMB, rHB],{'LS','MS','HS','LB','MB','HB'},nb.freq,'Returns');
prcfts = fillts(retfts(12:end),0);
prcfts(1) = 0;
prcfts = exp(cumsum(prcfts));
plot1 = plot(prcfts);
set(plot1(1),'DisplayName','LS','LineWidth',2);
set(plot1(2),'DisplayName','MS');
set(plot1(3),'DisplayName','HS','LineWidth',2,'LineStyle','--');
set(plot1(4),'DisplayName','LB','LineStyle','--');
set(plot1(5),'DisplayName','MB','LineWidth',2,'LineStyle',':');
set(plot1(6),'DisplayName','HB','LineStyle',':');
title(sprintf('FMP Index - Top %d',nindex));
set(gca,'YScale','log');
ylabel('Price Index');
xlabel('Date');
legend('location','NorthWest');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP1',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
% for pt=1:3
%     print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP1',nindex));
% end
% hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP1',nindex));

%% Create the quarterly sorts of H(igh)-M(inus)-L(ow) Book-to-Market
rHML = (1/2)*((rHB + rHS) - (rLB + rLS));

%% Create the quarterly sorts of S(mall)-M(inus)-B(ig) on Log-size
rSMB = (1/3)*((rHS + rMS + rLS) - (rHB + rMB +rLB));

%% Create a market index (market cap weighted - balanced portfolio)
normMV = transpose(nansum(transpose(mvx)));
rMKT = transpose(nansum(transpose(retx .* mvx))) ./ normMV;

%% Price index for 3 factor mimicing portfolios
h=figure;
fmpfts = fints(nb.dates,[rHML rSMB rMKT],{'HML','SMB','MKT'},nb.freq,'Returns');
prcfmp = fillts(fmpfts(12:end),0);
prcfmp(1) = 0;
prcfmp = exp(cumsum(prcfmp));
plot1 = plot(prcfmp);
set(plot1(1),'DisplayName','HML','LineWidth',1,'LineStyle','-');
set(plot1(2),'DisplayName','SMB','LineWidth',2,'LineStyle','--');
set(plot1(3),'DisplayName','MKT','LineWidth',2,'LineStyle',':');
title(sprintf('Top %d factor loading portfolio price index',nindex));
ylabel('Price Index');
xlabel('Date');
% set(gca,'YScale','log');
legend('location','NorthWest');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
% for pt=1:3
%     print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP2',nindex));
% end
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP2',nindex));
% hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP2',nindex));

%% Active and Excess Returns
% EXCESS RETURNS
eret = ret - repmat(fts2mat(rfr),1,size(ret,2));
% find the data matrix
eretx = fts2mat(eret);
% remove inf values
eretx((abs(eretx)==Inf))=NaN;
% remove the initial value
eretx(:,1) = 0;
% ACTIVE RETURNS
aret = ret - repmat(fts2mat(fmpfts.MKT),1,size(ret,2));
% find the data matrix
aretx = fts2mat(aret);
% remove inf values
aretx((abs(aretx)==Inf))=NaN;
% remove the initial value
aretx(:,1) = 0;

%% Create the pre- through post- formation date indices
% See Table 2 Daniel & Titman 1997
ct = 24;
% forward looking P(t-L)/P(t-1-L) * Index(t) for  L = 0 to 12
for i=1:ct
    % LAGTS  -->   Date #2  Data #1
    %              Date #3  Data #2
    % create the lag
    % 1.  Correct for the 1 Quarter lag in BVTP and MV to ensure that 
    %     the formation data is set to zero at the same point as that for 
    %     the return date. Note that this means that BVTP(t) and MV(t)
    %     line up with the return from P(t)/P(t-1). The actual HML and SMB
    %     portfolio as based on a 1 quarter lag in this paper - that is 
    %     the BVTP and MV numbers 1 quarter prior to the date at which the 
    %     return are computed. Zero is at Lag0 = 1 Quarter
    % 2.  The lag is then iteratively increased from this point. (i)
    %
    retxf = lagts(ret,i-1+freqsamplerate(freq)/4);
    retxf = fts2mat(retxf);
    % remove inf values
    retxf((abs(retxf)==Inf))=NaN;
    % remove the initial value
    retxf(:,1) = 0;
    % compute the returns
    rLSf(:,i) = transpose(nansum(transpose(retxf .* LS))) ./ normLS;
    rMSf(:,i) = transpose(nansum(transpose(retxf .* MS))) ./ normMS;
    rHSf(:,i) = transpose(nansum(transpose(retxf .* HS))) ./ normHS;
    rLBf(:,i) = transpose(nansum(transpose(retxf .* LB))) ./ normLB;
    rMBf(:,i) = transpose(nansum(transpose(retxf .* MB))) ./ normMB;
    rHBf(:,i) = transpose(nansum(transpose(retxf .* HB))) ./ normHB;
end
% backward looking P(t-L)/P(t-1-L) * Index(t) for L = 0 to -12
for i=1:ct
    % LEADTS  -->  Date #2  Data #3
    %              Date #3  Data #4
    % create the leadts
    % 1.  Correct for the 1 Quarter lag in BVTP and MV to ensure that 
    %     the formation data is set to zero at the same point as that for 
    %     the return date. Note that this means that BVTP(t) and MV(t)
    %     line up with the return from P(t)/P(t-1). The actual HML and SMB
    %     portfolio as based on a 1 quarter lag in this paper - that is 
    %     the BVTP and MV numbers 1 quarter prior to the date at which the 
    %     return are computed. Zero is at Lag0 = 1 Quarter
    % 2.  The lag is then iteratively increased from this point. (i)
    %
    retxf = leadts(ret,i-1+freqsamplerate(freq)/4);
    retxf = fts2mat(retxf);
    % remove inf values
    retxf((abs(retxf)==Inf))=NaN;
    % remove the initial value
    retxf(:,1) = 0;
    % compute the returns
    rLSf(:,ct+i) = transpose(nansum(transpose(retxf .* LS))) ./ normLS;
    rMSf(:,ct+i) = transpose(nansum(transpose(retxf .* MS))) ./ normMS;
    rHSf(:,ct+i) = transpose(nansum(transpose(retxf .* HS))) ./ normHS;
    rLBf(:,ct+i) = transpose(nansum(transpose(retxf .* LB))) ./ normLB;
    rMBf(:,ct+i) = transpose(nansum(transpose(retxf .* MB))) ./ normMB;
    rHBf(:,ct+i) = transpose(nansum(transpose(retxf .* HB))) ./ normHB;
end

%% Create the quarterly sorts of H(igh)-M(inus)-L(ow) Book-to-Market
rHMLf = (1/2)*((rHBf + rHSf) - (rLBf + rLSf));

%% Create the quarterly sorts of S(mall)-M(inus)-B(ig) on Log-size
rSMBf = (1/3)*((rHSf + rMSf + rLSf) - (rHBf + rMBf +rLBf));

%% Plot the pre- post- formation performance
times= [-23:0,0:23];
h = figure;
subplot(3,1,1);
[haxes, hline1,hline2] = plotyy(times,nanmean(rLSf),times,nanstd(rLSf));
title(sprintf('%s - Top %d','LS pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
legend([hline1; hline2],'Return','Volatility');
subplot(3,1,2);
[haxes, hline1,hline2] = plotyy(times,nanmean(rMSf),times,nanstd(rMSf));
title(sprintf('%s - Top %d','MS pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
subplot(3,1,3);
[haxes, hline1,hline2] = plotyy(times,nanmean(rHSf),times,nanstd(rHSf));
title(sprintf('%s - Top %d','HS pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF1',nindex));
end
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF1',nindex));
hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF1',nindex));
h=figure;
subplot(3,1,1);
[haxes, hline1,hline2] = plotyy(times,nanmean(rLBf),times,nanstd(rLBf));
title(sprintf('%s - Top %d','LB pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
legend([hline1; hline2],'Return','Volatility');
subplot(3,1,2);
[haxes, hline1,hline2] = plotyy(times,nanmean(rMBf),times,nanstd(rMBf));
title(sprintf('%s - Top %d','MB pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
subplot(3,1,3);
[haxes, hline1,hline2] = plotyy(times,nanmean(rHBf),times,nanstd(rHBf));
title(sprintf('%s - Top %d','HB pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF2',nindex));
end
% print(h,'-depsx','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF2',nindex));
hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF2',nindex));

%% Plot the pre- post- formation performance HML SMB
times= [-23:0,0:23];
h = figure;
subplot(2,1,1);
[haxes, hline1,hline2] = plotyy(times,nanmean(rHMLf),times,nanstd(rHMLf));
title(sprintf('%s - Top %d','HML pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
legend([hline1; hline2],'Return','Volatility');
subplot(2,1,2);
[haxes, hline1,hline2] = plotyy(times,nanmean(rSMBf),times,nanstd(rSMBf));
title(sprintf('%s - Top %d','SMB pre-formation and post-formation behaviour',nindex));
xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
set(hline1,'LineWidth',2,'LineStyle','--');
set(hline2,'LineWidth',1,'LineStyle','-');
ylabel(haxes(1),'Return R_t[r]');
ylabel(haxes(2),'Volatility \sigma(R_t)');
legend([hline1; hline2],'Return','Volatility');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF3',nindex));
end
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF3',nindex));
hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF3',nindex));

%% Find the Beta's of the FMP's over entire sample window HML, SMB, MKT
clear multi;
% degrees of freedom
mdof = 6 * freqsamplerate(freq);
% set up the factors
factors = fmpfts.({'HML','SMB','MKT'});
% dependent data
ydata = fts2mat(ret);
% independent data
fdata = fts2mat(factors);
% create excess returns (using RFR)
% >-----------------------------------------------------------
fdata(:,3) = fdata(:,3) - fts2mat(rfr.CLOSE);
ydata = ydata - repmat(fts2mat(rfr.CLOSE),1,size(ydata,2));
% <-----------------------------------------------------------
% LOOP ACROSS EACH ENTITY
% for k=1:size(ret,2),
    % k-th entities time-series data as dependent variable
    % LOOP OVER THE EPOCH DATES
    k=1;
    for j=mdof+1:length(factors),
        % j-th epoch
        % MULTIFACTOR REGRESSIONS
        y = ydata(j-mdof:j,k);
        x = fdata(j-mdof:j,:);
        % set all the Infs to NaN
        x(abs(x)==Inf)=NaN;
        y(abs(x)==Inf)=NaN;
        % remove NaN data
        nidx = ~(isnan(y) | transpose(sum(transpose(isnan(x)))>0));
        % we only keep those data point that have all the factors (selection effect)
        y=y(nidx);
        x=x(nidx,:);
        if any(isnan(y)) | (length(y)<5),
            % sort the model statistics
            multi.beta(k,:,j)       = NaN*ones(size(x,2)+1,1);
            multi.t(k,:,j)          = NaN*ones(size(x,2)+1,1); 
        else
            % carry out regressions (must check the compatibility of robustrisk)
            [beta, stats] = robustfit(x,y);
            % raw model statisics
            s.X = [ones(size(x,1),1) x];
            s.t    = stats.t;
            s.p    = stats.p;
            s.b    = beta;
            s.se   = stats.se;
            % derived model statistics
            s.y        = y;
            s.yhat     = s.X * s.b;
            s.ybar     = nanmean(y);
            s.resid    = s.y - s.yhat;
            s.sse      = norm(s.resid)^2;
            s.ssr      = norm(s.yhat - s.ybar)^2;
            s.sst      = norm(s.y - s.ybar)^2;
            s.dfe      = length(y) - length(s.b); % # obs - # beta's
            s.dfr      = length(s.b) - 1;
            s.dft      = length(y)-1;
            s.F        = (s.ssr / s.dfr) * (s.sse / s.dfe);
            s.pval     = 1 - fcdf(s.F,s.dfr,s.dfe);
            s.adjr2    = 1 - s.sse ./ s.sst * (s.dft./s.dfe); % for constant term
            % sort the model statistics
            multi.beta(k,:,j)       = s.b;
            multi.se(k,:,j)         = s.se;
            multi.t(k,:,j)          = s.t;
            multi.adjr2(k,:,j)      = s.adjr2;
            multi.pval(k,:,j)       = s.pval;
        end;
    end;
% end;

%% Create the time-series objects of Multi-Factor Beta's
beta_bias = fints(ret.dates,transpose(squeeze(multi.beta(:,1,:))),fieldnames(ret,1),ret.freq);
beta_hml  = fints(ret.dates,transpose(squeeze(multi.beta(:,2,:))),fieldnames(ret,1),ret.freq);
beta_smb  = fints(ret.dates,transpose(squeeze(multi.beta(:,3,:))),fieldnames(ret,1),ret.freq);
beta_mkt  = fints(ret.dates,transpose(squeeze(multi.beta(:,4,:))),fieldnames(ret,1),ret.freq);

%% Compute Beta's - Market Weighted beta's (approximation to the index)
a(1:mdof,1:size(retx,2))=NaN;
b(1:mdof,1:size(retx,2))=NaN;
t(1:mdof,1:size(retx,2))=NaN;
rfrx = fts2mat(rfr);
retx(retx==0)=NaN;
for j=mdof+1:size(retx,1),
    for i=1:size(retx,2),
        x = rMKT(j-mdof:j) - rfrx(j-mdof:j);
        y = retx(j-mdof:j,i) - rfrx(j-mdof:j);
        % deal with missing data
        
        %nidx = ~(isnan(y) | (sum(isnan(x))>0));
        nidx = ~(isnan(y) | isnan(x));
        % we only keep those data point that have both MV and BVTP data
        y=y(nidx,:);
        x=x(nidx,:);
        % ----------------------
        if length(y)>3
            [beta, stats] = robustfit(x,y,[],[],'on');
        else
            beta = [NaN; NaN];
            stats.t = [NaN; NaN];
        end
        a(j,i) = beta(1);
        b(j,i) = beta(2);
        t(j,i) = stats.t(2);
    end;
end;

%% Create the time-series objects of Market Beta's
beta0_mkt  = fints(ret.dates,b,fieldnames(ret,1),ret.freq);
alpha0_mkt = fints(ret.dates,a,fieldnames(ret,1),ret.freq);

%% Compute the average Beta's
% compute the average beta's
bLS = transpose(nansum(transpose(b .* LS))) ./ normLS; 
bMS = transpose(nansum(transpose(b .* MS))) ./ normMS;
bHS = transpose(nansum(transpose(b .* HS))) ./ normHS;
bLB = transpose(nansum(transpose(b .* LB))) ./ normLB;
bMB = transpose(nansum(transpose(b .* MB))) ./ normMB;
bHB = transpose(nansum(transpose(b .* HB))) ./ normHB;

%% Visualise the average Beta's
h=figure
bfts = fints(nb.dates,[bLS, bMS, bHS, bLB, bMB, bHB],{'LS','MS','HS','LB','MB','HB'},nb.freq,'Beta');
plot1=plot(bfts(mdof+12:end));
set(plot1(1),'DisplayName','LS','LineWidth',2);
set(plot1(2),'DisplayName','MS');
set(plot1(3),'DisplayName','HS','LineWidth',2,'LineStyle','--');
set(plot1(4),'DisplayName','LB','LineStyle','--');
set(plot1(5),'DisplayName','MB','LineWidth',2,'LineStyle',':');
set(plot1(6),'DisplayName','HB','LineStyle',':');
title('Average factor loading portfolio CAPM market factor');
ylabel('\beta');
legend('location','NorthWest')
set(h,'PaperType','A5');
% orient(h1,'Landscape');
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP4',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP4',nindex));
end
hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP4',nindex));

%% The beta's of the FMP's 
% Create the quarterly sorts of H(igh)-M(inus)-L(ow) Book-to-Market
bHML = (1/2)*((bHB + bHS) - (bLB + bLS));
% Create the quarterly sorts of S(mall)-M(inus)-B(ig) on Log-size
bSMB = (1/3)*((bHS + bMS + bLS) - (bHB + bMB + bLB));
% save as time-series
bfmp = fints(nb.dates,[bHML, bSMB],{'HML','SMB'},nb.freq,'Beta');

%% Plot the BETA's 
h = figure;
plot1 = plot(bfmp(mdof+1:end));
set(plot1(1),'DisplayName','HML','LineWidth',1,'LineStyle','-');
set(plot1(2),'DisplayName','SMB','LineWidth',2,'LineStyle','--');
title('Average factor loading portfolio CAPM market factor');
ylabel('\beta');
set(h,'PaperType','A5');
% orient(h1,'Landscape');
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP5',nindex));
end
% print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP5',nindex));
hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_FMP5',nindex));

%% Characteristic based cross-sectional model (BVTP & MV)
%
% The cross-sectional pay-offs to the factors for the universe under
% consideration. 
%
% NB - ensure that the data is lead/laggeg back to where it was and then
% correctly used for forecasting of returns.
clear xs;
reg_type = 'robustfit';
flag_ridge = false;
rp.i = 1;
for k =1:14,
    for i=1:length(ret)
        % get the data
        % EXCESS RETURNS (equivalent to using eret)
        y0 = fts2mat(ret(i))-fts2mat(rfr(i));
        % ACTIVE RETURNS (equivalent to using aret)
        % y1 = fts2mat(ret(i))-fts2mat(factors(i).MKT);
        % SELECT RETURN TYPE
        y = y0;
        % without beta
        switch k
            case 1
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i)))]; % BVTP & MV
            case 2
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA
            case 3
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i))]; % BVTP & MV & CAPM BETA
            case 4
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA
            case 5            
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i))]; % BVTP & MV & CAPM BETA
            case 6
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA
            case 7
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(vol(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA
            case 8
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(vol(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & VOL
            case 9
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(ey(i)); fts2mat(beta0_mkt(i))]; % BVTP & 
            case 10
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(ey(i))]; % BVTP & 
            case 11
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(ey(i)); fts2mat(dy(i))]; % BVTP & MV & CAPM BETA
            case 12
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(dy(i))]; % BVTP & MV & DY
            case 13
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(dy(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA
            case 14
                x = [fts2mat(bvtp(i)); (fts2mat(mv(i))); fts2mat(moml(i)); fts2mat(moms(i)); fts2mat(dy(i)); fts2mat(ey(i)); fts2mat(beta0_mkt(i))]; % BVTP & MV & CAPM BETA      
            otherwise
        end
        % create the correct index
        xind = logical(index(i,:)==1);
        % index the data correctly to the universe
        y = y(xind);
        x = x(:,xind);
        % set all the Infs to NaN
        x(abs(x)==Inf)=NaN;
        % remove NaN data
        nidx = ~(isnan(y) | (sum(isnan(x))>0));
        % we only keep those data point that have both MV and BVTP data
        % (selection effect)
        y=y(nidx);
        x=x(:,nidx);
        % clear outputs
        clear s stats;
        % format the data correctly
        x = transpose(x);
        y = transpose(y);
        % winsorize the data
        if (size(x,1) > size(x,2)+2),  
            % winsorize the data
            x = winsorize(x);
            % z-score data
            x = nanzscore(x);
            switch reg_type
                case 'robustfit'
                    % carry out regressions
                    [beta, stats] = robustfit(x,y);
                case  'ridge'
                    [beta] = ridge(y,x,k,0); % no centering and scaling
                    % generate the regression statistics
                    stats = regstats(y,x);
                    stats.t = stats.tstat.t;
                    stats.p = stats.fstat.pval;
                    stats.se = sqrt(diag(stats.covb));
                otherwise
            end
            if flag_ridge & (k==9),
                % ridge plot regressions
                k0 = 0:0.1:10;
                if (mod(i,freqsamplerate(freq))==0)
                    [rp.beta(:,:,rp.i)] = ridge(y,x,k0);
                    rp.dates(rp.i) = i;
                    rp.i = rp.i+1;
                end;
            end
            % raw model statisics
            s.X    = [ones(size(x,1),1) x];
            s.t    = stats.t;
            s.p    = stats.p;
            s.b    = beta;
            s.se   = stats.se;
            % derived model statistics
            s.y        = y;
            s.yhat     = s.X * s.b;
            s.ybar     = nanmean(y);
            s.resid    = s.y - s.yhat;
            s.sse      = norm(s.resid)^2;
            s.ssr      = norm(s.yhat - s.ybar)^2;
            s.sst      = norm(s.y - s.ybar)^2;
            s.dfe      = length(y) - length(s.b); % # obs - # beta's
            s.dfr      = length(s.b) - 1;
            s.dft      = length(y)-1;
            s.F        = (s.ssr / s.dfr) * (s.sse / s.dfe);
            s.pval     = 1 - fcdf(s.F,s.dfr,s.dfe);
            s.R2       = 1 - s.sse ./ s.sst;
            s.adjR2    = 1 - s.sse ./ s.sst * (s.dft./s.dfe); % for constant term
            % sort the model statistics
            if (all(s.b==0) | all(isnan(s.b==0))), s.b=xs{k}.beta(:,i-1); end; % xero-order hold payoffs
            xs{k}.beta(:,i)       = s.b;
            xs{k}.se(:,i)         = s.se;
            xs{k}.t(:,i)          = s.t;
            xs{k}.adjr2(:,i)      = s.adjR2;
            xs{k}.pval(:,i)       = s.pval;
            xs{k}.r2(:,i)         = s.R2;
        end;
    end;
end;

%% Create the FINTS data structures for the cross-sectional pay-offs
payoffs{1} = fints(nb.dates,[transpose(xs{1}.beta)], {'BIAS','BVTP','MV'},nb.freq,'Payoffs');
payoffs{2} = fints(nb.dates,[transpose(xs{2}.beta)], {'BIAS','BVTP','MV','MKT'},nb.freq,'Payoffs');
payoffs{3} = fints(nb.dates,[transpose(xs{3}.beta)], {'BIAS','BVTP','MV','MOML'},nb.freq,'Payoffs');
payoffs{4} = fints(nb.dates,[transpose(xs{4}.beta)], {'BIAS','BVTP','MV','MOML','MKT'},nb.freq,'Payoffs');
payoffs{5} = fints(nb.dates,[transpose(xs{5}.beta)], {'BIAS','BVTP','MV','MOML','MOMS'},nb.freq,'Payoffs');
payoffs{6} = fints(nb.dates,[transpose(xs{6}.beta)], {'BIAS','BVTP','MV','MOML','MOMS','MKT'},nb.freq,'Payoffs');
payoffs{7} = fints(nb.dates,[transpose(xs{7}.beta)], {'BIAS','BVTP','MV','MOML','MOMS','VOL','MKT'},nb.freq,'Payoffs');
payoffs{8} = fints(nb.dates,[transpose(xs{8}.beta)], {'BIAS','BVTP','MV','VOL','MKT'},nb.freq,'Payoffs');
payoffs{9} = fints(nb.dates,[transpose(xs{9}.beta)], {'BIAS','BVTP','MV','MOML','MOMS','EY','MKT'},nb.freq,'Payoffs');
payoffs{10}= fints(nb.dates,[transpose(xs{10}.beta)],{'BIAS','BVTP','MV','EY'},nb.freq,'Payoffs');
payoffs{11}= fints(nb.dates,[transpose(xs{11}.beta)],{'BIAS','BVTP','MV','MOML','MOMS','EY','DY'},nb.freq,'Payoffs');
payoffs{12}= fints(nb.dates,[transpose(xs{12}.beta)],{'BIAS','BVTP','MV','DY'},nb.freq,'Payoffs');
payoffs{13}= fints(nb.dates,[transpose(xs{13}.beta)],{'BIAS','BVTP','MV','MOML','MOMS','DY','MKT'},nb.freq,'Payoffs');
payoffs{14}= fints(nb.dates,[transpose(xs{14}.beta)],{'BIAS','BVTP','MV','MOML','MOMS','DY','EY','MKT'},nb.freq,'Payoffs');

%% Table 8 - Average Factor Payoffs
%
% The CBM payoffs to each factor is given as the median of the factor
% values of the test period for each different CBM model. This is indicative 
% of the broad payoffs to a given factor. The standard deviation of the factor 
% payoffs is given as is the average t-statistic to a given factor. 
%
hf1 = 6*freqsamplerate(freq)+3;
xis = {'BIAS','BVTP','MV','MKT','MOML','MOMS','EY','DY','VOL'};
fid = fopen(sprintf('table8_%d.txt',nindex),'wt');
clear table8 xi;
table8= NaN * zeros(length(payoffs),length(xis));
% index to characteristics
for i=1:length(payoffs), xi(i,:) = ismemstr(xis,fieldnames(payoffs{i},1)); end;
% write to file
for k=1:3,
    fprintf(fid,'\hline \\\\');
    fprintf(fid,'Model \\# & $\\alpha$ & $\\beta_{_{BVTP}}$ & $\\beta_{_{MV}}$ & $\\beta_{_{MKT}}$ & $\\beta_{_{MOML}}$ & $\\beta_{_{MOMS}}$ & $\\beta_{_{EY}}$ & $\\beta_{_{DY}}$ & $\\beta_{_{VOL}}$ \\\\\n');
    fprintf(fid,'\hline \\\\');
    for i=1:length(payoffs),
        switch k
            case 1
                % medians
                table8(i,find(xi(i,:))) = 100*nanmedian(fts2mat(payoffs{i}(hf1+1:end)));
            case 2
                % mean-absolute-deviations
                table8(i,find(xi(i,:))) = 100*mad(fts2mat(payoffs{i}(hf1+1:end)));
            case 3
                % t-statistics that are greater than 2
                pts = (xs{i}.t(:,hf1+1:end)>2);
                % work out the percentage of significant t-statistics
                table8(i,find(xi(i,:))) = nansum(transpose(pts))./size(pts,2);
            otherwise
        end
        fprintf(fid,'%d & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\\\ \n',i,table8(i,:));
    end;
end;
fclose(fid);

%% Table 9: Explained by Characteristics vs. Unexplained
%
% The sum of the average CBM payoffs to each factor is given over the test 
% period for a given model. This is indicative of how much is explained due
% factor commonalities and how much is unexplained and hence possibly due
% to stock unique factors or market wide factors unrelated to the 
% characteristics.
%
hf = freqsamplerate(freq);
hf1 = 6*freqsamplerate(freq)+3;
fid = fopen(sprintf('table9_%d.txt',nindex),'wt');
% construct the data
fprintf(fid,'CBM Model \\# & Bias & Characteristic & $\\sigma$ (Bias) & $\\sigma$ (Char.) \\\\ \n');
for i=1:length(payoffs),
    % exp{i} = payoffs{i} .* payoffs{i} ./ ((payoffs{i}.^2).^(1/2));
    exp{i} = payoffs{i} .* sign(fts2mat(payoffs{i}));
    exp{i} = exp{i} ./ repmat(sum(fts2mat(exp{i}),2),1,size(exp{i},2));
    x2 = fts2mat(exp{i});
    exp{i} = fints(exp{i}.dates,[x2(:,1) transpose(sum(x2(:,[2:end])'))],{'BIAS','CHAR'},exp{i}.freq,exp{i}.desc);
    fprintf(fid,'Model \\#%d & %3.0f\\%% & %3.0f\\%% & %3.0f\\%% & %3.0f\\%% \\\\ \n',i,100*nanmean(fts2mat(exp{i}(hf1+1:end))),100*nanstd(fts2mat(exp{i}(hf1+1:end))));
end;
fclose(fid);

%% N-tile on HML, SMB and MKT Factors (Factor Based Models)
qt = 3;
% 3-tile sort of MV ->   nm3
nm3 = ntile(transpose(index .* fts2mat(mv)),qt,'descend');
nm3 = fints(mv.dates,transpose(nm3),fieldnames(mv,1),mv.freq,mv.desc);
% 3-tile sort of BVTP -> nb
nb3 = ntile(transpose(index .* fts2mat(bvtp)),qt,'descend');
nb3 = fints(bvtp.dates,transpose(nb3),fieldnames(bvtp,1),bvtp.freq,bvtp.desc);
% 3-tile sort on beta_bias, beta_HML, beta_SMB, beta_MKT
na{1} = ntile(transpose(index .* fts2mat(beta_bias)),qt,'descend');
na{1} = fints(bvtp.dates,transpose(na{1}),fieldnames(bvtp,1),bvtp.freq,'BIAS');
na{2} = ntile(transpose(index .* fts2mat(beta_hml)),qt,'descend');
na{2} = fints(bvtp.dates,transpose(na{2}),fieldnames(bvtp,1),bvtp.freq,'HML');
na{3} = ntile(transpose(index .* fts2mat(beta_smb)),qt,'descend');
na{3} = fints(bvtp.dates,transpose(na{3}),fieldnames(bvtp,1),bvtp.freq,'SMB');
na{4} = ntile(transpose(index .* fts2mat(beta_mkt)),qt,'descend');
na{4} = fints(bvtp.dates,transpose(na{4}),fieldnames(bvtp,1),bvtp.freq,'MKT');

%% Table 3
%
% We provide the average monthly excess returns, the average Book-to-Price
% and the average Market Capitalisation for each of the balanced portfolios
% resulting from the characteristic sorts into Book-to-Price and Market
% Capitalisation and then re-sorted on the factor loading balanced
% portfolio loadings. This is done for sorts on each of the three factor
% loading portfolio \beta's and the bias as estimated from the 3-factor
% multi-factor APT model.
% 
% $$R_{ijk} - R_{_{RFR}} = \alpha +\beta_{_MKT} R_{MKT} + \beta_{_HML} R_{HML} + \beta_{SMB} R_{SMB}$$
%
% For i-th BVTP sorted characteristic, j-th Size sorted characteristics and
% the k-th factor loading portfolio, where i=1,2,3 , j=1,2,3 and k = $\alpha$,
% $\beta_{_{HML}}$, $\beta_{_{SMB}}$, $\beta_{_{MKT}}$.
%
fid1 = fopen(sprintf('table3_%d.txt',nindex),'wt');
% construct the average book-to-price time-series for total
nBVTP = transpose(nanmean(transpose(fts2mat(bvtp))));
% construct the average size time-series for total universe
nMV = transpose(nanmean(transpose(fts2mat(mv))));
% construct the average size time-series for total universe
nDY = transpose(nanmean(transpose(fts2mat(dy))));
% zero normalizations are treated as NaN
nBVTP(nBVTP==0) = NaN;
nMV(nMV==0)     = NaN;
nDY(nDY==0)     = NaN;
% construct the index
for p = 1:4,
    for k=1:qt, % BVTP
        for n=1:qt, % MV
            for m=1:qt, % BETA FACTOR
                % the sort (BVTP, MV and FACTOR)
                FP = (fts2mat(nm3)==n) & (fts2mat(nb3)==k) & (fts2mat(na{p})==m);
                % construct the normalizations
                normFP = transpose(sum(transpose(FP)));
                % construct the (excess) returns for each portfolio
                rFP = transpose(nansum(transpose(eretx .* FP))) ./ normFP;
                % construct the average book-to-price time-series for FP
                rFPBVTP = transpose(nansum(transpose(fts2mat(bvtp) .* FP))) ./ normFP;
                % construct the average size time-series for FP
                rFPMV = transpose(nansum(transpose(fts2mat(mv) .* FP))) ./ normFP;
                % construct the average dividend yield time-series for FP
                rFPDY = transpose(nansum(transpose(fts2mat(dy) .* FP))) ./ normFP;
                % zero returns are treated as NaN
                rFP(rFP==0)         = NaN;
                rFPBVTP(rFPBVTP==0) = NaN;
                rFPMV(rFPMV==0)     = NaN;
                rFPDY(rFPDY==0)     = NaN;
                % populate the return matrix (as monthly percentage)
                kn_data(qt*(k-1)+n+1,m) = nanmean(rFP)*100;
                % populate the return matrix (relative to the total average)
                kn_data(qt*(k-1)+n+1,m+qt) = nanmean(rFPBVTP ./ nBVTP);
                % populate the return matrix (relative to the total average)
                kn_data(qt*(k-1)+n+1,m+2*qt) = nanmean(rFPMV ./ nMV);
                % populate the dividend yield (relative to the total average)
                kn_data(qt*(k-1)+n+1,m+3*qt) = nanmean(rFPDY ./ nDY);
            end
            % LaTeX output of table to file
            fprintf(fid1,'%d & %d & %3.2f & %3.2f & %3.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %3.2f & %3.2f & %3.2f \\\\ \n',k,n,kn_data(qt*(k-1)+n+1,:));
        end
    end
    % LaTeX output of table to file and add on averages
    fprintf(fid1,' & &  %3.2f & %3.2f & %3.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %3.2f & %3.2f & %3.2f \\\\ \n',nanmean(kn_data));
end;
fclose(fid1);

%% TABLE 5 : Characteristic based portfolio regressions.
%
% We provide the factor \beta's to Quintiles 1 through 3 minus for 
% factor loading sorts of the characteristic based sorts on Book-to-Price
% and Size characteristics. The return for each double sorted time-series
% is regressed against the multi-factors APT 
% model. The resulting loadings as provided in terms of excess returns ER:
%
% $$ER_{ij\beta_{k}^1} = \alpha +\beta_{_MKT} R_{MKT} + \beta_{_HML} R_{HML} + \beta_{SMB} ER_{SMB}$$
%
% For i-th BVTP sorted characteristic, j-th Size sorted characteristics and
% the k-th factor loading portfolio, where i=1,2,3 , j=1,2,3. k = $\alpha$,
% $\beta_{_{HML}}$, $\beta_{_{SMB}}$, $\beta_{_{MKT}}$.
% 
fdata = fts2mat(factors);
rdata = fts2mat(rfr);
% >--------------------------------
% Excess returns on market factors
fdata(:,3) = fdata(:,3) - rdata;
% <--------------------------------
fid = fopen(sprintf('table5_a_%d.txt',nindex),'wt');
fid1 = fopen(sprintf('table5_b_%d.txt',nindex),'wt');
% construct the index
for p = 1:4, % FACTORS
    % reset b0 and t0
    b0 = zeros(9,12);
    t0 = b0;
    for k=1:qt, % BVTP
        for n=1:qt, % MV
            % initialize the ydata
            ydata = NaN * ones(size(fdata,1),1);
            for m=1:qt, % BETA FACTOR QUINTILES
                % the sort (BVTP, MV and FACTOR)
                FP = (fts2mat(nm3)==n) & (fts2mat(nb3)==k) & (fts2mat(na{p})==m);
                % construct the normalizations
                normFP = transpose(sum(transpose(FP)));
                % construct the excess returns for each portfolio
                rFP = transpose(nansum(transpose(eretx .* FP))) ./ normFP;
                % >--------------------------------------------------------
                ydata(:,m) = rFP;
                % <--------------------------------------------------------
                % Regression Data
                x = fdata(12:end,:);
                % >----------------------------------
                % Excess returns on the double sort
                y = ydata(12:end,m)-rdata(12:end,1); % High Minus Low on Factor Loading
                % <----------------------------------
                % remove NaN data and only consider overlapping data
                indx = ~isnan(y) & ~transpose(sum(transpose(isnan(x)))>0);
                x = x(indx,:);
                y = y(indx);
                % the regressions
                [beta, stats] = robustfit(x,y);
                % clear statistics structures
                clear s;
                % raw model statisics
                s.X = [ones(size(x,1),1) x];
                s.t    = stats.t;
                s.p    = stats.p;
                s.b    = beta;
                s.se   = stats.se;
                % derived model statistics
                s.y        = y;
                s.yhat     = s.X * s.b;
                s.ybar     = nanmean(y);
                s.resid    = s.y - s.yhat;
                s.sse      = norm(s.resid)^2;
                s.ssr      = norm(s.yhat - s.ybar)^2;
                s.sst      = norm(s.y - s.ybar)^2;
                s.dfe      = length(y) - length(s.b); % # obs - # beta's
                s.dfr      = length(s.b) - 1;
                s.dft      = length(y)-1;
                s.F        = (s.ssr / s.dfr) * (s.sse / s.dfe);
                s.pval     = 1 - fcdf(s.F,s.dfr,s.dfe);
                s.adjr2    = 1 - s.sse ./ s.sst * (s.dft./s.dfe); % for constant term
                % the output for the k-th BTVP and n-th SIZE sorts for
                % loadings quintiles 1 through 3 on the loadings of the
                % p-th factor
                % bias
                b0(qt*(k-1)+n,m) = beta(1);
                t0(qt*(k-1)+n,m) = s.t(1);
                % HML factor
                b0(qt*(k-1)+n,m+3) = beta(2);
                t0(qt*(k-1)+n,m+3) = s.t(2);
                % SMB factor
                b0(qt*(k-1)+n,m+6) = beta(3);
                t0(qt*(k-1)+n,m+6) = s.t(3);
                % MKT factor
                b0(qt*(k-1)+n,m+9) = beta(4);
                t0(qt*(k-1)+n,m+9) = s.t(4);
            end % m - quintiles on the factor loading
            % keep the average
            % LaTeX output of table to file with coefficients
            fprintf(fid,'%d & %d & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f \\\\ \n', ...
                k,n,b0(qt*(k-1)+n,:));
            % LaTeX output of table to file with t-statisics
            fprintf(fid1,'%d & %d & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f \\\\ \n', ...
                k,n,t0(qt*(k-1)+n,:));
        end % n
    end % k   
    % compute the averages
    at0 = nanmean(t0);
    ab0 = nanmean(b0);
    % LaTeX output of table to file writing the average
    fprintf(fid,'\\multicolumn{2}{Average} &  %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f  \\\\ \n',ab0);
    % LaTeX output of table to file writing the average
    fprintf(fid1,'\\multicolumn{2}{Average} &  %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f  \\\\ \n',at0);    
end % p
% close file
fclose(fid);
fclose(fid1);

%% Dividend Yield, Size and Book-to-Price
%
% To confirm that larger companies are in fact paying smaller dividends
% in the universe but that there is a difference between the size of the
% universe.
%
clear x1 y1 x2 y2;
dy1   = index .* fts2mat(dy);
mv1   = index .* fts2mat(mv); % log mv
bvtp1 = index .* fts2mat(bvtp);
% full market
% ---------------------------
dy2 = dy1;
mv2 = mv1;
% rescale the market value by the average and rebased 
nmv2 = nansum(transpose(mv2));
mv2 = 100 * mv2 ./ repmat(transpose(nmv2),1,size(mv1,2));
% construct the averages
dy2 = nanmedian(dy2);
dy2(dy2==0)=NaN;
mv2 = nanmedian(mv2);
% remove the NaN values
id2 = isnan(mv2) | isnan(dy2);
dy2 = dy2(~id2);
mv2 = mv2(~id2);
% winsorize dividend yields
dy2 = transpose(winsorize(dy2(:),3.5));
% the given universe
index1 = (nansum(index)>0);
% create the histograms
[hmv2,x2] = hist(mv2,30);
for i=1:length(x2), 
    x2a = [0 x2]; 
    y2(i) = nanmean(dy2(find((mv2>=x2a(i))&(mv2<x2a(i+1))))); 
end;

%% Plot the average dividend yield against size
h1 = figure;
c = ['k' 'r'];
mt ={'s','o'};
ls ={'--','-'};
clear s h;
brob = robustfit(x2,y2);
h(1)= bar(x2,hmv2,'y');
legend('# of stocks');
grid on; hold on
h(2)= plot(x2,brob(1)+brob(2)*x2,'k--','LineWidth',2);
s(2)=scatter(x2,y2,'filled','ks'); 
title(sprintf('Dividend yield differential in Top %d',nindex))
ylabel('Median Dividend Yield');
xlabel('Log Size');
set(h1,'PaperType','A5');
% orient(h1,'Landscape');
% print(h1,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_DivSize',nindex));
end
hgsave(h1,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_DivSize',nindex));

%% TABLE 6 : Characteristic based portfolio regressions.
%
% We provide the factor \beta's to the 1-st Quintile minus 3rd-Quintile 
% factor loading sorts of the characteristic based sorts on Book-to-Price
% and Size characteristics. That is the difference between the 
% excess return on the highest factor loading portfolio is subtracted
% from the lowest factor loading portfolio from the characteristic 
% sorted portfolio resulting from the highest Book-to-Price and highest
% Size loadings, this is then regressed against the multi-factors APT 
% model. The resulting loadings as provided.
%
% $$(R_{ij\beta_{k}^1} - R_{ij\beta_{k}^3}) = \alpha +\beta_{_MKT} R_{MKT} + \beta_{_HML} R_{HML} + \beta_{SMB} R_{SMB}$$
%
% For i-th BVTP sorted characteristic, j-th Size sorted characteristics and
% the k-th factor loading portfolio, where i=1,2,3 , j=1,2,3. k = $\alpha$,
% $\beta_{_{HML}}$, $\beta_{_{SMB}}$, $\beta_{_{MKT}}$.
% 
fdata = fts2mat(factors);
fid = fopen(sprintf('table6_%d.txt',nindex),'wt');
% construct the index
for p = 1:4,
    % initialize the combined portfolio
    cy = [];
    % initialize the combined index
    cindx = logical(ones(size(fdata(12:end,:),1),1));
    for k=1:qt, % BVTP
        for n=1:qt, % MV   
            % initialize the ydata
            ydata = NaN * ones(size(fdata,1),1);
            for m=1:qt, % BETA FACTOR (balanced fund)
                % the sort (BVTP, MV and FACTOR)
                FP = (fts2mat(nm3)==n) & (fts2mat(nb3)==k) & (fts2mat(na{p})==m);
                % construct the normalizations
                normFP = transpose(sum(transpose(FP)));
                % construct the excess returns for each portfolio
                rFP = transpose(nansum(transpose(eretx .* FP))) ./ normFP;
                % >--------------------------------------------------------
                ydata(:,m) = rFP;
                % <--------------------------------------------------------
            end; % m
            % Regression Data
            % -------------------------
            y = ydata(12:end,1)-ydata(12:end,3); % High Minus Low on Factor Loading
            x = fdata(12:end,:); % here because of loop dependent data range 
            % ---------------------------
            % update the combined data
            % ------------------------
            cy = [cy, y(:)]; % dependent
            cx = x; % here because of loop dependent data range
            % ------------------------
            % remove NaN data and only consider overlapping data
            indx = ~isnan(y) & ~transpose(sum(transpose(isnan(x)))>0);
            x = x(indx,:);
            y = y(indx);
            % update the combined index
            % -------------------------
            cindx = cindx & indx;
            % -------------------------
            % the regressions
            [beta, stats] = robustfit(x,y);
            % clear statistics structures
            clear s;
            % raw model statisics
            s.X = [ones(size(x,1),1) x];
            s.t    = stats.t;
            s.p    = stats.p;
            s.b    = beta;
            s.se   = stats.se;
            % derived model statistics
            s.y        = y;
            s.yhat     = s.X * s.b;
            s.ybar     = nanmean(y);
            s.resid    = s.y - s.yhat;
            s.sse      = norm(s.resid)^2;
            s.ssr      = norm(s.yhat - s.ybar)^2;
            s.sst      = norm(s.y - s.ybar)^2;
            s.dfe      = length(y) - length(s.b); % # obs - # beta's
            s.dfr      = length(s.b) - 1;
            s.dft      = length(y)-1;
            s.F        = (s.ssr / s.dfr) * (s.sse / s.dfe);
            s.pval     = 1 - fcdf(s.F,s.dfr,s.dfe);
            s.r2       = 1 - s.sse ./ s.sst;
            s.adjr2    = 1 - s.sse ./ s.sst * (s.dft./s.dfe); % for constant term
            % LaTeX output of table to file
            fprintf(fid,'%d & %d & %3.1f & %4.3f (%3.1f) & %4.3f (%3.1f) & %4.3f (%3.1f) & %4.3f (%3.1f) & %2.1f \\\\ \n', ...
                          k,n,100*s.ybar,beta(1),s.t(1),beta(2),s.t(2),beta(3),s.t(3),beta(4),s.t(4),s.r2 * 100);
        end % size characteristic n
    end % bvtp characteristic k
    % Shared Not NaN data
    cy = transpose(nanmean(transpose(cy)));
    % the regressions for the combined portfolio
    [beta, stats] = robustfit(cx,cy);
    % clear statistics structures
    clear s;
    % raw model statisics
    s.X = [ones(size(cx,1),1) cx];
    s.t    = stats.t;
    s.p    = stats.p;
    s.b    = beta;
    s.se   = stats.se;
    % derived model statistics
    s.y        = cy;
    s.yhat     = s.X * s.b;
    s.ybar     = nanmean(cy);
    s.resid    = s.y - s.yhat;
    s.sse      = norm(s.resid)^2;
    s.ssr      = norm(s.yhat - s.ybar)^2;
    s.sst      = norm(s.y - s.ybar)^2;
    s.dfe      = length(cy) - length(s.b); % # obs - # beta's
    s.dfr      = length(s.b) - 1;
    s.dft      = length(cy)-1;
    s.F        = (s.ssr / s.dfr) * (s.sse / s.dfe);
    s.pval     = 1 - fcdf(s.F,s.dfr,s.dfe);
    s.r2       = 1 - s.sse ./ s.sst;
    s.adjr2    = 1 - s.sse ./ s.sst * (s.dft./s.dfe); % for constant term
    % LaTeX for the combined portfolios
    fprintf(fid,'\\hline \n');
    fprintf(fid,'\\multicolumn{2}{c}{Comb. Port.} & %3.1f & %4.3f (%3.1f) & %4.3f (%3.1f) & %4.3f (%3.1f) & %4.3f (%3.1f) & %2.1f \\\\ \n', ...
        100*s.ybar,beta(1),s.t(1),beta(2),s.t(2),beta(3),s.t(3),beta(4),s.t(4),s.r2 * 100);
    fprintf(fid,'\\hline \n');
end % p factors
% close file
fclose(fid);

%% Characteristic based portfolio's
%
% Following the Haugen & Baker type approach we average the payoffs over
% the preceeding 12 months using a MA(12,1) filter. 
%
% NEED TO BE VERY CAREFUL ABOUT THE LEAD/LAG DATES!
%
clear qr erf wtfts qx qm qs;
% 2. fill in the factors using zero-order hold
fbvtp        = fillts(bvtp,'z');
fmv          = fillts(mv,'z');
fmoml        = inf2nan(fillts(moml,'z'));
fmoms        = inf2nan(fillts(moms,'z'));
fbeta0_mkt   = fillts(beta0_mkt,'z');
fey          = fillts(ey,'z');
fdy          = fillts(dy,'z');
tickers      = fieldnames(bvtp,1);
% 3. Create the time-series of expected alpha's
klist = [1 2 5 6 9 13 14 15 16];
qn = 5;
% pre-allocate alpha
alpha = NaN * ones(size(bvtp));
for k = klist
    % 1. Create the beta expectations from the beta's (payoffs) and factors
    switch k
        case {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
            erf = tsmovavg(payoffs{k},'s',12); % 1
        otherwise
    end
    % compute the expected excess returns
    for t=12:length(erf)
        switch k
            case {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                % without beta
                switch k
                    case 1
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t)))]; % BVTP & MV
                    case 2
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    case 3
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t))]; % BVTP & MV & CAPM BETA
                    case 4
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    case 5
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t))]; % BVTP & MV & CAPM BETA
                    case 6
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    case 7
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(vol(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    case 8
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(vol(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & VOL
                    case 9
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(ey(t)); fts2mat(beta0_mkt(t))]; % BVTP &
                    case 10
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(ey(t))]; % BVTP &
                    case 11
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(ey(t)); fts2mat(dy(t))]; % BVTP & MV & CAPM BETA
                    case 12
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(dy(t))]; % BVTP & MV & DY
                    case 13
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(dy(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    case 14
                        x = [fts2mat(bvtp(t)); (fts2mat(mv(t))); fts2mat(moml(t)); fts2mat(moms(t)); fts2mat(dy(t)); fts2mat(ey(t)); fts2mat(beta0_mkt(t))]; % BVTP & MV & CAPM BETA
                    otherwise
                end
                % create the correct index
                xind = logical(index(i,:)==1);
                % index the data correctly to the universe
                x = x(:,xind);
                % z-score the data
                x=nanzscore_mat(x);
                % compute the expected returns
                % ------------------------------------------------------
                alpha(t+1,xind) = fts2mat(erf(t)) * [ones(1,size(x,2)); x];
                % ------------------------------------------------------
            case 15
                % CAPM (MKT) *
                alpha0 = fts2mat(alpha0_mkt(t));
                beta0  = fts2mat(beta0_mkt(t));
                % ------------------------------------------------------
                alpha(t+1,xind) = alpha0(:,xind) + beta0(:,xind) * fts2mat(factors.MKT(t));
                % ------------------------------------------------------
            case 16
                % APT (HML, SMB, MKT) *
                bias = fts2mat(beta_bias(t));
                bsmb = fts2mat(beta_smb(t));
                bhml = fts2mat(beta_hml(t));
                bmkt = fts2mat(beta_mkt(t));
                % ------------------------------------------------------
                alpha(t+1,xind) = bias(:,xind) +  bhml(:,xind) * fts2mat(factors.HML(t)) + bhml(:,xind) * fts2mat(factors.SMB(t))+ bmkt(:,xind) * fts2mat(factors.MKT(t));
                % ------------------------------------------------------
            otherwise
        end
    end % t
    % create FINTS
    erfts{k} = fints(erf.dates,alpha(2:end,:),fieldnames(bvtp,1),bvtp.freq,'Expected Returns');
    % correct the dates to t+1
    erfts{k} = lagts(erfts{k});
    % create the ntile [on expected returns]
    [nt1{k}]= ntile(transpose(index .* fts2mat(erfts{k})),qn,'descend');
    % create the ntile
    wtfts{k} = fints(erfts{k}.dates,transpose(nt1{k}),fieldnames(bvtp,1),bvtp.freq,'Weights');
end % k

%% Create quintiles of equally weighted (balanced) portfolio's
% quintiles 1 & 2 - 4 & 5
clear qindex qm qs qr qx qb;
ret = zero2nan(ret);
ret = inf2nan(ret);
for k=klist
    for j=1:qn, % quintiles
        % find the quintile index
        qindex = (fts2mat(wtfts{k})==j);
        qindex = qindex ./ repmat(transpose(nansum(transpose(qindex))),1,size(qindex,2));
        % j = 
        qr(:,j,k) = transpose(nansum(transpose(fts2mat(ret) .* qindex)));
        qb0(:,j,k) = transpose(nansum(transpose(fts2mat(beta0_mkt) .* qindex)));
    end
     % The means and variances of each quintile
     qm(:,k) = freqsamplerate(freq) * nanmean(qr(76:end,:,k));
     qs(:,k) = sqrt(freqsamplerate(freq)) * nanstd(qr(76:end,:,k));
     qb(:,k) = nanmean(qb0(76:end,:,k));
     qx(:,k) = 1:qn;
end

%% Plot pre- and post- formation date dynamics of CBM models

%% CMB vs. APT : Plot returns against quintiles
h1 = figure; 
c = ['b' 'k' 'g' 'r'];
n = [2 6 15 16];
lw =[1 2 1 2];
mt ={'s','o','<','>'};
ls ={'--','-','-','--'};
clear s;
for j=1:length(n)
    x = qx(:,n(j));
    y = qm(:,n(j));
    % scatter(x,y);
    brob = robustfit(x,y);
    s(j)=scatter(x,y,'filled',sprintf('%s%s',c(j),mt{j})); 
    grid on; 
    hold on
    h(j)=plot(x,brob(1)+brob(2)*x,sprintf('%s%s',ls{j},c(j)),'LineWidth',lw(j));
end
set(gca,'XTick',x);
title(sprintf('Quintile Performance - Top %d',nindex));
ylabel('Annualized Return');
xlabel('Quintile');
legend([s h],'CBM #2','CBM #6','CAPM', 'APT','CBM #2 line','CBM #6 line','CAPM line', 'APT line','location','SouthWest');
% legend(h,'CBM #2(\alpha+ \beta_{BVTP} BVTP_i + \beta_{MV} MV_i + \beta_{MKT} \beta_i)',  ...
%         'CBM #6(\alpha+ \beta_{BVTP} BVTP_i + \beta_{MV} MV_i + \beta_{MOM} MOM_i + \beta_{MKT} \beta_i)', ...
%         'CAPM (\alpha^i + \beta^i r_{MKT})', ...
%         'APT ( \alpha^i + \beta^i_{HML} r_{HML} + \beta^i_{SMB} r_{SMB} + \beta^i r_{MKT} )','location','SouthWest');
set(h1,'PaperType','A5');
% orient(h1,'Landscape');
% print(h1,'-depsc','tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_mean',nindex));
end
hgsave(h1,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_mean',nindex));

%% CBM vs. APT : Volatilities against quintiles
h1 = figure;
for j=1:length(n)
    x = qx(:,n(j));
    y = qs(:,n(j));
    % scatter(x,y);
    brob = robustfit(x,y);
    s(j)=scatter(x,y,'filled',sprintf('%s%s',c(j),mt{j})); 
    grid on; 
    hold on;
    h(j)=plot(x,brob(1)+brob(2)*x,sprintf('%s%s',ls{j},c(j)),'LineWidth',lw(j));
end
set(gca,'XTick',x);
title(sprintf('Quintile Volatility - Top %d',nindex));
ylabel('Annualized Volatility');
xlabel('Quintile');
legend([s h],'CBM #2','CBM #6','CAPM', 'APT','CBM #2 line','CBM #6 line','CAPM line', 'APT line','location','NorthWest');
set(h1,'PaperType','A5');
% orient(h1,'Landscape');
% print(h1,'-depsc','tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_vol',nindex));
end
hgsave(h1,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_vol',nindex));
% best fit lines to the data

%% CBM vs. APT : beta's against quintiles
h1 = figure;
for j=1:length(n)
    x = qx(:,n(j));
    y = qb(:,n(j));
    % scatter(x,y);
    brob = robustfit(x,y);
    s(j)=scatter(x,y,'filled',sprintf('%s%s',c(j),mt{j})); 
    grid on; 
    hold on
    h(j)=plot(x,brob(1)+brob(2)*x,sprintf('%s%s',ls{j},c(j)),'LineWidth',lw(j));
end
set(gca,'XTick',x);
title(sprintf('Quintile Market Loadings - Top %d',nindex));
ylabel('Market \beta');
xlabel('Quintile');
legend([s h],'CBM #2','CBM #6','CAPM', 'APT','CBM #2 line','CBM #6 line','CAPM line', 'APT line','location','Best');
set(h1,'PaperType','A5');
% orient(h1,'Landscape');;
% print(h1,'-depsc','tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_beta',nindex));
end
hgsave(h1,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT_beta',nindex));
% best fit lines to the data

%% Plot the expected pay-offs (EX loc)
h1 = figure;
erfp = erf(hf1+1:end);
x = fts2mat(erfp);
items = fieldnames(erfp,1);
hline1 = plot(erfp.dates,x(:,2:end));
items = items(2:end);
set(gca,'XTick',get(gca,'XTick'),'XTickLabel',datestr(get(gca,'XTick')));
title(sprintf('Expected characteristic payoffs for the Top %d',nindex));
xlabel(sprintf('Datetime',freqstr(freqnum(freq))));
ylabel('Expected Payoff');
% set(hline1(1),'DisplayName',items{1},'LineWidth',2,'LineStyle','-.'); % BIAS
set(hline1(1),'DisplayName',items{1}); % BVTP
set(hline1(2),'DisplayName',items{2},'LineWidth',1,'LineStyle','-.'); % MV
set(hline1(3),'DisplayName',items{3},'LineStyle','--'); % MOML
set(hline1(4),'DisplayName',items{4},'LineWidth',2,'LineStyle','--'); % MOMS
set(hline1(5),'DisplayName',items{5},'LineWidth',2); % DY
set(hline1(6),'DisplayName',items{6},'LineStyle',':'); % EY
set(hline1(7),'DisplayName',items{7},'LineWidth',2,'LineStyle',':'); % MKT
legend(hline1,items);
grid on;
set(h1,'PaperType','A5');
% orient(h1,'Landscape');
% print(h1,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMvsAPT',nindex));
pt_type = {'-dps2','-deps2','-dpng'};
for pt=1:3
    print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_Payoffs',nindex));
end
hgsave(h1,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_Payoffs',nindex));

%% Create the pre- through post- formation date indices for CBM
clear CBM rCBMf;
ret = zero2nan(ret);
ret = inf2nan(ret);
% quintiles 1 and 2 minus 4 and 5
qwts = [1 1 0 -1 -1];
for k=klist
    CBM{k}=zeros(size(wtfts{k}));
    for j=1:qn, % quintiles
        % find the quintile index
        windex = (fts2mat(wtfts{k})==j);
        wnorm  = transpose(nansum(transpose(windex)));
        windex = qwts(j) * (windex ./ repmat(wnorm,1,size(windex,2)));
        CBM{k} = CBM{k} + windex;
    end
    ct = 24;
    % forward looking P(t-L)/P(t-1-L) * Index(t) for  L = 0 to 12
    for i=1:ct
        % LAGTS  -->   Date #2  Data #1
        %              Date #3  Data #2
        % create the lag
        % 1.  Correct for the 1 Quarter lag in BVTP and MV to ensure that
        %     the formation data is set to zero at the same point as that for
        %     the return date. Note that this means that BVTP(t) and MV(t)
        %     line up with the return from P(t)/P(t-1). The actual HML and SMB
        %     portfolio as based on a 1 quarter lag in this paper - that is
        %     the BVTP and MV numbers 1 quarter prior to the date at which the
        %     return are computed. Zero is at Lag0 = 1 Quarter
        % 2.  The lag is then iteratively increased from this point. (i)
        %
        retxf = lagts(ret,i-1+freqsamplerate(freq)/4);
        retxf = fts2mat(retxf);
        % remove inf values
        retxf((abs(retxf)==Inf))=NaN;
        % remove the initial value
        retxf(:,1) = 0;
        % compute the returns
        rCBMf{k}(:,i) = transpose(nansum(transpose(retxf .* CBM{k})));
    end
    % backward looking P(t-L)/P(t-1-L) * Index(t) for L = 0 to -12
    for i=1:ct
        % LEADTS  -->  Date #2  Data #3
        %              Date #3  Data #4
        % create the leadts
        % 1.  Correct for the 1 Quarter lag in BVTP and MV to ensure that
        %     the formation data is set to zero at the same point as that for
        %     the return date. Note that this means that BVTP(t) and MV(t)
        %     line up with the return from P(t)/P(t-1). The actual HML and SMB
        %     portfolio as based on a 1 quarter lag in this paper - that is
        %     the BVTP and MV numbers 1 quarter prior to the date at which the
        %     return are computed. Zero is at Lag0 = 1 Quarter
        % 2.  The lag is then iteratively increased from this point. (i)
        %
        retxf = leadts(ret,i-1+freqsamplerate(freq)/4);
        retxf = fts2mat(retxf);
        % remove inf values
        retxf((abs(retxf)==Inf))=NaN;
        % remove the initial value
        retxf(:,1) = 0;
        % compute the returns
        rCBMf{k}(:,ct+i) = transpose(nansum(transpose(retxf .* CBM{k})));
    end
end

%% Plot CBM Pre- and Post- Formation date dynamics
times= [-(ct-1):0,0:(ct-1)];
nlist = [2 6 13 14 15 16]
nk = length(nlist);
for k=1:3:nk-2,  
    h = figure;
    subplot(3,1,1);
    [haxes, hline1,hline2] = plotyy(times,nanmean(rCBMf{nlist(k)}),times,nanstd(rCBMf{nlist(k)}));
    title(sprintf('Top %d Model # %d %s',nindex,nlist(k),'pre-formation and post-formation behaviour'));
    xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
    set(hline1,'LineWidth',2,'LineStyle','--');
    set(hline2,'LineWidth',1,'LineStyle','-');
    ylabel(haxes(1),'Return R_t[r]');
    ylabel(haxes(2),'Volatility \sigma(R_t)');
    legend([hline1; hline2],'Return','Volatility');
    subplot(3,1,2);
    [haxes, hline1,hline2] = plotyy(times,nanmean(rCBMf{nlist(k+1)}),times,nanstd(rCBMf{nlist(k+1)}));
    title(sprintf('Top %d Model # %d %s',nindex,nlist(k+1),'pre-formation and post-formation behaviour'));
    xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
    set(hline1,'LineWidth',2,'LineStyle','--');
    set(hline2,'LineWidth',1,'LineStyle','-');
    ylabel(haxes(1),'Return R_t[r]');
    ylabel(haxes(2),'Volatility \sigma(R_t)');
    legend([hline1; hline2],'Return','Volatility');
    subplot(3,1,3);
    [haxes, hline1,hline2] = plotyy(times,nanmean(rCBMf{nlist(k+2)}),times,nanstd(rCBMf{nlist(k+2)}));
    title(sprintf('Top %d Model # %d %s',nindex,nlist(k+2),'pre-formation and post-formation behaviour'));
    xlabel(sprintf('Formation Delay (%s)',freqstr(freqnum(freq))));
    ylabel(haxes(1),'Return R_t[r]');
    ylabel(haxes(2),'Volatility \sigma(R_t)');
    set(hline1,'LineWidth',2,'LineStyle','--');
    set(hline2,'LineWidth',1,'LineStyle','-');
    legend([hline1; hline2],'Return','Volatility');
    set(h,'PaperType','A5');
    % orient(h1,'Landscape');
    pt_type = {'-dps2','-deps2','-dpng'};
    for pt=1:3
        print(h1,pt_type{pt},sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMPPF%d',nindex,k));
    end
    % print(h,'-depsc','-tiff',sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_PPF1',nindex));
    hgsave(h,sprintf('EPC_HML_SMB_J203_FMP_R2007a_%d_CBMPPF%d',nindex,k));
end;
% EOF
 



