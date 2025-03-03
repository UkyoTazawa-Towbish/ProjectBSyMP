function [P,P_test,Q_Full,Q_BSyMP,Q_BMR] = Main_RL(SaveDirParent,RuleID)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Main routine of the blind source separation task.
%
% Input
% SaveDirParent:	Parent directory where the results are saved with 
%					subdirectories
% RuleID:			Natural number (1,2,3) that designates rule for task
% 
% Output
% P:			Info of outer world (truth), used for training sessions
% P_test:		Info of outer world (truth), used for test sessions       
% Q_Full:		Beliefs of full model agent
% Q_BSyMP:		Beliefs of BSyMP agent
% Q_BMR:		Beliefs of BMR agent
%__________________________________________________________________________

close all;


%% Resources
ShowProgress = false;		% show progress of the main routine or not
ShowProgressBatch = false;	% show progress of each session or not
FlagSave = false;			% save mat-files or not
FlagVisualize = false;		% make figures of the results or not
FlagSavePng = false;			% save figures as png-files or not
FlagSaveFig = false;		% save figures as fig-files or not
if nargin<1; SaveDirParent = pwd; end
if nargin<2; RuleID = 1; end

BatchSize = 100;			% time point length of a training session
nSessions = 100;			% number of training sessions
T = BatchSize*nSessions;	% total time points
OnProb = 0.9;				% P(o=1|s=1) for pixels on shape
OffProb = 0.1;				% P(o=1|s=0) for pixels on shape
FlagFigOfTask = false;		% make figures of the task or not
lenTest = 1000;				% time point length of a test session


%% Generate data, and Run optimization: 1st session
P = GenerateTask_RL(BatchSize,OnProb,OffProb,RuleID,FlagFigOfTask,ShowProgressBatch);

% Prepare priors
%--------------------------------------------------------------------------
eps     = 0.01;			% bias for prior of parameters, must be in [0,1]
amp		= 100;			% amplitude for prior of parameters
FlagInformative = true;	% make prior imformative or not
qa_init = setPriorA4RL(eps,eps*40,amp*1000,FlagInformative);
qb_init = setPriorB(eps,amp);
qd_init = setPriorD(eps,amp);

qc_init = 0.9;			% base value for prior of C
qc_eps = 0.05;			% amplitude of fluctuation for prior of C
[qcA_init,qcB_init] = setPriorC(qc_init,qc_eps);

% Run optimization and pack the results in Q
%--------------------------------------------------------------------------
Q_Full = RunOptimization_RL(P,qa_init,qb_init,qd_init,false,false,ShowProgressBatch);
Q_BSyMP = RunOptimization_RL(P,qa_init,qb_init,qd_init,qcA_init,qcB_init,ShowProgressBatch);

% Preallocate nSessions*1 structure arrays for the following steps
%--------------------------------------------------------------------------
P = PreallocateStruct(P,nSessions);
Q_Full = PreallocateStruct(Q_Full,nSessions);
Q_BSyMP = PreallocateStruct(Q_BSyMP,nSessions);


%% Generate data, and Run optimization: subsequent session
if ShowProgress
	wb = waitbar(0,'Optimizing Q Full and Q BSyMP...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end
for n=2:nSessions
	if ShowProgress; waitbar(n/nSessions,wb); end
	P(n) = GenerateTask_RL(BatchSize,OnProb,OffProb,RuleID,false,ShowProgressBatch);

	Q_Full(n) = RunOptimization_RL(P(n),Q_Full(n-1).a,Q_Full(n-1).b,Q_Full(n-1).d,false,false,ShowProgressBatch);
	Q_BSyMP(n) = RunOptimization_RL(P(n),Q_BSyMP(n-1).a,Q_BSyMP(n-1).b,Q_BSyMP(n-1).d,Q_BSyMP(n-1).cA,Q_BSyMP(n-1).cB,ShowProgressBatch);
end
if ShowProgress; close(wb); end


%% Apply BMR
MagnA = 1000000;			% magnification for posterior matrix a
MagnB = 1000000;			% magnification for posterior matrix b
MagnD = 1000;				% magnification for posterior matrix d
threshA_post = 0;			% reduction threshold for posterior matrix a
threshB_post = 0;			% reduction threshold for posterior matrix b
threshD_post = 0;			% reduction threshold for posterior matrix d
Q_BMR = BMR_RL(P,Q_Full,MagnA,MagnB,MagnD,threshA_post,threshB_post,threshD_post,ShowProgress);


%% Calculate free-energy
Q_Full(end).F = CalcFreeEnergy(Q_Full,P);
Q_BSyMP(end).F = CalcFreeEnergy(Q_BSyMP,P);
Q_BMR(end).F = CalcFreeEnergy(Q_BMR,P);


%% Test session
P_test = GenerateTask_RL(lenTest,OnProb,OffProb,RuleID,false,ShowProgress);
[Q_Full(end).sTest,Q_Full(end).sPredTest] = EvalPerformance_RL(Q_Full(end),P_test,ShowProgress);
[Q_BSyMP(end).sTest,Q_BSyMP(end).sPredTest] = EvalPerformance_RL(Q_BSyMP(end),P_test,ShowProgress);
[Q_BMR(end).sTest,Q_BMR(end).sPredTest] = EvalPerformance_RL(Q_BMR(end),P_test,ShowProgress);


%% Save the results as .mat file
if FlagSave || FlagSavePng || FlagSaveFig
	SaviDir = NumberSaveDir(fullfile(SaveDirParent,['RL_BatchSize',num2str(BatchSize),'_Session',num2str(nSessions),'_On',num2str(OnProb),'_Off',num2str(OffProb),'_FlagInfo',num2str(FlagInformative),'_qcInit',num2str(qc_init),'_Simulation']));
end
if FlagSave
	save(fullfile(SaviDir,'P.mat'),'P','-v7.3'); save(fullfile(SaviDir,'P_test.mat'),'P_test','-v7.3');
	save(fullfile(SaviDir,'Q_Full.mat'),'Q_Full','-v7.3'); save(fullfile(SaviDir,'Q_BSyMP.mat'),'Q_BSyMP','-v7.3'); save(fullfile(SaviDir,'Q_BMR.mat'),'Q_BMR','-v7.3');
end


%% Visualize the results
if FlagVisualize
	set(0,'units','pixels'); ScreenSize = get(0,'ScreenSize'); fullscreen = [0,0,ScreenSize(3:4)];
	
	nScreen = 3;

	%%------Timecourse of qs------%%
	for screen=1:nScreen
		figure('Position',fullscreen); tile = tiledlayout(P(1).Ns/nScreen,3); title(tile,['Timecourse of qs, screen ',num2str(screen)]);
		stateToAdd = (screen-1)*P(1).Ns/nScreen; % number to add to shift screen of interest
		nexttile; plot_qs(Q_Full,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; plot_qs(Q_BSyMP,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; plot_qs(Q_BMR,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; plot_qs(Q_Full,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; plot_qs(Q_BSyMP,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; plot_qs(Q_BMR,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; plot_qs(Q_Full,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; plot_qs(Q_BSyMP,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; plot_qs(Q_BMR,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; plot_qs(Q_Full,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; plot_qs(Q_BSyMP,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; plot_qs(Q_BMR,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_BMR(1).Label]);
		if FlagSavePng; saveas(gcf,fullfile(SaviDir,['qs_screen',num2str(screen),'.png'])); end
		if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,['qs_screen',num2str(screen),'.fig'])); catch; disp(['saving qs_screen',num2str(screen),'.fig failed.']); end; end
	end

	%%------Timecourse of qsPred------%%
	for screen=1:nScreen
		figure('Position',fullscreen); tile = tiledlayout(P(1).Ns/nScreen,3); title(tile,['Timecourse of qsPred, screen ',num2str(screen)]);
		stateToAdd = (screen-1)*P(1).Ns/nScreen; % number to add to shift screen of interest
		nexttile; plot_qs(Q_Full,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_Full(1).Label],true);
		nexttile; plot_qs(Q_BSyMP,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_BSyMP(1).Label],true);
		nexttile; plot_qs(Q_BMR,1+stateToAdd,['state ',num2str(1+stateToAdd),' ',Q_BMR(1).Label],true);
		nexttile; plot_qs(Q_Full,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_Full(1).Label],true);
		nexttile; plot_qs(Q_BSyMP,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_BSyMP(1).Label],true);
		nexttile; plot_qs(Q_BMR,2+stateToAdd,['state ',num2str(2+stateToAdd),' ',Q_BMR(1).Label],true);
		nexttile; plot_qs(Q_Full,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_Full(1).Label],true);
		nexttile; plot_qs(Q_BSyMP,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_BSyMP(1).Label],true);
		nexttile; plot_qs(Q_BMR,3+stateToAdd,['state ',num2str(3+stateToAdd),' ',Q_BMR(1).Label],true);
		nexttile; plot_qs(Q_Full,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_Full(1).Label],true);
		nexttile; plot_qs(Q_BSyMP,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_BSyMP(1).Label],true);
		nexttile; plot_qs(Q_BMR,4+stateToAdd,['state ',num2str(4+stateToAdd),' ',Q_BMR(1).Label],true);
		if FlagSavePng; saveas(gcf,fullfile(SaviDir,['qsPred_screen',num2str(screen),'.png'])); end
		if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,['qsPred_screen',num2str(screen),'.fig'])); catch; disp(['saving qsPred_screen',num2str(screen),'.fig failed.']); end; end
	end

	%%------sProb vs qsPred------%%
	for screen=1:nScreen
		figure('Position',fullscreen); tile = tiledlayout(P(1).Ns/nScreen,3); title(tile,['Timecourse of sProb vs qsPred, screen ',num2str(screen)]);
		stateToAdd = (screen-1)*P(1).Ns/nScreen; % number to add to shift screen of interest
		SoI = round(0.9*length(P)+1):length(P); % Session of Interest
		nexttile; vsPlotProbPred(Q_Full,1+stateToAdd,SoI,['state ',num2str(1+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; vsPlotProbPred(Q_BSyMP,1+stateToAdd,SoI,['state ',num2str(1+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; vsPlotProbPred(Q_BMR,1+stateToAdd,SoI,['state ',num2str(1+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; vsPlotProbPred(Q_Full,2+stateToAdd,SoI,['state ',num2str(2+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; vsPlotProbPred(Q_BSyMP,2+stateToAdd,SoI,['state ',num2str(2+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; vsPlotProbPred(Q_BMR,2+stateToAdd,SoI,['state ',num2str(2+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; vsPlotProbPred(Q_Full,3+stateToAdd,SoI,['state ',num2str(3+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; vsPlotProbPred(Q_BSyMP,3+stateToAdd,SoI,['state ',num2str(3+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; vsPlotProbPred(Q_BMR,3+stateToAdd,SoI,['state ',num2str(3+stateToAdd),' ',Q_BMR(1).Label]);
		nexttile; vsPlotProbPred(Q_Full,4+stateToAdd,SoI,['state ',num2str(4+stateToAdd),' ',Q_Full(1).Label]);
		nexttile; vsPlotProbPred(Q_BSyMP,4+stateToAdd,SoI,['state ',num2str(4+stateToAdd),' ',Q_BSyMP(1).Label]);
		nexttile; vsPlotProbPred(Q_BMR,4+stateToAdd,SoI,['state ',num2str(4+stateToAdd),' ',Q_BMR(1).Label]);
		if FlagSavePng; saveas(gcf,fullfile(SaviDir,['sProb_vs_qsPred_screen',num2str(screen),'.png'])); end
		if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,['sProb_vs_qsPred_screen',num2str(screen),'.fig'])); catch; disp(['saving sProb_vs_qsPred_screen',num2str(screen),'.fig failed.']); end; end
	end

	%%------Timecourse of qA------%%
	figure('Position',fullscreen); tile = tiledlayout(2,3); title(tile,['Timecourse of qA, screen ',num2str(screen)]);

	qA_Full = NormalizeArray(Q_Full,'a');
	ApowC = arrayfun(@(S)((S.a).^permute(repmat(S.cA,1,1,2,2),[3,4,1,2])),Q_BSyMP,'UniformOutput',false); % A to the power of C
	qA_BSyMP = NormalizeArray(ApowC);
	qA_BMR = NormalizeArray(Q_BMR,'a');

	ThinningRate = 100; % the number of elements to plot is propotial to 1/ThinningRate
	nexttile; plot_qMat(qA_Full(1,:,:,:),P(1).A(1,:,:,:),ThinningRate,['Timecourse of qA ',Q_Full(1).Label]);
	nexttile; plot_qMat(qA_BSyMP(1,:,:,:),P(1).A(1,:,:,:),ThinningRate,['Timecourse of qA ',Q_BSyMP(1).Label]);
	nexttile; plot_qMat(qA_BMR(1,:,:,:),P(1).A(1,:,:,:),ThinningRate,['Timecourse of qA ',Q_BMR(1).Label]);

	nexttile; PlotSurvRate(cat(3,Q_BMR.MatRetainedA),P(1).A); title('Survival-Rate during BMR');

	nexttile;
	semilogy(squeeze(sum((qA_Full-cat(5,P.A)).^2,[1,2,3,4]))); hold on;
	semilogy(squeeze(sum((qA_BSyMP-cat(5,P.A)).^2,[1,2,3,4]))); hold on;
	semilogy(squeeze(sum((qA_BMR-cat(5,P.A)).^2,[1,2,3,4]))); hold off;
	xlabel('session'); ylabel('(qA-A)^2'); title('(qA-A)^2 const'); legend(Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label);
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qA.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qA.fig')); catch; warning('saving qA.fig failed.'); end; end

	%%------Timecourse of qcA and Distribution of qcA------%%
	figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,'Property of qcA'); nexttile;
	ThinningRate = 100; % the number of elements to plot is propotial to 1/ThinningRate
	[qcA_NotFlat,qcA_Flat] = plot_qc(cat(3,Q_BSyMP.cA),P(1).A,ThinningRate,'qcA');

	nexttile; edges = 0:0.05:1; VisSession = nSessions; % session for visualization
	histogram(qcA_NotFlat(:,VisSession),edges,'FaceColor','r'); hold on;
	histogram(qcA_Flat(:,VisSession),edges,'FaceColor','b'); hold off;
	title(['Distribution of qcA(',num2str(VisSession),')']); legend('To be retained','To be reduced');
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qcA.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qcA.fig')); catch; warning('saving qcA.fig failed.'); end; end

	%%------Timecourse of qB and qcB------%%
	figure('Position',fullscreen); tile = tiledlayout(2,3); title(tile,'Timecourse of qB');

	qB_Full = NormalizeArray(Q_Full,'b');
	BpowC = arrayfun(@(S)((S.b).^permute(repmat(S.cB,1,1,2,2),[3,4,1,2])),Q_BSyMP,'UniformOutput',false); % B to the power of C
	qB_BSyMP = NormalizeArray(BpowC);
	qB_BMR = NormalizeArray(Q_BMR,'b');

	ThinningRate = 1; % the number of elements to plot is propotial to 1/ThinningRate
	nexttile; plot_qMat(qB_Full(1,:,:,:),P(1).B(1,:,:,:),ThinningRate,['Timecourse of qB ',Q_Full(1).Label]);
	nexttile; plot_qMat(qB_BSyMP(1,:,:,:),P(1).B(1,:,:,:),ThinningRate,['Timecourse of qB ',Q_BSyMP(1).Label]);
	nexttile; plot_qMat(qB_BMR(1,:,:,:),P(1).B(1,:,:,:),ThinningRate,['Timecourse of qB ',Q_BMR(1).Label]);

	nexttile;
	ThinningRate = 1; % the number of elements to plot is propotial to 1/ThinningRate
	plot_qc(cat(3,Q_BSyMP.cB),P(1).B,ThinningRate,'qcB');
	
	nexttile; PlotSurvRate(cat(3,Q_BMR.MatRetainedB),P(1).B); title('Survival-Rate during BMR');

	nexttile;
	semilogy(squeeze(sum((qB_Full-cat(5,P.B)).^2,[1,2,3,4])),'r'); hold on;
	semilogy(squeeze(sum((qB_BSyMP-cat(5,P.B)).^2,[1,2,3,4])),'b'); hold on;
	semilogy(squeeze(sum((qB_BMR-cat(5,P.B)).^2,[1,2,3,4])),'m'); hold off;
	xlabel('session'); ylabel('(qB-B)^2'); title('(qB-B)^2'); legend(Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label);
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qB_qcB.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qB_qcB.fig')); catch; warning('saving qB_qcB.fig failed.'); end; end
	
	%%------Error of each block in B------%%
	figure('Position',fullscreen); tile = tiledlayout('flow');
	title(tile,'Error of each block in B'); xlabel(tile,'cause'); ylabel(tile,'effect');

	ClrLim = [min([sum((qB_Full(:,:,:,:,VisSession)-P(1).B).^2,[1,2]),sum((qB_BSyMP(:,:,:,:,VisSession)-P(1).B).^2,[1,2]),sum((qB_BMR(:,:,:,:,VisSession)-P(1).B).^2,[1,2])],[],'all'),...
		max([sum((qB_Full(:,:,:,:,VisSession)-P(1).B).^2,[1,2]),sum((qB_BSyMP(:,:,:,:,VisSession)-P(1).B).^2,[1,2]),sum((qB_BMR(:,:,:,:,VisSession)-P(1).B).^2,[1,2])],[],'all')]; % Color Limits
	
	nexttile; heatmap(squeeze(sum((qB_Full(:,:,:,:,VisSession)-P(1).B).^2,[1,2])),'ColorLimits',ClrLim); title(Q_Full(1).Label);
	nexttile; heatmap(squeeze(sum((qB_BSyMP(:,:,:,:,VisSession)-P(1).B).^2,[1,2])),'ColorLimits',ClrLim); title(Q_BSyMP(1).Label);
	nexttile; heatmap(squeeze(sum((qB_BMR(:,:,:,:,VisSession)-P(1).B).^2,[1,2])),'ColorLimits',ClrLim); title(Q_BMR(1).Label);
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qB_error.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qB_error.fig')); catch; warning('saving qB_error.fig failed.'); end; end

	%%------Value of  qB(1,1,:,:)------%%
	figure('Position',fullscreen); tile = tiledlayout('flow');
	title(tile,'Value of  qB_{11}'); xlabel(tile,'cause'); ylabel(tile,'effect');

	ClrLim = [0.2,0.8]; % Color Limits
	
	nexttile; heatmap(squeeze(qB_Full(1,1,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_Full(1).Label);
	nexttile; heatmap(squeeze(qB_BSyMP(1,1,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_BSyMP(1).Label);
	nexttile; heatmap(squeeze(qB_BMR(1,1,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_BMR(1).Label);
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qB_11.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qB_11.fig')); catch; warning('saving qB_11.fig failed.'); end; end

	%%------Value of  qB(1,2,:,:)------%%
	figure('Position',fullscreen); tile = tiledlayout('flow');
	title(tile,'Value of  qB_{12}'); xlabel(tile,'cause'); ylabel(tile,'effect');

	ClrLim = [0.2,0.8]; % Color Limits
	
	nexttile; heatmap(squeeze(qB_Full(1,2,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_Full(1).Label);
	nexttile; heatmap(squeeze(qB_BSyMP(1,2,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_BSyMP(1).Label);
	nexttile; heatmap(squeeze(qB_BMR(1,2,:,:,VisSession)),'ColorLimits',ClrLim); title(Q_BMR(1).Label);
	if FlagSavePng; saveas(gcf,fullfile(SaviDir,'qB_12.png')); end
	if FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qB_12.fig')); catch; warning('saving qB_11.fig failed.'); end; end

	%%------Value of  true B(1,:,:,:)------%%
	figure('Position',fullscreen); tile = tiledlayout('flow');
	title(tile,'Value of  B_{1\bullet}'); xlabel(tile,'cause'); ylabel(tile,'effect');

	ClrLim = [0.2,0.8]; % Color Limits

	nexttile; heatmap(squeeze(P(1).B(1,1,:,:)),'ColorLimits',ClrLim); title('Value of  B_{11}');
	nexttile; heatmap(squeeze(P(1).B(1,2,:,:)),'ColorLimits',ClrLim); title('Value of  B_{12}');
end


%% nested functions
function qa_init = setPriorA4RL(epsFlat,epsNotFlat,amp,FlagInformative)
% eps:				bias for prior of parameters, must be in [0,1]
% amp:				amplitude for prior of parameters
% FlagInformative:	make prior informative or not
qa_init = NaN(size(P(1).A));
if FlagInformative
	LogiMatFlat = P(1).A==0.5;
	qa_init(1,:,:,:) = LogiMatFlat(1,:,:,:).*(0.5 + epsFlat*(2*rand(size(LogiMatFlat(1,:,:,:)))-1))...
					+ ~LogiMatFlat(1,:,:,:).*[0.5+epsNotFlat,0.5-epsNotFlat];
else
	qa_init(1,:,:,:) = 0.5 + epsFlat*(2*rand(size(qa_init(1,:,:,:)))-1);
end
qa_init(2,:,:,:) = 1-qa_init(1,:,:,:);
qa_init = qa_init*amp;
end

function qb_init = setPriorB(eps,amp)
qb_init = NaN(2,2,P(1).Ns,P(1).Ns);
qb_init(1,:,:,:) = 0.5 + eps*(2*rand(1,2,P(1).Ns,P(1).Ns)-1);
qb_init(2,:,:,:) = 1-qb_init(1,:,:,:);
qb_init = qb_init*amp;
end

function qd_init = setPriorD(eps,amp)
qd_init = NaN(2,P(1).Ns);
qd_init(1,:) = 0.5 + eps*(2*rand(1,P(1).Ns)-1);
qd_init(2,:) = 1-qd_init(1,:,:,:);
qd_init = qd_init*amp;
end

function [qcA_init,qcB_init] = setPriorC(qc_init,qc_eps)
qcA_init = qc_init*ones(P(1).No,P(1).Ns) + qc_eps*(2*rand(P(1).No,P(1).Ns)-1);
qcB_init = qc_init*ones(P(1).Ns,P(1).Ns) + qc_eps*(2*rand(P(1).Ns,P(1).Ns)-1);
end

function [F,AccA,AccB,qsCmp,qaCmp,qbCmp,qcA_Cmp,qcB_Cmp] = CalcFreeEnergy(Q,P)
No = size(Q(1).a,3); Ns = size(Q(1).a,4);
tmp = NaN(nSessions,1);
F = tmp;		% Free enerygy
AccA = tmp;		% Accuracy regarding A-matrix
AccB = tmp;		% Accuracy regarding B-matrix
qsCmp = tmp;	% Complexity of qs
qaCmp = tmp;	% Complexity of qa
qbCmp = tmp;	% Complexity of qb
qcA_Cmp = tmp;	% Complexity of qcA
qcB_Cmp = tmp;	% Complexity of qcB
qa_init_cur = qa_init; % duplicate the value to keep qa_init unchanged
qb_init_cur = qb_init; % duplicate the value to keep qa_init unchanged
for h=1:nSessions
	% Calculate accuracy
	%----------------------------------------------------------------------
	if strcmp(Q(1).Label,'BSyMP') % if the agent adopted BSyMP
		qcAqlnA_T = permute( Q(h).lnA.*permute(repmat(Q(h).cA,1,1,2,2),[3,4,1,2]) ,[2,1,3,4]);
		qcAqlnA_T_times_o = reshape( reshape(permute(qcAqlnA_T,[1,4,2,3]),2*Ns,2*No) * reshape(P(h).o,[],BatchSize), [],1);
		AccA(h) = -reshape(Q(h).s,1,[])*qcAqlnA_T_times_o + log(2)*(No-sum(Q(h).cA,'all'))*BatchSize;

		qcBqlnB_times_s = reshape( reshape(permute(Q(h).lnB.*permute(repmat(Q(h).cB,1,1,2,2),[3,4,1,2]),[1,3,2,4]),2*Ns,2*Ns) * reshape(Q(h).s(:,:,1:end-1),[],BatchSize-1), [],1);
		AccB(h) = -reshape(Q(h).s(:,:,2:end),1,[])*qcBqlnB_times_s + log(2)*(Ns-sum(Q(h).cB,'all'))*(BatchSize-1);
	else
		qlnA_T = permute(Q(h).lnA,[2,1,3,4]);
		qlnA_T_times_o = reshape( reshape(permute(qlnA_T,[1,4,2,3]),2*Ns,2*No) * reshape(P(h).o,[],BatchSize), [],1);
		AccA(h) = -reshape(Q(h).s,1,[])*qlnA_T_times_o + log(2)*(No-No*Ns)*BatchSize;

		qlnB_times_s = reshape( reshape(permute(Q(h).lnB,[1,3,2,4]),2*Ns,2*Ns) * reshape(Q(h).s(:,:,1:end-1),[],BatchSize-1), [],1);
		AccB(h) = -reshape(Q(h).s(:,:,2:end),1,[])*qlnB_times_s + log(2)*(Ns-Ns*Ns)*(BatchSize-1);
	end

	% Calculate complexity
	%----------------------------------------------------------------------
	qsCmp(h) = reshape(Q(h).s,1,[]) * log(reshape(Q(h).s,[],1));
	
	if ~strcmp(Q(1).Label,'BSyMP')
		if strcmp(Q(1).Label,'BMR')
			qa_init_cur = qa_init; % duplicate the value to keep qa_init unchanged
			idxRdc = permute(repmat(~Q(h).MatRetainedA,[1,1,2,2]),[3,4,1,2]);
			qa_init_cur(idxRdc) =  Q(1).a_init_rdc;

			qb_init_cur = qb_init; % duplicate the value to keep qb_init unchanged
			idxRdc = permute(repmat(~Q(h).MatRetainedB,[1,1,2,2]),[3,4,1,2]);
			qb_init_cur(idxRdc) =  Q(1).b_init_rdc;
		end
	else
		qcA_Cmp(h) = sum(Q(h).cA.*(log(Q(h).cA)-log(qcA_init)),'all','omitnan') + sum((1-Q(h).cA).*(log(1-Q(h).cA)-log(1-qcA_init)),'all','omitnan');
		qcB_Cmp(h) = sum(Q(h).cB.*(log(Q(h).cB)-log(qcB_init)),'all','omitnan') + sum((1-Q(h).cB).*(log(1-Q(h).cB)-log(1-qcB_init)),'all','omitnan');
	end
	qaCmp(h) = sum((Q(h).a-qa_init_cur).*Q(h).lnA,'all') - sum(betaln(Q(h).a(1,:,:,:),Q(h).a(2,:,:,:)),'all') + sum(betaln(qa_init_cur(1,:,:,:),qa_init_cur(2,:,:,:)),'all');
	qbCmp(h) = sum((Q(h).b-qb_init_cur).*Q(h).lnB,'all') - sum(betaln(Q(h).b(1,:,:,:),Q(h).b(2,:,:,:)),'all') + sum(betaln(qb_init_cur(1,:,:,:),qb_init_cur(2,:,:,:)),'all');
	
	F(h) = sum([AccA;AccB;qsCmp(h);qaCmp(h);qbCmp(h)],'omitnan');
end
end

function [qsTest,qsPredTest] = EvalPerformance_RL(Q,P_test,ShowProgress)
if ShowProgress
	f = waitbar(0,['Evaluating performance of ',Q.Label,'...']);
	f.NumberTitle = 'off';
	f.Name = 'Progress Status';
end
if strcmp(Q(1).Label,'BSyMP') % if the agent adopted BSyMP
	t=1;
	termA = squeeze(pagemtimes(reshape(permute(Q.lnA.*permute(repmat(Q.cA,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],P_test.Ns) , repmat(reshape(P_test.o(:,:,t),[],1),1,1,P_test.Ns)));
	qsTest(:,:,t) = softmax(termA+Q.lnD);
	qsPredTest(:,:,t) = softmax(Q.lnD);
	for t=2:P_test.T
		if ShowProgress; waitbar(t/P_test.T,f); end
		[qsTest(:,:,t),qsPredTest(:,:,t)] = Optimize_qs_RL(qsTest(:,:,t-1),P_test.o(:,:,t),Q.lnA,Q.lnB,Q.cA,Q.cB);
	end
else
	t=1;
	termA = squeeze(pagemtimes(reshape(permute(Q.lnA,[2,1,3,4]),2,[],P_test.Ns) , repmat(reshape(P_test.o(:,:,t),[],1),1,1,P_test.Ns)));
	qsTest(:,:,t) = softmax(termA+Q.lnD);
	qsPredTest(:,:,t) = softmax(Q.lnD);
	for t=2:P_test.T
		if ShowProgress; waitbar(t/P_test.T,f); end
		[qsTest(:,:,t),qsPredTest(:,:,t)] = Optimize_qs_RL(qsTest(:,:,t-1),P_test.o(:,:,t),Q.lnA,Q.lnB);
	end
end
if ShowProgress; close(f); end
end

function plot_qs(Q,j,Title,FlagPred)
if nargin<4; FlagPred = false; end
s = cat(3,P.s);
if FlagPred; qs = cat(3,Q.sPred); else; qs = cat(3,Q.s); end
x1 = find(s(1,j,:)==1); y1 = squeeze(qs(1,j,x1));
x0 = find(s(1,j,:)==0); y0 = squeeze(qs(1,j,x0));
plot(x1(1:10:end),y1(1:10:end),'r.',x0(1:10:end),y0(1:10:end),'b.'); hold on;
curve1 = NaN(T,1); curve1(x1) = y1; curve0 = NaN(T,1); curve0(x0) = y0;
plot(movmean(curve1,T/50,'omitnan'),'m','LineWidth',3); hold on; plot(movmean(curve0,T/50,'omitnan'),'c','LineWidth',3);
title(Title); xlabel('time'); legend('s=1','s=0');
if FlagPred; ylabel('qsPred'); else; ylabel('qs'); end
end

function vsPlotProbPred(Q,j,SoI,Title)
sProb = cat(3,P(SoI).sProb); sProb = squeeze(sProb(1,j,:));
sPred = cat(3,Q(SoI).sPred); sPred = squeeze(sPred(1,j,:));
scatter(sProb,sPred); hold on;
plot(linspace(0,1,1000),linspace(0,1,1000));
title(Title); xlabel('sProb'); ylabel('sPred');
end

function MatNorm = NormalizeArray(ArrayIn,field)
if isstruct(ArrayIn)||nargin>1
	Mat = cat(5,ArrayIn.(field));
	MatNorm = Mat./sum(Mat,1);
else
	CellNorm = cellfun(@(mat)(mat./sum(mat,1)),ArrayIn,'UniformOutput',false);
	MatNorm = cell2mat(reshape(CellNorm,1,1,1,1,length(CellNorm)));
end
end

function plot_qMat(qMat_upper,pMat_upper,ThinningRate,char_title,colorArray)
if nargin<5
	colorArray = {'#268cdd','#f57729','#ffe864','#c05cfb','#49db40','#6cf4ff',...
		'#f267c5','#ff7a74','#7da9ff','#fec04c','#1fcfbe','#dc996c'}; % glow12
end
[~,~,ValArray]=find(unique(pMat_upper));
plotHandle = gobjects(1,length(ValArray)); legendArray = cell(1,length(ValArray));
for i=1:length(ValArray)
	val = ValArray(i);
	qA_cur = reshape(qMat_upper(repmat(pMat_upper==val,[1,1,1,nSessions])),[sum(pMat_upper==val,'all'),nSessions]);
	p = plot(1:nSessions,qA_cur(1:ThinningRate:end,:),'Color',colorArray{i}); hold on;
	plotHandle(i) = p(1); legendArray{i} = num2str(val);
end
title(char_title); xlabel('session');
legend(plotHandle,legendArray);
end

function PlotSurvRate(MatRetained,TrueMat)
SurvRate_NotFlat = mean(reshape(MatRetained(repmat(squeeze(any(TrueMat~=0.5,[1,2])),[1,1,nSessions])),[sum(any(TrueMat~=0.5,[1,2]),'all'),nSessions]));
SurvRate_Flat = mean(reshape(MatRetained(repmat(squeeze(all(TrueMat==0.5,[1,2])),[1,1,nSessions])),[sum(all(TrueMat==0.5,[1,2]),'all'),nSessions]));
plot(1:nSessions,SurvRate_NotFlat,'r',1:nSessions,SurvRate_Flat,'b');
xlabel('session'); ylabel('Survival-Rate'); legend('To be reduced','To be retained');
end

function [qcNotFlat,qcFlat] = plot_qc(Mat_qc,TrueMat,ThinningRate,Label)
qcNotFlat = reshape(Mat_qc(repmat(squeeze(any(TrueMat~=0.5,[1,2])),[1,1,nSessions])),[sum(any(TrueMat~=0.5,[1,2]),'all'),nSessions]);
qcFlat = reshape(Mat_qc(repmat(squeeze(all(TrueMat==0.5,[1,2])),[1,1,nSessions])),[sum(all(TrueMat==0.5,[1,2]),'all'),nSessions]);
p1 = plot(1:nSessions,qcNotFlat(1:ThinningRate:end,:),'r'); hold on;
p2 = plot(1:nSessions,qcFlat(1:ThinningRate:end,:),'b'); hold off;
xlabel('session'); ylabel('Survival-Rate'); legend([p1(1),p2(1)],{'To be retained','To be reduced'});
title(['Timecourse of ',Label]);
end
end