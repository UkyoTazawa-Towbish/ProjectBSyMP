function [P,P_test,Q_Full,Q_BSyMP,Q_BMR,Q_OnlineBMR] = Main_BSS(SaveDirParent,ShapeID)
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
% ShapeID:			Natural number (1,2,3) that designates shape for task
% 
% Output
% P:			Info of outer world (truth), used for training sessions
% P_test:		Info of outer world (truth), used for test sessions       
% Q_Full:		Beliefs of full model agent
% Q_BSyMP:		Beliefs of BSyMP agent
% Q_BMR:		Beliefs of BMR agent
% Q_OnlineBMR:	Beliefs of online-BMR agent
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
if nargin<2; ShapeID = 1; end

BatchSize = 10;				% time point length of a training session
nSessions = 1000;			% number of training sessions
T = BatchSize*nSessions;	% total time points
OnProb = 0.6;				% P(o=1|s=1) for pixels on shape
OffProb = 0.4;				% P(o=1|s=0) for pixels on shape
FlagFigOfTask = false;		% make figures of the task or not
lenTest = 1000;				% time point length of a test session


%% Generate data, and Run optimization: 1st session
P = GenerateTask_BSS(BatchSize,OnProb,OffProb,ShapeID,FlagFigOfTask,ShowProgressBatch);

% Prepare priors
%--------------------------------------------------------------------------
eps     = 0.01;			% bias for prior of parameters, must be in [0,1]
amp     = 50;			% amplitude for prior of parameters
FlagInformative = true;	% make prior imformative or not
qa_init = setPriorA(eps,amp,FlagInformative);

qc_init = 0.9;			% base value for prior of C
qc_eps = 0.05;			% amplitude of fluctuation for prior of C
qcA_init = setPriorC(qc_init,qc_eps);

% Run optimization and pack the results in Q
%--------------------------------------------------------------------------
Q_Full = RunOptimization_BSS(P,qa_init,false,ShowProgressBatch);
Q_BSyMP = RunOptimization_BSS(P,qa_init,qcA_init,ShowProgressBatch);

% Preallocate nSessions*1 structure arrays for the following steps
%--------------------------------------------------------------------------
P = PreallocateStruct(P,nSessions);
Q_Full = PreallocateStruct(Q_Full,nSessions);
Q_BSyMP = PreallocateStruct(Q_BSyMP,nSessions);


%% Generate data, and Run optimization: subsequent sessions
if ShowProgress
	wb = waitbar(0,'Optimizing Q Full and Q BSyMP...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end
for n=2:nSessions
	if ShowProgress; waitbar(n/nSessions,wb); end
	P(n) = GenerateTask_BSS(BatchSize,OnProb,OffProb,ShapeID,false,ShowProgressBatch);

	Q_Full(n) = RunOptimization_BSS(P(n),Q_Full(n-1).a,false,ShowProgressBatch);
	Q_BSyMP(n) = RunOptimization_BSS(P(n),Q_BSyMP(n-1).a,Q_BSyMP(n-1).cA,ShowProgressBatch);
end
if ShowProgress; close(wb); end


%% Apply BMR
MagnA = 1000000;			% magnification for posterior matrix a
threshA_post = 0;			% reduction threshold for posterior matrix a
threshA_online = -3;		% reduction threshold for posterior matrix a
Q_BMR = BMR_BSS(P,Q_Full,MagnA,threshA_post,false,ShowProgress);
Q_OnlineBMR = BMR_BSS(P,Q_Full,MagnA,threshA_online,true,ShowProgress);


%% Calculate free-energy
Q_Full(end).F = CalcFreeEnergy(Q_Full,P);
Q_BSyMP(end).F = CalcFreeEnergy(Q_BSyMP,P);
Q_BMR(end).F = CalcFreeEnergy(Q_BMR,P);
Q_OnlineBMR(end).F = CalcFreeEnergy(Q_OnlineBMR,P);


%% Test session
P_test = GenerateTask_BSS(lenTest,OnProb,OffProb,ShapeID,false,ShowProgress);
Q_Full(end).sTest = EvalPerformance_BSS(Q_Full(end),P_test,ShowProgress);
Q_BSyMP(end).sTest = EvalPerformance_BSS(Q_BSyMP(end),P_test,ShowProgress);
Q_BMR(end).sTest = EvalPerformance_BSS(Q_BMR(end),P_test,ShowProgress);
Q_OnlineBMR(end).sTest = EvalPerformance_BSS(Q_OnlineBMR(end),P_test,ShowProgress);


%% Save the results as .mat file
if FlagSave || FlagSavePng || FlagSaveFig
	SaviDir = NumberSaveDir(fullfile(SaveDirParent,['BSS_BatchSize',num2str(BatchSize),'_Session',num2str(nSessions),'_On',num2str(OnProb),'_Off',num2str(OffProb),'_FlagInfo',num2str(FlagInformative),'_qcInit',num2str(qc_init),'_Simulation']));
end
if FlagSave
	save(fullfile(SaviDir,'P.mat'),'P','-v7.3'); save(fullfile(SaviDir,'P_test.mat'),'P_test','-v7.3');
	save(fullfile(SaviDir,'Q_Full.mat'),'Q_Full','-v7.3'); save(fullfile(SaviDir,'Q_BSyMP.mat'),'Q_BSyMP','-v7.3');
	save(fullfile(SaviDir,'Q_BMR.mat'),'Q_BMR','-v7.3'); save(fullfile(SaviDir,'Q_OnlineBMR.mat'),'Q_OnlineBMR','-v7.3');
end


%% Visualize the results
if FlagVisualize
	set(0,'units','pixels'); screen = get(0,'ScreenSize'); fullscreen = [0,0,screen(3:4)];

	%%------Timecourse of qs------%%
	figure('Position',fullscreen); tile = tiledlayout(P(1).Ns,4); title(tile,'Timecourse of qs');
	nexttile; plot_qs(Q_Full,1,['state 1 ',Q_Full(1).Label]);
	nexttile; plot_qs(Q_BSyMP,1,['state 1 ',Q_BSyMP(1).Label]);
	nexttile; plot_qs(Q_BMR,1,['state 1 ',Q_BMR(1).Label]);
	nexttile; plot_qs(Q_OnlineBMR,1,['state 1 ',Q_OnlineBMR(1).Label]);
	nexttile; plot_qs(Q_Full,2,['state 2 ',Q_Full(1).Label]);
	nexttile; plot_qs(Q_BSyMP,2,['state 2 ',Q_BSyMP(1).Label]);
	nexttile; plot_qs(Q_BMR,2,['state 2 ',Q_BMR(1).Label]);
	nexttile; plot_qs(Q_OnlineBMR,2,['state 2 ',Q_OnlineBMR(1).Label]);
	if FlagSave&&FlagSavePng; saveas(gcf,fullfile(SaviDir,'qs.png')); end
	if FlagSave&&FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qs.fig')); catch; warning('saving qs.fig failed.'); end; end

	%%------Timecourse of qA------%%
	figure('Position',fullscreen); tile = tiledlayout(2,4); title(tile,'Timecourse of qA');
	
	qA_Full = NormalizeArray(Q_Full,'a');
	ApowC = arrayfun(@(S)((S.a).^permute(repmat(S.cA,1,1,2,2),[3,4,1,2])),Q_BSyMP,'UniformOutput',false); % A to the power of C
	qA_BSyMP = NormalizeArray(ApowC);
	qA_BMR = NormalizeArray(Q_BMR,'a');
	qA_SeqBMR = NormalizeArray(Q_OnlineBMR,'a');

	nexttile; plot_qA(qA_Full,['Timecourse of qA ',Q_Full(1).Label]);
	nexttile; plot_qA(qA_BSyMP,['Timecourse of qA ',Q_BSyMP(1).Label]);
	nexttile; plot_qA(qA_BMR,['Timecourse of qA ',Q_BMR(1).Label]);
	nexttile; plot_qA(qA_SeqBMR,['Timecourse of qA ',Q_OnlineBMR(1).Label]);

	nexttile; PlotSurvA(P,Q_BMR); title('Survival-Rate during BMR');
	nexttile; PlotSurvA(P,Q_OnlineBMR); title('Survival-Rate during Online-BMR');

	nexttile;
	semilogy(squeeze(sum((qA_Full-cat(5,P.A)).^2,[1,2,3,4]))); hold on;
	semilogy(squeeze(sum((qA_BSyMP-cat(5,P.A)).^2,[1,2,3,4]))); hold on;
	semilogy(squeeze(sum((qA_BMR-cat(5,P.A)).^2,[1,2,3,4]))); hold on;
	semilogy(squeeze(sum((qA_SeqBMR-cat(5,P.A)).^2,[1,2,3,4]))); hold off;
	xlabel('session'); ylabel('(qA-A)^2'); title('(qA-A)^2'); legend(Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,Q_OnlineBMR(1).Label);
	if FlagSave&&FlagSavePng; saveas(gcf,fullfile(SaviDir,'qA.png')); end
	if FlagSave&&FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qA.fig')); catch; warning('saving qA.fig failed.'); end; end

	%%------Timecourse of qcA and Distribution of qcA------%%
	figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,'Property of qcA'); nexttile;
	Mat_cA = cat(3,Q_BSyMP.cA);
	p1 = plot(reshape([Mat_cA(~P(1).LogiVecCircle,1,:);Mat_cA(~P(1).LogiVecRect,2,:)],...
		[sum(~P(1).LogiVecCircle)+sum(~P(1).LogiVecRect), nSessions])','g:'); hold on;
	p2 = plot(reshape(Mat_cA(P(1).LogiVecCircle,1,:),[sum(P(1).LogiVecCircle), nSessions])','r','LineWidth',3); hold on;
	p3 = plot(reshape(Mat_cA(P(1).LogiVecRect,2,:),[sum(P(1).LogiVecRect), nSessions])','b','LineWidth',3); hold off;
	title('Timecourse of qcA'); legend([p1(1),p2(1),p3(1)],{'Random','Circ','Rect'});

	nexttile; edges = 0:0.05:1; VisSession = nSessions; % session for visualization
	histogram(Q_BSyMP(VisSession).cA(P(1).LogiVecCircle,1),edges); hold on;
	histogram(Q_BSyMP(VisSession).cA(P(1).LogiVecRect,2),edges); hold on;
	histogram([Q_BSyMP(VisSession).cA(~P(1).LogiVecCircle,1);Q_BSyMP(VisSession).cA(~P(1).LogiVecRect,2)],edges); hold off;
	title(['Distribution of qcA(',num2str(VisSession),')']); legend('Circ','Rect','Random');
	if FlagSave&&FlagSavePng; saveas(gcf,fullfile(SaviDir,'qcA.png')); end
	if FlagSave&&FlagSaveFig; try saveas(gcf,fullfile(SaviDir,'qcA.fig')); catch; warning('saving qcA.fig failed.'); end; end
end


%% nested functions
function qa_init = setPriorA(eps,amp,FlagInformative)
% eps:				bias for prior of parameters, must be in [0,1]
% amp:				amplitude for prior of parameters
% FlagInformative:	make prior informative or not
qa_init = NaN(size(P(1).A));
if FlagInformative
	LogiMatFlat = P(1).A==0.5;
	qa_init(1,:,:,:) = LogiMatFlat(1,:,:,:).*(0.5 + eps*(2*rand(size(LogiMatFlat(1,:,:,:)))-1))...
					+ ~LogiMatFlat(1,:,:,:).*[0.5+eps,0.5-eps];
else
	qa_init(1,:,:,:) = 0.5 + eps*(2*rand(size(qa_init(1,:,:,:)))-1);
end
qa_init(2,:,:,:) = 1-qa_init(1,:,:,:);
qa_init = qa_init*amp;
end

function qcA_init = setPriorC(qc_init,qc_eps)
qcA_init = qc_init*ones(P(1).No,P(1).Ns) + qc_eps*(2*rand(P(1).No,P(1).Ns)-1);
end

function [F,Acc,qsCmp,qaCmp,qcCmp] = CalcFreeEnergy(Q,P)
No = size(Q(1).a,3); Ns = size(Q(1).a,4);
tmp = NaN(nSessions,1);
F = tmp;		% Free enerygy
Acc = tmp;		% Accuracy
qsCmp = tmp;	% Complexity of qs
qaCmp = tmp;	% Complexity of qa
qcCmp = tmp;	% Complexity of qc
qa_init_cur = qa_init; % duplicate the value to keep qa_init unchanged
for h=1:nSessions
	% Calculate accuracy
	%----------------------------------------------------------------------
	if strcmp(Q(1).Label,'BSyMP') % if the agent adopted BSyMP
		qcAqlnA_T = permute( Q(h).lnA.*permute(repmat(Q(h).cA,1,1,2,2),[3,4,1,2]) ,[2,1,3,4]);
		qcAqlnA_T_times_o = reshape( reshape(permute(qcAqlnA_T,[1,4,2,3]),2*Ns,2*No) * reshape(P(h).o,[],BatchSize), [],1);
		Acc(h) = -reshape(Q(h).s,1,[])*qcAqlnA_T_times_o + log(2)*(No-sum(Q(h).cA,'all'))*BatchSize;
	else
		qlnA_T = permute(Q(h).lnA,[2,1,3,4]);
		qlnA_T_times_o = reshape( reshape(permute(qlnA_T,[1,4,2,3]),2*Ns,2*No) * reshape(P(h).o,[],BatchSize), [],1);
		Acc(h) = -reshape(Q(h).s,1,[])*qlnA_T_times_o + log(2)*(No-No*Ns)*BatchSize;
	end

	% Calculate complexity
	%----------------------------------------------------------------------
	qsCmp(h) = reshape(Q(h).s,1,[]) * log(reshape(Q(h).s,[],1));
	
	if ~strcmp(Q(1).Label,'BSyMP')
		if strcmp(Q(1).Label,'BMR')||strcmp(Q(1).Label,'Online BMR')
			qa_init_cur = qa_init; % duplicate the value to keep qa_init unchanged
			idxRdc = permute(repmat(~Q(h).MatRetainedA,[1,1,2,2]),[3,4,1,2]);
			qa_init_cur(idxRdc) =  Q(1).a_init_rdc;
		end
	else
		qcCmp(h) = sum(Q(h).cA.*(log(Q(h).cA)-log(qcA_init)),'all','omitnan') + sum((1-Q(h).cA).*(log(1-Q(h).cA)-log(1-qcA_init)),'all','omitnan');
	end
	qaCmp(h) = sum((Q(h).a-qa_init_cur).*Q(h).lnA,'all') - sum(betaln(Q(h).a(1,:,:,:),Q(h).a(2,:,:,:)),'all') + sum(betaln(qa_init_cur(1,:,:,:),qa_init_cur(2,:,:,:)),'all');
	
	F(h) = sum([Acc;qsCmp(h);qaCmp(h)],'omitnan');
end
end

function qsTest = EvalPerformance_BSS(Q,P,ShowProgress)
if ShowProgress
	wb = waitbar(0,['Evaluating performance of ',Q.Label,'...']);
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end
if strcmp(Q(1).Label,'BSyMP') % if the agent adopted BSyMP
	t=1;
	termA = squeeze(pagemtimes(reshape(permute(Q.lnA.*permute(repmat(Q.cA,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
	qsTest(:,:,t) = softmax(termA);
	for t=2:P.T
		if ShowProgress; waitbar(t/P.T,wb); end
		qsTest(:,:,t) = Optimize_qs_BSS(P.o(:,:,t),Q.lnA,Q.cA);
	end
else
	t=1;
	termA = squeeze(pagemtimes(reshape(permute(Q.lnA,[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
	qsTest(:,:,t) = softmax(termA);
	for t=2:P.T
		if ShowProgress; waitbar(t/P.T,wb); end
		qsTest(:,:,t) = Optimize_qs_BSS(P.o(:,:,t),Q.lnA);
	end
end
if ShowProgress; close(wb); end
end

function plot_qs(Q,j,Title)
s = cat(3,P.s); qs = cat(3,Q.s);
x1 = find(s(1,j,:)==1); y1 = squeeze(qs(1,j,x1));
x0 = find(s(1,j,:)==0); y0 = squeeze(qs(1,j,x0));
plot(x1(1:10:end),y1(1:10:end),'r.',x0(1:10:end),y0(1:10:end),'b.'); hold on;
curve1 = NaN(T,1); curve1(x1) = y1; curve0 = NaN(T,1); curve0(x0) = y0;
plot(movmean(curve1,T/50,'omitnan'),'m','LineWidth',3); hold on; plot(movmean(curve0,T/50,'omitnan'),'c','LineWidth',3);
title(Title); xlabel('time'); ylabel('qs'); legend('s=1','s=0');
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

function plot_qA(qA,char_title)
p5 = plot(1:nSessions,reshape(qA(1,1,~P(1).LogiVecCircle,1,:),[sum(~P(1).LogiVecCircle), nSessions])','g',...
	1:nSessions,reshape(qA(1,2,~P(1).LogiVecCircle,1,:),[sum(~P(1).LogiVecCircle), nSessions])','g',...
	1:nSessions,reshape(qA(1,1,~P(1).LogiVecRect,2,:),[sum(~P(1).LogiVecRect), nSessions])','g',...
	1:nSessions,reshape(qA(1,2,~P(1).LogiVecRect,2,:),[sum(~P(1).LogiVecRect), nSessions])','g'); hold on;
p1 = plot(1:nSessions,reshape(qA(1,1,P(1).LogiVecCircle,1,:),[sum(P(1).LogiVecCircle), nSessions])','r'); hold on;
p2 = plot(1:nSessions,reshape(qA(1,1,P(1).LogiVecRect,2,:),[sum(P(1).LogiVecRect), nSessions])','b'); hold on;
p3 = plot(1:nSessions,reshape(qA(1,2,P(1).LogiVecCircle,1,:),[sum(P(1).LogiVecCircle), nSessions])','m'); hold on;
p4 = plot(1:nSessions,reshape(qA(1,2,P(1).LogiVecRect,2,:),[sum(P(1).LogiVecRect), nSessions])','c'); hold off;
title(char_title); xlabel('session');
legend([p1(1),p2(1),p3(1),p4(1),p5(1)],{'Circ 1','Rect 1','Circ 0','Rect 0','Random'});
end

function PlotSurvA(P,Q)
MatRetainedA = cat(3,Q.MatRetainedA);
CircSurvA = squeeze(sum(MatRetainedA(:,1,:) & P(1).LogiVecCircle,1))/sum(P(1).LogiVecCircle);
RectSurvA = squeeze(sum(MatRetainedA(:,2,:) & P(1).LogiVecRect,1))/sum(P(1).LogiVecRect);
RandSurvA = squeeze(sum(MatRetainedA(:,1,:) & ~P(1).LogiVecCircle,1) + sum(MatRetainedA(:,2,:) & ~P(1).LogiVecRect,1)) / sum([~P(1).LogiVecCircle;~P(1).LogiVecRect]);
plot(1:nSessions,CircSurvA,'r',1:nSessions,RectSurvA,'b',1:nSessions,RandSurvA,'g');
xlabel('session'); ylabel('Survival-Rate'); legend('Circle','Rectangle','Random');
end
end