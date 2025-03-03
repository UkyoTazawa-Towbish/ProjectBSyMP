%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Repeat the main routine (Main_RL.m)
% and make figures that summarize the results of repeated simulations.
%__________________________________________________________________________

clear; close all;

nRuleIDs = 3; % number of RuleIDs, should be 3 for the most of cases
N = 100;  % number of repetitions

for RuleID=1:nRuleIDs
	LabelID = ['RuleID_',num2str(RuleID)];
	SaveDirParent = fullfile(pwd,['Results_RL_',LabelID]); mkdirIfNotExist(SaveDirParent);
	DiaryFileName = fullfile(SaveDirParent,['Diary_',char(datetime('now','TimeZone','local','Format','yyyyMMdd_HHmm')),'.txt']);
	diary(DiaryFileName);

	tStart = tic;
	disp(['    RuleID=',num2str(RuleID),': Start working!']);
	fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
	for n=1:N
		tic; disp(['n=',num2str(n),': Start working!']);

		% Run main routine
		%------------------------------------------------------------------
		[P,P_test,Q_Full,Q_BSyMP,Q_BMR] = Main_RL(SaveDirParent,RuleID);

		% Prepare variables to store the results only after the 1st run
		%------------------------------------------------------------------
		if n==1 && RuleID == 1
			Ns = P(1).Ns; No = P(1).No;
			tmp = NaN(N,nRuleIDs);
			F_Full = tmp; F_BSyMP = tmp; F_BMR = tmp;
			qsTestPredError_Full = tmp; qsTestPredError_BSyMP = tmp; qsTestPredError_BMR = tmp;
			qbError_Full = tmp; qbError_BSyMP = tmp; qbError_BMR = tmp;
			tmp = cell(N,nRuleIDs,2);
			qcB_Init_BSyMP = tmp; qcB_Init_BMR = tmp;
			qcB_Final_BSyMP = tmp; qcB_Final_BMR = tmp;
			clear tmp;
		end

		% Store free energy in arrays
		%------------------------------------------------------------------
		F_Full(n,RuleID) = Q_Full(end).F(end);
		F_BSyMP(n,RuleID) = Q_BSyMP(end).F(end);
		F_BMR(n,RuleID) = Q_BMR(end).F(end);

		% Store test errors in arrays
		%------------------------------------------------------------------
		Norm_s = sqrt(Ns);
		qsTestPredError_Full(n,RuleID) = mean(sqrt(sum((Q_Full(end).sPredTest-P_test.sProb).^2,[1,2]))/Norm_s);
		qsTestPredError_BSyMP(n,RuleID) = mean(sqrt(sum((Q_BSyMP(end).sPredTest-P_test.sProb).^2,[1,2]))/Norm_s);
		qsTestPredError_BMR(n,RuleID) = mean(sqrt(sum((Q_BMR(end).sPredTest-P_test.sProb).^2,[1,2]))/Norm_s);

		% Store learning errors in arrays
		%------------------------------------------------------------------
		Norm_B = sqrt(sum((P(1).B).^2,'all'));
		qbError_Full(n,RuleID) = sqrt(sum((NormalizeArray(Q_Full(end),'b') - P(end).B).^2,'all'))/Norm_B;
		BpowC = (Q_BSyMP(end).b).^permute(repmat(Q_BSyMP(end).cB,1,1,2,2),[3,4,1,2]); % B to the power of C
		qbError_BSyMP(n,RuleID) = sqrt(sum((NormalizeArray(BpowC) - P(end).B).^2,'all'))/Norm_B;
		qbError_BMR(n,RuleID) = sqrt(sum((NormalizeArray(Q_BMR(end),'b') - P(end).B).^2,'all'))/Norm_B;

		% Store qcB in arrays
		%------------------------------------------------------------------
		LogeMat = squeeze(any(P(1).B~=0.5,[1,2]));

		h = 1;
		qcB_Init_BSyMP{n,RuleID,1} = Q_BSyMP(h).cB(LogeMat); qcB_Init_BSyMP{n,RuleID,2} = Q_BSyMP(h).cB(~LogeMat);
		qcB_Init_BMR{n,RuleID,1} = Q_BMR(h).MatRetainedB(LogeMat); qcB_Init_BMR{n,RuleID,2} = Q_BMR(h).MatRetainedB(~LogeMat);
		qcB_Final_BSyMP{n,RuleID,1} = Q_BSyMP(end).cB(LogeMat); qcB_Final_BSyMP{n,RuleID,2} = Q_BSyMP(end).cB(~LogeMat);
		qcB_Final_BMR{n,RuleID,1} = Q_BMR(end).MatRetainedB(LogeMat); qcB_Final_BMR{n,RuleID,2} = Q_BMR(end).MatRetainedB(~LogeMat);

		toc; fprintf('----------------------------\n');
	end
	disp([num2str(N),' repetition has been done!']);
	fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');

	elapsedTime = toc(tStart);
	close all; diary off;
end


%% Save the results and make figures of all the data
% Prepare the linearized variables for CSV-files
%--------------------------------------------------------------------------
qcB_Init_BSyMP_Histogram_NotFlat = [cell2mat(qcB_Init_BSyMP(:,1,1));cell2mat(qcB_Init_BSyMP(:,2,1));cell2mat(qcB_Init_BSyMP(:,3,1))];
qcB_Init_BMR_Histogram_NotFlat = [cell2mat(qcB_Init_BMR(:,1,1));cell2mat(qcB_Init_BMR(:,2,1));cell2mat(qcB_Init_BMR(:,3,1))];
qcB_Init_BSyMP_Histogram_Flat = [cell2mat(qcB_Init_BSyMP(:,1,2));cell2mat(qcB_Init_BSyMP(:,2,2));cell2mat(qcB_Init_BSyMP(:,3,2))];
qcB_Init_BMR_Histogram_Flat = [cell2mat(qcB_Init_BMR(:,1,2));cell2mat(qcB_Init_BMR(:,2,2));cell2mat(qcB_Init_BMR(:,3,2))];
qcB_Final_BSyMP_Histogram_NotFlat = [cell2mat(qcB_Final_BSyMP(:,1,1));cell2mat(qcB_Final_BSyMP(:,2,1));cell2mat(qcB_Final_BSyMP(:,3,1))];
qcB_Final_BMR_Histogram_NotFlat = [cell2mat(qcB_Final_BMR(:,1,1));cell2mat(qcB_Final_BMR(:,2,1));cell2mat(qcB_Final_BMR(:,3,1))];
qcB_Final_BSyMP_Histogram_Flat = [cell2mat(qcB_Final_BSyMP(:,1,2));cell2mat(qcB_Final_BSyMP(:,2,2));cell2mat(qcB_Final_BSyMP(:,3,2))];
qcB_Final_BMR_Histogram_Flat = [cell2mat(qcB_Final_BMR(:,1,2));cell2mat(qcB_Final_BMR(:,2,2));cell2mat(qcB_Final_BMR(:,3,2))];

% Save the results as .mat files
%--------------------------------------------------------------------------
DirSummary = fullfile(pwd,'Summary_RL'); mkdirIfNotExist(DirSummary);
save(fullfile(DirSummary,'F_All.mat'),'F_Full','F_BSyMP','F_BMR','-v7.3');
save(fullfile(DirSummary,'qsTestPredError_All.mat'),'qsTestPredError_Full','qsTestPredError_BSyMP','qsTestPredError_BMR','-v7.3');
save(fullfile(DirSummary,'qbError_All.mat'),'qbError_Full','qbError_BSyMP','qbError_BMR','-v7.3');
save(fullfile(DirSummary,'qcB_Init_All.mat'),'qcB_Init_BSyMP','qcB_Init_BMR','-v7.3');
save(fullfile(DirSummary,'qcB_Final_All.mat'),'qcB_Final_BSyMP','qcB_Final_BMR','-v7.3');
save(fullfile(DirSummary,'qcB_Init_All_Histogram.mat'),'qcB_Init_BSyMP_Histogram_NotFlat','qcB_Init_BSyMP_Histogram_Flat','qcB_Init_BMR_Histogram_NotFlat','qcB_Init_BMR_Histogram_Flat','-v7.3');
save(fullfile(DirSummary,'qcB_Final_All_Histogram.mat'),'qcB_Final_BSyMP_Histogram_NotFlat','qcB_Final_BSyMP_Histogram_Flat','qcB_Final_BMR_Histogram_NotFlat','qcB_Final_BMR_Histogram_Flat','-v7.3');

% Save the results as .csv files
%--------------------------------------------------------------------------
SaveDirCSV = fullfile(DirSummary,'CSVs'); mkdirIfNotExist(SaveDirCSV);
CellMatFiles = {'F_All.mat','qsTestPredError_All.mat','qbError_All.mat','qcB_Init_All_Histogram.mat','qcB_Final_All_Histogram.mat'};
MatFiles2CSV(DirSummary,CellMatFiles,SaveDirCSV);

% Make figures
%--------------------------------------------------------------------------
DirFigures = fullfile(DirSummary,'KeyFigures_RL'); mkdirIfNotExist(DirFigures);

%%------Free Energy vs-plot------%%
vsPlotAllComb(@vsPlot,F_Full,F_BSyMP,F_BMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,...
	DirFigures,'Free Energy','FreeEnergy');

%%------qsPredTestError vs-plot------%%
vsPlotAllComb(@vsPlot,qsTestPredError_Full,qsTestPredError_BSyMP,qsTestPredError_BMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,...
	DirFigures,'<|qsPred-sProb|/|sProb|>','qsTestPredError');

%%------qaError vs-plot------%%
vsPlotAllComb(@vsPlot,qbError_Full,qbError_BSyMP,qbError_BMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,...
	DirFigures,'<|qB-B|/|B|>','qbError');

%%------Distribution of qcB------%%
PlotAllDistributions(qcB_Init_BSyMP,qcB_Final_BSyMP,qcB_Init_BMR,qcB_Final_BMR,...
    Q_BSyMP(1).Label,Q_BMR(1).Label,...
    DirFigures,'Distribution of qcB','qcB_Distribution');


%% nested functions
function MatNorm = NormalizeArray(ArrayIn,field)
if isstruct(ArrayIn)||nargin>1
	Mat = cat(5,ArrayIn.(field));
	MatNorm = Mat./sum(Mat,1);
elseif iscell(ArrayIn)
	CellNorm = cellfun(@(mat)(mat./sum(mat,1)),ArrayIn,'UniformOutput',false);
	MatNorm = cell2mat(reshape(CellNorm,1,1,1,1,length(CellNorm)));
else
	MatNorm = ArrayIn./sum(ArrayIn,1);
end
end

function vsPlotAllComb(funPlot,Val1,Val2,Val3,Label1,Label2,Label3,DirSave,FigTitle,FileTitle)
set(0,'units','pixels'); screen = get(0,'ScreenSize'); fullscreen = [0,0,screen(3:4)];
figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,FigTitle);
nexttile; funPlot(Val1,Val2,Label1,Label2);
nexttile; funPlot(Val1,Val3,Label1,Label3);
nexttile; funPlot(Val2,Val3,Label2,Label3);
saveas(gcf,fullfile(DirSave,[FileTitle,'.png']));
saveas(gcf,fullfile(DirSave,[FileTitle,'.fig']));
end

function vsPlot(ValX,ValY,LabelX,LabelY)
ValX = reshape(ValX,[],1); ValY = reshape(ValY,[],1); N = length(ValX);
scatter(ValX,ValY); hold on;
axMin = min([ValX,ValY],[],'all'); axMax = max([ValX,ValY],[],'all');
plot(linspace(axMin,axMax,1000),linspace(axMin,axMax,1000)); axis([axMin,axMax,axMin,axMax]);
title([LabelX,' vs ',LabelY,', N=',num2str(N)]);
xlabel(LabelX); ylabel(LabelY);
p = signrank(ValX,ValY);
xpos = 0.1; ypos = xpos;
text(axMin+xpos*(axMax-axMin),axMax-ypos*(axMax-axMin),['p value =  ',num2str(p)]);
end

function PlotAllDistributions(ValInit1,ValFinal1,ValInit2,ValFinal2,Label1,Label2,DirSave,FigTitle,FileTitle)
set(0,'units','pixels'); screen = get(0,'ScreenSize'); fullscreen = [0,0,screen(3:4)];
figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,FigTitle);
nexttile; PlotDistribution(ValInit1,ValFinal1,Label1);
nexttile; PlotDistribution(ValInit2,ValFinal2,Label2);
saveas(gcf,fullfile(DirSave,[FileTitle,'.png']));
saveas(gcf,fullfile(DirSave,[FileTitle,'.fig']));
end

function PlotDistribution(ValInit,ValFinal,Label)
ValInit_Red = [cell2mat(ValInit(:,1,1));cell2mat(ValInit(:,2,1));cell2mat(ValInit(:,3,1))];
ValInit_Blue = [cell2mat(ValInit(:,1,2));cell2mat(ValInit(:,2,2));cell2mat(ValInit(:,3,2))];
ValFinal_Red = [cell2mat(ValFinal(:,1,1));cell2mat(ValFinal(:,2,1));cell2mat(ValFinal(:,3,1))];
ValFinal_Blue = [cell2mat(ValFinal(:,1,2));cell2mat(ValFinal(:,2,2));cell2mat(ValFinal(:,3,2))];
edges = 0:0.05:1;
h = histogram(ValInit_Red,edges); h.FaceColor = 'none'; h.EdgeColor = 'r'; h.LineStyle = '--'; h.Normalization = 'probability'; hold on;
h = histogram(ValInit_Blue,edges); h.FaceColor = 'none'; h.EdgeColor = 'b'; h.LineStyle = '--'; h.Normalization = 'probability'; hold on;
h = histogram(ValFinal_Red,edges); h.FaceColor = 'r'; h.Normalization = 'probability'; hold on;
h = histogram(ValFinal_Blue,edges); h.FaceColor = 'b'; h.Normalization = 'probability'; hold off;
title(['Distribution of qcB of ',Label]); legend('qcB to be retained at initial session','qcB to be reduced at initial session','qcB to be retained at final session','qcB to be reduced at final session');
end