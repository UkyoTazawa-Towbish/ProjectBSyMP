%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Repeat the main routine (Main_BSS.m) 
% and make figures that summarize the results of repeated simulations.
%__________________________________________________________________________

clear; close all;

nShapeIDs = 3; % number of ShapeIDs, should be 3 for the most of cases
N = 100;  % number of repetitions
threshLow = 0.05; threshHigh = 0.5; % threshold for analysis of "synaptic genesis"

for ShapeID = 1:nShapeIDs
	LabelID = ['ShapeID_',num2str(ShapeID)];
	SaveDirParent = fullfile(pwd,['Results_BSS_',LabelID]); mkdirIfNotExist(SaveDirParent);
	DiaryFileName = fullfile(SaveDirParent,['Diary_',char(datetime('now','TimeZone','local','Format','yyyyMMdd_HHmm')),'.txt']);
	diary(DiaryFileName);

	tStart = tic;
	disp(['    ShapeID=',num2str(ShapeID),': Start working!']);
	fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
	for n=1:N
		tic; disp(['n=',num2str(n),': Start working!']);

		% Run main routine
		%------------------------------------------------------------------
		[P,P_test,Q_Full,Q_BSyMP,Q_BMR,Q_OnlineBMR] = Main_BSS(SaveDirParent,ShapeID);

		% Prepare variables to store the results only after the 1st run
		%------------------------------------------------------------------
		if n==1 && ShapeID == 1
			Ns = P(1).Ns; No = P(1).No;
			tmp = NaN(N,nShapeIDs);
			F_Full = tmp; F_BSyMP = tmp; F_BMR = tmp; F_OnlineBMR = tmp;
			qsTestError_Full = tmp; qsTestError_BSyMP = tmp; qsTestError_BMR = tmp; qsTestError_OnlineBMR = tmp;
			qaError_Full = tmp; qaError_BSyMP = tmp; qaError_BMR = tmp; qaError_OnlineBMR = tmp;
			qcA_nUnderThreshLow = tmp; qcA_nLowToHigh = tmp; % variables regarding analysis of "synaptic genesis"
			tmp = cell(N,nShapeIDs,2);
			qcA_10th_BSyMP = tmp; qcA_10th_BMR = tmp; qcA_10th_OnlineBMR = tmp;
			qcA_Final_BSyMP = tmp; qcA_Final_BMR = tmp; qcA_Final_OnlineBMR = tmp;
			clear tmp;
		end

		% Store free energy in arrays
		%------------------------------------------------------------------
		F_Full(n,ShapeID) = Q_Full(end).F(end);
		F_BSyMP(n,ShapeID) = Q_BSyMP(end).F(end);
		F_BMR(n,ShapeID) = Q_BMR(end).F(end);
		F_OnlineBMR(n,ShapeID) = Q_OnlineBMR(end).F(end);

		% Store test errors in arrays
		%------------------------------------------------------------------
		Norm_s = sqrt(Ns);
		qsTestError_Full(n,ShapeID) = mean(sqrt(sum((Q_Full(end).sTest-P_test.s).^2,[1,2]))/Norm_s);
		qsTestError_BSyMP(n,ShapeID) = mean(sqrt(sum((Q_BSyMP(end).sTest-P_test.s).^2,[1,2]))/Norm_s);
		qsTestError_BMR(n,ShapeID) = mean(sqrt(sum((Q_BMR(end).sTest-P_test.s).^2,[1,2]))/Norm_s);
		qsTestError_OnlineBMR(n,ShapeID) = mean(sqrt(sum((Q_OnlineBMR(end).sTest-P_test.s).^2,[1,2]))/Norm_s);

		% Store learning errors in arrays
		%------------------------------------------------------------------
		Norm_A = sqrt(sum((P(1).A).^2,'all'));
		qaError_Full(n,ShapeID) = sqrt(sum((NormalizeArray(Q_Full(end),'a') - P(end).A).^2,'all'))/Norm_A;
		ApowC = (Q_BSyMP(end).a).^permute(repmat(Q_BSyMP(end).cA,1,1,2,2),[3,4,1,2]); % A to the power of C
		qaError_BSyMP(n,ShapeID) = sqrt(sum((NormalizeArray(ApowC) - P(end).A).^2,'all'))/Norm_A;
		qaError_BMR(n,ShapeID) = sqrt(sum((NormalizeArray(Q_BMR(end),'a') - P(end).A).^2,'all'))/Norm_A;
		qaError_OnlineBMR(n,ShapeID) = sqrt(sum((NormalizeArray(Q_OnlineBMR(end),'a') - P(end).A).^2,'all'))/Norm_A;

		% Store qcA in arrays
		%------------------------------------------------------------------
		LogeVec = [P(1).LogiVecCircle,P(1).LogiVecRect];
		h = 10;
		qcA_10th_BSyMP{n,ShapeID,1} = Q_BSyMP(h).cA(LogeVec); qcA_10th_BSyMP{n,ShapeID,2} = Q_BSyMP(h).cA(~LogeVec);
		qcA_10th_BMR{n,ShapeID,1} = Q_BMR(h).MatRetainedA(LogeVec); qcA_10th_BMR{n,ShapeID,2} = Q_BMR(h).MatRetainedA(~LogeVec);
		qcA_10th_OnlineBMR{n,ShapeID,1} = Q_OnlineBMR(h).MatRetainedA(LogeVec); qcA_10th_OnlineBMR{n,ShapeID,2} = Q_OnlineBMR(h).MatRetainedA(~LogeVec);
		qcA_Final_BSyMP{n,ShapeID,1} = Q_BSyMP(end).cA(LogeVec); qcA_Final_BSyMP{n,ShapeID,2} = Q_BSyMP(end).cA(~LogeVec);
		qcA_Final_BMR{n,ShapeID,1} = Q_BMR(end).MatRetainedA(LogeVec); qcA_Final_BMR{n,ShapeID,2} = Q_BMR(end).MatRetainedA(~LogeVec);
		qcA_Final_OnlineBMR{n,ShapeID,1} = Q_OnlineBMR(end).MatRetainedA(LogeVec); qcA_Final_OnlineBMR{n,ShapeID,2} = Q_OnlineBMR(end).MatRetainedA(~LogeVec);

		% Compute the number of "synaptic pruning"  and store the result
		%------------------------------------------------------------------
		[qcA_nLowToHigh(n,ShapeID),qcA_nUnderThreshLow(n,ShapeID)] = countSyanpricGenesis(cat(3,Q_BSyMP.cA),threshLow,threshHigh);

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
qcA_10th_BSyMP_Histogram_NotFlat = [cell2mat(qcA_10th_BSyMP(:,1,1));cell2mat(qcA_10th_BSyMP(:,2,1));cell2mat(qcA_10th_BSyMP(:,3,1))];
qcA_10th_BMR_Histogram_NotFlat = [cell2mat(qcA_10th_BMR(:,1,1));cell2mat(qcA_10th_BMR(:,2,1));cell2mat(qcA_10th_BMR(:,3,1))];
qcA_10th_OnlineBMR_Histogram_NotFlat = [cell2mat(qcA_10th_OnlineBMR(:,1,1));cell2mat(qcA_10th_OnlineBMR(:,2,1));cell2mat(qcA_10th_OnlineBMR(:,3,1))];
qcA_10th_BSyMP_Histogram_Flat = [cell2mat(qcA_10th_BSyMP(:,1,2));cell2mat(qcA_10th_BSyMP(:,2,2));cell2mat(qcA_10th_BSyMP(:,3,2))];
qcA_10th_BMR_Histogram_Flat = [cell2mat(qcA_10th_BMR(:,1,2));cell2mat(qcA_10th_BMR(:,2,2));cell2mat(qcA_10th_BMR(:,3,2))];
qcA_10th_OnlineBMR_Histogram_Flat = [cell2mat(qcA_10th_OnlineBMR(:,1,2));cell2mat(qcA_10th_OnlineBMR(:,2,2));cell2mat(qcA_10th_OnlineBMR(:,3,2))];
qcA_Final_BSyMP_Histogram_NotFlat = [cell2mat(qcA_Final_BSyMP(:,1,1));cell2mat(qcA_Final_BSyMP(:,2,1));cell2mat(qcA_Final_BSyMP(:,3,1))];
qcA_Final_BMR_Histogram_NotFlat = [cell2mat(qcA_Final_BMR(:,1,1));cell2mat(qcA_Final_BMR(:,2,1));cell2mat(qcA_Final_BMR(:,3,1))];
qcA_Final_OnlineBMR_Histogram_NotFlat = [cell2mat(qcA_Final_OnlineBMR(:,1,1));cell2mat(qcA_Final_OnlineBMR(:,2,1));cell2mat(qcA_Final_OnlineBMR(:,3,1))];
qcA_Final_BSyMP_Histogram_Flat = [cell2mat(qcA_Final_BSyMP(:,1,2));cell2mat(qcA_Final_BSyMP(:,2,2));cell2mat(qcA_Final_BSyMP(:,3,2))];
qcA_Final_BMR_Histogram_Flat = [cell2mat(qcA_Final_BMR(:,1,2));cell2mat(qcA_Final_BMR(:,2,2));cell2mat(qcA_Final_BMR(:,3,2))];
qcA_Final_OnlineBMR_Histogram_Flat = [cell2mat(qcA_Final_OnlineBMR(:,1,2));cell2mat(qcA_Final_OnlineBMR(:,2,2));cell2mat(qcA_Final_OnlineBMR(:,3,2))];

% Save the results as .mat files
%--------------------------------------------------------------------------
DirSummary = fullfile(pwd,'Summary_BSS'); mkdirIfNotExist(DirSummary);
save(fullfile(DirSummary,'F_All.mat'),'F_Full','F_BSyMP','F_BMR','F_OnlineBMR','-v7.3');
save(fullfile(DirSummary,'qsTestError_All.mat'),'qsTestError_Full','qsTestError_BSyMP','qsTestError_BMR','qsTestError_OnlineBMR','-v7.3');
save(fullfile(DirSummary,'qaError_All.mat'),'qaError_Full','qaError_BSyMP','qaError_BMR','qaError_OnlineBMR','-v7.3');
save(fullfile(DirSummary,'qcA_10th_All.mat'),'qcA_10th_BSyMP','qcA_10th_BMR','qcA_10th_OnlineBMR','-v7.3');
save(fullfile(DirSummary,'qcA_Final_All.mat'),'qcA_Final_BSyMP','qcA_Final_BMR','qcA_Final_OnlineBMR','-v7.3');
save(fullfile(DirSummary,'qcA_10th_All_Histogram.mat'),'qcA_10th_BSyMP_Histogram_NotFlat','qcA_10th_BSyMP_Histogram_Flat','qcA_10th_BMR_Histogram_NotFlat','qcA_10th_BMR_Histogram_Flat','qcA_10th_OnlineBMR_Histogram_NotFlat','qcA_10th_OnlineBMR_Histogram_Flat','-v7.3');
save(fullfile(DirSummary,'qcA_Final_All_Histogram.mat'),'qcA_Final_BSyMP_Histogram_NotFlat','qcA_Final_BSyMP_Histogram_Flat','qcA_Final_BMR_Histogram_NotFlat','qcA_Final_BMR_Histogram_Flat','qcA_Final_OnlineBMR_Histogram_NotFlat','qcA_Final_OnlineBMR_Histogram_Flat','-v7.3');
save(fullfile(DirSummary,'NumberOfSynapticGenesis.mat'),'qcA_nUnderThreshLow','qcA_nLowToHigh','-v7.3');

% Save the results as .csv files
%--------------------------------------------------------------------------
SaveDirCSV = fullfile(DirSummary,'CSVs'); mkdirIfNotExist(SaveDirCSV);
CellMatFiles = {'F_All.mat','qsTestError_All.mat','qaError_All.mat','qcA_10th_All_Histogram.mat','qcA_Final_All_Histogram.mat'};
MatFiles2CSV(DirSummary,CellMatFiles,SaveDirCSV);

% Make figures
%--------------------------------------------------------------------------
DirFigures = fullfile(DirSummary,'KeyFigures_BSS'); mkdirIfNotExist(DirFigures);

%%------Free Energy vs-plot------%%
vsPlotAllComb(@vsPlot,F_Full,F_BSyMP,F_BMR,F_OnlineBMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,Q_OnlineBMR(1).Label,...
	DirFigures,'Free Energy','FreeEnergy');

%%------qsTestError vs-plot------%%
vsPlotAllComb(@vsPlot,qsTestError_Full,qsTestError_BSyMP,qsTestError_BMR,qsTestError_OnlineBMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,Q_OnlineBMR(1).Label,...
	DirFigures,'<|qs-s|/|s|>','qsTestError');

%%------qaError vs-plot------%%
vsPlotAllComb(@vsPlot,qaError_Full,qaError_BSyMP,qaError_BMR,qaError_OnlineBMR,...
	Q_Full(1).Label,Q_BSyMP(1).Label,Q_BMR(1).Label,Q_OnlineBMR(1).Label,...
	DirFigures,'<|qA-A|/|A|>','qaError');

%%------Distribution of qcA------%%
PlotAllDistributions(qcA_10th_BSyMP,qcA_Final_BSyMP,qcA_10th_BMR,qcA_Final_BMR,qcA_10th_OnlineBMR,qcA_Final_OnlineBMR,...
    Q_BSyMP(1).Label,Q_BMR(1).Label,Q_OnlineBMR(1).Label,...
    DirFigures,'Distribution of qcA','qcA_Distribution');

%%------Number of "synaptic pruning"------%%
figure();
scatter(qcA_nUnderThreshLow,qcA_nLowToHigh);
xlabel('nUnderThreshLow'); ylabel('nLowToHigh'); title('Number Of Synaptic Genesis"');
saveas(gcf,fullfile(DirFigures,'NumberOfSynapticGenesis.png'));
saveas(gcf,fullfile(DirFigures,'NumberOfSynapticGenesis.fig'));


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

function [qcA_nLowToHigh,qcA_nUnderThreshLow] = countSyanpricGenesis(qcA_Timecourse,threshLow,threshHigh)
isUnderThreshLow = qcA_Timecourse < threshLow;
isAboveThreshHigh = qcA_Timecourse > threshHigh;

firstIdxUnderThreshLow = findFirstAlongSession(isUnderThreshLow);
qcA_nUnderThreshLow = sum(~isnan(firstIdxUnderThreshLow),'all');

qcA_nLowToHigh = 0;
for i = 1:size(qcA_Timecourse,1)
	for j = 1:size(qcA_Timecourse,2)
		if isnan(firstIdxUnderThreshLow(i,j)); continue; end
		qcA_nLowToHigh = qcA_nLowToHigh + any(isAboveThreshHigh(i,j,firstIdxUnderThreshLow(i,j):end));
	end
end
end

function firstIdx = findFirstAlongSession(logiVec)
firstIdx = NaN(size(logiVec,1:2));
for i = 1:size(logiVec,1)
	for j = 1:size(logiVec,2)
		idx = find(squeeze(logiVec(i,j,:)),1);
		if isempty(idx)
			firstIdx(i,j) = NaN;
		else
			firstIdx(i,j) = idx;
		end
	end
end
end

function vsPlotAllComb(funPlot,Val1,Val2,Val3,Val4,Label1,Label2,Label3,Label4,DirSave,FigTitle,FileTitle)
set(0,'units','pixels'); screen = get(0,'ScreenSize'); fullscreen = [0,0,screen(3:4)];
figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,FigTitle);
nexttile; funPlot(Val1,Val2,Label1,Label2);
nexttile; funPlot(Val1,Val3,Label1,Label3);
nexttile; funPlot(Val1,Val4,Label1,Label4);
nexttile; funPlot(Val2,Val3,Label2,Label3);
nexttile; funPlot(Val2,Val4,Label2,Label4);
nexttile; funPlot(Val3,Val4,Label3,Label4);
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

function PlotAllDistributions(ValInit1,ValFinal1,ValInit2,ValFinal2,ValInit3,ValFinal3,Label1,Label2,Label3,DirSave,FigTitle,FileTitle)
set(0,'units','pixels'); screen = get(0,'ScreenSize'); fullscreen = [0,0,screen(3:4)];
figure('Position',fullscreen); tile = tiledlayout('flow'); title(tile,FigTitle);
nexttile; PlotDistribution(ValInit1,ValFinal1,Label1);
nexttile; PlotDistribution(ValInit2,ValFinal2,Label2);
nexttile; PlotDistribution(ValInit3,ValFinal3,Label3);
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
title(['Distribution of qcA of ',Label]); legend('qcA to be retained at 10th session','qcA to be reduced at 10th session','qcA to be retained at final session','qcA to be reduced at final session');
end