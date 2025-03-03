function P = GenerateTask_BSS(T,OnProb,OffProb,ShapeID,FlagFig,ShowProgress)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Generate data for the blind source separation task.
%
% Input
% T:			Total time steps of a session
% OnProb:		P(o=1|s=1) where there is connection between s and o
% OffProb:		P(o=1|s=0) where there is connection between s and o
% ShapeID:		Natural number (1,2,3) that designates shape for task
% FlagFig:		Flag to switch whether to make figures or not
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% P:			Info of outer world (truth)
%__________________________________________________________________________
nRows = 32; nCols = 32; No = nRows*nCols; Ns = 2; ColorLimits = [OffProb,OnProb];

%% Prepare A-matrix
A = NaN(2,2,No,Ns);

% s1 shows a circle
%--------------------------------------------------------------------------
switch ShapeID
	case 1; center = [16,16]; rad = 13;
	case 2; center = [21,12]; rad = 10;
	case 3; center = [16,16]; rad = 15;
	otherwise; error('ShapeID should be an integer from 1 to 3');
end
LogiMatCircle = OpenCircleMat([nRows,nCols],center,rad);
LogiVecCircle = Mat2Vec(LogiMatCircle); % Logical Vector specifying Circle
A(:,:,:,1) = Shape2A(A(:,:,:,1),LogiVecCircle,OnProb,OffProb);

% s2 shows a rectangle
%--------------------------------------------------------------------------
switch ShapeID
	case 1; pos = [3,3]; rect = [27,25]; % rect = [height,width]
	case 2; pos = [4,2]; rect = [26,26];
	case 3; pos = [5,7]; rect = [23,23];
	otherwise; error('ShapeID should be an integer from 1 to 3');
end
LogiMatRect = OpenRectMat([nRows,nCols],pos,rect(1),rect(2));
LogiVecRect = Mat2Vec(LogiMatRect); % Logical Vector specifying Rectangle
A(:,:,:,2) = Shape2A(A(:,:,:,2),LogiVecRect,OnProb,OffProb);


%% Prepare D-matrix and B-matrix
D = [0.5,0.5;0.5,0.5]; % initial states are randomly determined
B = ones(2,2,Ns,Ns)*0.5; % states change randomly


%% Generate data and pack them in P
FlagFigGen = false;
P = GenerateData_POMDP(A,B,D,T,FlagFigGen,ShowProgress); P.nRows = nRows; P.nCols = nCols;
P.ShapeID = ShapeID; P.LogiVecCircle = LogiVecCircle; P.LogiVecRect = LogiVecRect;
if FlagFig
	nTiles = 12; t=round(linspace(1,T,nTiles));
	figure(); tile = tiledlayout('flow'); title(tile,['Observations, ID: ',num2str(ShapeID)]);
	for i = 1:nTiles
		nexttile;
		o = reshape(P.o(1,:,t(i)),nRows,nCols);
		imshow(o);
		title(['t = ',num2str(t(i))]);
	end

	figure(); tile = tiledlayout('flow'); title(tile,['Expected Observation, ID: ',num2str(ShapeID)]);
	for state_vis = 1:Ns
		A_OnProb = reshape(squeeze(A(1,1,:,state_vis)),nRows,nCols);
		A_OffProb = reshape(squeeze(A(1,2,:,state_vis)),nRows,nCols);
		nexttile; heatmap(A_OnProb,'ColorLimits',ColorLimits); title(['s',num2str(state_vis),' = 1']);
		nexttile; heatmap(A_OffProb,'ColorLimits',ColorLimits); title(['s',num2str(state_vis),' = 0']);
	end

	figure(); tile = tiledlayout('flow'); title(tile,['Average Observation, ID: ',num2str(ShapeID)]);
	MakeFigAveObs(P);
end
end

%% nested functions
function Mat = OpenCircleMat(size,center,rad)
Mat = zeros(size,'logical');
th = linspace(0,2*pi,500);
x = round(rad*cos(th) + center(1));
y = round(rad*sin(th) + center(2));
ind = sub2ind(size,x,y);
Mat(ind) = true;
end

function Mat = OpenRectMat(size,pos,width,height)
Mat = zeros(size,'logical');
x = pos(1):(pos(1)+width-1);
y = pos(2):(pos(2)+height-1);
Mat(x(1),y) = true; Mat(x(end),y) = true; 
Mat(x,y(1)) = true; Mat(x,y(end)) = true; 
end

function Vec = Mat2Vec(Mat)
Vec = reshape(Mat,numel(Mat),1);
end

function A = Shape2A(A,ShapeVec,OnProb,OffProb)
A(:,:,ShapeVec) = repmat([OnProb,OffProb;1-OnProb,1-OffProb],1,1,sum(ShapeVec));
A(:,:,~ShapeVec) = 0.5;
end

function MakeFigAveObs(P)
idx = find(~any(P.s(1,:,:)));
if length(idx)<1; o_mean = NaN; disp('There is no time-point that satisfies [s1,s2] = [0,0]');
elseif length(idx)<2; o_mean = reshape(squeeze(P.o(1,:,idx)),P.nRows,P.nCols);
else; o_mean = reshape(mean(squeeze(P.o(1,:,idx)),2),P.nRows,P.nCols); end
nexttile; heatmap(o_mean); title('[s1,s2] = [0,0]');

idx = find(P.s(1,1,:)&~P.s(1,2,:));
if length(idx)<1; o_mean = NaN; disp('There is no time-point that satisfies [s1,s2] = [1,0]');
elseif length(idx)<2; o_mean = reshape(squeeze(P.o(1,:,idx)),P.nRows,P.nCols);
else; o_mean = reshape(mean(squeeze(P.o(1,:,idx)),2),P.nRows,P.nCols); end
nexttile; heatmap(o_mean); title('[s1,s2] = [1,0]');

idx = find(~P.s(1,1,:)&P.s(1,2,:));
if length(idx)<1; o_mean = NaN; disp('There is no time-point that satisfies [s1,s2] = [0,1]');
elseif length(idx)<2; o_mean = reshape(squeeze(P.o(1,:,idx)),P.nRows,P.nCols);
else; o_mean = reshape(mean(squeeze(P.o(1,:,idx)),2),P.nRows,P.nCols); end
nexttile; heatmap(o_mean); title('[s1,s2] = [0,1]');

idx = find(all(P.s(1,:,:)));
if length(idx)<1; o_mean = NaN; disp('There is no time-point that satisfies [s1,s2] = [1,1]');
elseif length(idx)<2; o_mean = reshape(squeeze(P.o(1,:,idx)),P.nRows,P.nCols);
else; o_mean = reshape(mean(squeeze(P.o(1,:,idx)),2),P.nRows,P.nCols); end
nexttile; heatmap(o_mean); title('[s1,s2] = [1,1]');
end