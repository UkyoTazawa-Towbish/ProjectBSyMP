function P = GenerateTask_RL(T,OnProb,OffProb,RuleID,FlagFig,ShowProgress)
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
nRows = 32; nCols = 96; No = nRows*nCols; Ns = 12;

%% Prepare A-matrix
A = NaN(2,2,No,Ns);
tmp = zeros(nRows*nCols,3,'logical'); LogiVecCircle = tmp; LogiVecRect = tmp; LogiVecX = tmp; LogiVecPlus = tmp;
for screen = 1:3
	% s1,s5,s9 shows a circle
	%----------------------------------------------------------------------
	center = [16,16+(screen-1)*32]; rad = 13; thickness = 3;
	LogiMatCircle = OpenCircleMat([nRows,nCols],center,rad,thickness);
	LogiVecCircle(:,screen) = Mat2Vec(LogiMatCircle);
	A(:,:,:,1+(screen-1)*4) = Shape2A(A(:,:,:,1+(screen-1)*4),LogiVecCircle(:,screen),OnProb,OffProb);

	% s2,s6,s8 shows a rectangle
	%----------------------------------------------------------------------
	pos = [2,2+(screen-1)*32]; width = 30; height = 30; thickness = 2;
	LogiMatRect = OpenRectMat([nRows,nCols],pos,width,height,thickness);
	LogiVecRect(:,screen) = Mat2Vec(LogiMatRect);
	A(:,:,:,2+(screen-1)*4) = Shape2A(A(:,:,:,2+(screen-1)*4),LogiVecRect(:,screen),OnProb,OffProb);

	% s3,s7,s11 shows an X
	%----------------------------------------------------------------------
	pos = [16,15+(screen-1)*32]; length = 12; thickness = 5;
	LogiMatX = XMat([nRows,nCols],pos,length,thickness);
	LogiVecX(:,screen) = Mat2Vec(LogiMatX);
	A(:,:,:,3+(screen-1)*4) = Shape2A(A(:,:,:,3+(screen-1)*4),LogiVecX(:,screen),OnProb,OffProb);

	% s4,s8,s12 shows a plus
	%----------------------------------------------------------------------
	pos = [16,16+(screen-1)*32]; length = 14; thickness = 4;
	LogiMatPlus = PlusMat([nRows,nCols],pos,length,thickness);
	LogiVecPlus(:,screen) = Mat2Vec(LogiMatPlus);
	A(:,:,:,4+(screen-1)*4) = Shape2A(A(:,:,:,4+(screen-1)*4),LogiVecPlus(:,screen),OnProb,OffProb);
end


%% Prepare D-matrix and B-matrix
D = NaN(2,Ns);
D(1,:) = 0.5;
D(2,:) = 1-D(1,:);
B = DefineRule(Ns,RuleID);


%% Generate data and pack them in P
FlagFigGen = false;
P = GenerateData_POMDP(A,B,D,T,FlagFigGen,ShowProgress); P.nRows = nRows; P.nCols = nCols;
P.LogiVecCircle = LogiVecCircle; P.LogiVecRect = LogiVecRect;
P.LogiVecX = LogiVecX; P.LogiVecPlus = LogiVecPlus;
if FlagFig
	nTiles = 12; t=round(linspace(1,T,nTiles));
	figure(); tile = tiledlayout('flow'); title(tile,'Observations');
	for i = 1:nTiles
		nexttile;
		o = reshape(P.o(1,:,t(i)),nRows,nCols);
		imshow(o);
		title(['t = ',num2str(t(i))]);
	end

	for screen = 1:3
		figure(); tile = tiledlayout('flow'); title(tile,['Expected Observation, screen ',num2str(screen)]);
		for state_vis = 1:4
			A_OnProb = reshape(squeeze(A(1,1,:,state_vis+(screen-1)*4)),nRows,nCols);
			A_OffProb = reshape(squeeze(A(1,2,:,state_vis+(screen-1)*4)),nRows,nCols);
			nexttile; heatmap(A_OnProb,'ColorLimits',[0,1]); title(['s',num2str(state_vis+(screen-1)*4),' = 1']);
			nexttile; heatmap(A_OffProb,'ColorLimits',[0,1]); title(['s',num2str(state_vis+(screen-1)*4),' = 0']);
		end
	end
end
end

%% nested functions
function Mat = OpenCircleMat(size,center,rad,thickness)
Mat = zeros(size,'logical');
theta = linspace(0,2*pi,500);
for i = 0:thickness-1
	x = round((rad-i)*cos(theta) + center(1));
	y = round((rad-i)*sin(theta) + center(2));
	ind = sub2ind(size,x,y);
	Mat(ind) = true;
end
end

function Mat = OpenRectMat(size,pos,width,height,thickness)
Mat = zeros(size,'logical');
for i = 0:thickness-1
	y = round((pos(1)+i):(pos(1)+(width-i)-1));
	x = round((pos(2)+i):(pos(2)+(height-i)-1));
	Mat(y(1),x) = true; Mat(y(end),x) = true;
	Mat(y,x(1)) = true; Mat(y,x(end)) = true;
end
end

function Mat = XMat(size,pos,length,thickness)
Mat = zeros(size,'logical');
y = round((pos(1)-length):(pos(1)+length));
for i = 0:thickness-1
	x = round(((pos(2)-length):(pos(2)+length))+i);
	Mat(sub2ind(size,y,x)) = true; Mat(sub2ind(size,flip(y),x)) = true;
end
end

function Mat = PlusMat(size,pos,length,thickness)
Mat = zeros(size,'logical');
y = round(((pos(1)-length):(pos(1)+length+1)));
x = round(((pos(2)-length):(pos(2)+length+1)));
for i = 0:thickness-1
	% dPos = [0,1,-1,2,-2,...]
	if i==0; dPos=0;
	elseif rem(i,2)==1; dPos = ceil(i/2);
	elseif rem(i,2)==0; dPos = -ceil(i/2);
	end
	Mat(pos(1)+dPos,x) = true; Mat(y,pos(2)+dPos) = true;
end
end

function Vec = Mat2Vec(Mat)
Vec = reshape(Mat,numel(Mat),1);
end

function A = Shape2A(A,ShapeVec,OnProb,OffProb)
A(:,:,ShapeVec) = repmat([OnProb,OffProb;1-OnProb,1-OffProb],1,1,sum(ShapeVec));
A(:,:,~ShapeVec) = 0.5;
end

function B = DefineRule(Ns,RuleID)
B = NaN(2,2,Ns,Ns);

% Note this: B(1,1,i,j) = P(s(i)_t+1 = 1 | s(j)_t = 1), B(1,2,i,j) = P(s(i)_t+1 = 1 | s(j)_t = 0)
switch RuleID
	case 1
		% define state-transition in screen 1
		B(1,1,2,3) = 0.2; B(1,1,3,3) = 0.4; B(1,1,4,3) = 0.2;

		% define state-transition in screen 2
		B(1,1,5,5) = 0.8; B(1,1,6,5) = 0.6; B(1,1,7,5) = 0.6; B(1,1,8,5) = 0.6;

		% define state-transition in screen 3
		B(1,1,9,10) = 0.2; B(1,1,10,10) = 0.2; B(1,1,11,10) = 0.4; B(1,1,12,10) = 0.3;
	case 2
		% define state-transition in screen 1
		B(1,1,1,2) = 0.2; B(1,1,1,4) = 0.7; B(1,1,2,4) = 0.7; B(1,1,3,2) = 0.2; B(1,1,3,4) = 0.7;

		% define state-transition in screen 2
		B(1,1,5,6) = 0.8; B(1,1,7,6) = 0.8; B(1,1,8,5) = 0.3; B(1,1,8,7) = 0.3;

		% define state-transition in screen 3
		B(1,1,12,10) = 0.8;

		% define state-transition between screens
		B(1,1,5,4) = 0.6; B(1,1,6,4) = 0.6; B(1,1,7,4) = 0.6;
		B(1,1,9,5) = 0.7; B(1,1,9,8) = 0.4; B(1,1,10,8) = 0.4; B(1,1,11,7) = 0.7; B(1,1,11,8) = 0.4;
	case 3
		% define state-transition in screen 1
		B(1,1,1,1) = 0.8; B(1,1,2,1) = 0.6; B(1,1,3,2) = 0.6; B(1,1,4,2) = 0.7;

		% define state-transition in screen 2
		B(1,1,5,5) = 0.8; B(1,1,6,5) = 0.6; B(1,1,7,5) = 0.6; B(1,1,8,5) = 0.6;

		% define state-transition in screen 3
		B(1,1,9,10) = 0.2; B(1,1,10,10) = 0.2; B(1,1,11,10) = 0.4; B(1,1,12,10) = 0.3;

		% define state-transition between screens
		B(1,1,4,6) = 0.6; B(1,1,8,10) = 0.3;
	otherwise; error('ShapeID should be an integer from 1 to 3');
end
% fill the rest elements
B(1,2,:,:) = 1-B(1,1,:,:); % make B symmetric
B(2,:,:,:) = 1-B(1,:,:,:); % fill the lower rows
B(isnan(B)) = 0.5; 
end