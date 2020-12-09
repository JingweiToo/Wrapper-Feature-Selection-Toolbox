%[2015]-"The ant lion optimizer"

% (9/12/2020)

function ALO = jAntLionOptimizer(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);  
% Initial: Ant & antlion
Xal = zeros(N,dim); 
for i = 1:N
	for j = 1:dim
    Xal(i,j) = lb + (ub - lb) * rand();
	end
end
Xa = zeros(N,dim);
for i = 1:N
	for j = 1:dim
    Xa(i,j)  = lb + (ub - lb) * rand();
	end
end
% Fitness of antlion
fitAL = zeros(1,N); 
fitE  = inf;
for i = 1:N
  fitAL(i) = fun(feat,label,(Xal(i,:) > thres),opts);
  % Elite update 
  if fitAL(i) < fitE
    Xe   = Xal(i,:); 
    fitE = fitAL(i);
  end
end
% Pre
fitA = ones(1,N);

curve = zeros(1,max_Iter);
curve(1) = fitE;
t = 2; 
% Iteration
while t <= max_Iter 
	% Set weight according to iteration
	I = 1;
	if t > 0.1 * max_Iter
    w = 2;
    I = (10 ^ w) * (t / max_Iter);
	elseif t > 0.5 * max_Iter
    w = 3; 
    I = (10 ^ w) * (t / max_Iter);
	elseif t > 0.75 * max_Iter
    w = 4; 
    I = (10 ^ w) * (t / max_Iter);
	elseif t > 0.9 * max_Iter
    w = 5;
    I = (10 ^ w) * (t / max_Iter);
	elseif t > 0.95 * max_Iter
    w = 6; 
    I = (10 ^ w) * (t / max_Iter);
  end
  % Radius of ant's random walks hyper-sphere (2.10-2.11)
  c = lb / I;
  d = ub / I; 
  % Convert probability
  Ifit = 1 ./ (1 + fitAL);
  prob = Ifit / sum(Ifit); 
  for i=1:N    
    % Select one antlion using roulette wheel
    rs = jRouletteWheelSelection(prob);
    % Apply random walk of ant around antlion
    RA = jRandomWalkALO(Xal(rs,:), c, d, max_Iter, dim);
    % Apply random walk of ant around elite
    RE = jRandomWalkALO(Xe, c, d, max_Iter, dim);
    % Elitism process (2.13)
    for j = 1:dim	
      Xa(i,j) = (RA(t,j) + RE(t,j)) / 2;
    end
    % Boundary
    XB = Xa(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xa(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    % Fitness of ant
    fitA(i) = fun(feat,label,(Xa(i,:) > thres),opts);
    % Elite update 
    if fitA(i) < fitE
      Xe   = Xa(i,:); 
      fitE = fitA(i);
    end
  end
  % Update antlion position, assume ant with best fitness is consumed 
  % by antlion and the position of ant has been replaced by antlion
  % for further trap building
  XX        = [Xal; Xa]; 
  FF        = [fitAL, fitA]; 
  [FF, idx] = sort(FF,'ascend');
  Xal       = XX(idx(1:N),:);
  fitAL     = FF(1:N);
  % Save
  curve(t) = fitE; 
  fprintf('\nIteration %d Best (ALO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xe > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
ALO.sf = Sf; 
ALO.ff = sFeat;
ALO.nf = length(Sf); 
ALO.c  = curve; 
ALO.f  = feat; 
ALO.l  = label;
end


%// Roulette Wheel Selection //
function Index = jRouletteWheelSelection(prob)
% Cummulative summation
C = cumsum(prob);
% Random one value, most probability value [0~1]
P = rand();
% Route wheel
for i = 1:length(C)
	if C(i) > P
    Index = i;
    break;
  end
end
end


%// Random Walk //
function RW = jRandomWalkALO(Xal, c, d, max_Iter, dim)
% Pre
RW = zeros(max_Iter + 1, dim); 
R  = zeros(max_Iter, 1);
% Random walk with C on antlion (2.8)
if rand() > 0.5
  c = Xal + c;
else
  c = Xal - c;
end
% Random walk with D on antlion (2.9)
if rand() > 0.5
  d = Xal + d;
else
  d = Xal - d;
end
for j = 1:dim
  % Random distribution (2.2)
  for t = 1:max_Iter
    if rand() > 0.5
      R(t) = 1;
    else
      R(t) = 0;
    end
  end
  % Actual random walk (2.1)
  X = [0, cumsum((2 * R) - 1)'];
  % [a,b]-->[c,d]
  a = min(X); 
  b = max(X); 
  % Normalized (2.7)
  Xnorm = (((X - a) * (d(j) - c(j))) ./ (b - a)) + c(j);
  % Store result
  RW(:,j) = Xnorm;
end
end



