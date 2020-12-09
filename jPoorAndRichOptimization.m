%[2019]-"Poor and rich optimization algorithm: A new human-based and 
%multi populations algorithm"

% (8/12/2020)

function PRO = jPoorAndRichOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
Pmut  = 0.06;   % mutation probability

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'Pmut'), Pmut = opts.Pmut; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Double population size: Main = Poor + Rich (1)
N   = N + N;
% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand(); 
  end
end
% Fitness
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Sort poor & rich (2)
[fit, idx] = sort(fit,'ascend'); 
X          = X(idx,:); 
% Pre
XRnew   = zeros(N / 2, dim); 
XPnew   = zeros(N / 2, dim); 
fitRnew = zeros(1, N / 2); 
fitPnew = zeros(1, N / 2); 

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iteration
while t <= max_Iter
  % Divide poor & rich
  XR   = X(1 : N / 2, :); 
  fitR = fit(1 : N / 2);
  XP   = X(N / 2 + 1 : N, :); 
  fitP = fit(N / 2 + 1 : N);
  % Select best rich individual
  [~, idxR] = min(fitR);
  XR_best   = XR(idxR,:);
  % Select best poor individual
  [~, idxP] = min(fitP);
  XP_best   = XP(idxP,:);
  % Compute mean of rich
  XR_mean   = mean(XR,1); 
  % Compute worst of rich
  [~, idxW] = max(fitR); 
  XR_worst  = XR(idxW,:);
  % [Rich population] 
  for i = 1 : N / 2
    for d = 1:dim
      % Generate new rich (3)
      XRnew(i,d) = XR(i,d) + rand() * (XR(i,d) - XP_best(d));
      % Mutation (6)
      if rand() < Pmut
        % Normal random number with mean = 0 & sd = 1 
        G = 0 + 1 * randn();
        % Mutation
        XRnew(i,d) = XRnew(i,d) + G;
      end
    end
    % Boundary 
    XB = XRnew(i,:); XB(XB > ub) = ub; XB(XB <lb) = lb; 
    XRnew(i,:) = XB;
    % Fitness of new rich 
    fitRnew(i) = fun(feat,label,(XRnew(i,:) > thres),opts);
  end  
  % [Poor population] 
  for i = 1 : N / 2
    for d = 1:dim
      % Calculate pattern (5)
      pattern    = (XR_best(d) + XR_mean(d) + XR_worst(d)) / 3; 
      % Generate new poor (4)
      XPnew(i,d) = XP(i,d) + (rand() * pattern - XP(i,d));
      % Mutation (7)
      if rand() < Pmut
        % Normal random number with mean = 0 & sd = 1 
        G = 0 + 1 * randn();
        % Mutation
        XPnew(i,d) = XPnew(i,d) + G;
      end
    end
    % Boundary 
    XB = XPnew(i,:); XB(XB > ub) = ub; XB(XB <lb) = lb;
    XPnew(i,:) = XB;
    % Fitness of new poor 
    fitPnew(i) = fun(feat,label,(XPnew(i,:) > thres),opts);
  end
  % Merge all four groups
  X   = [XR; XP; XRnew; XPnew];
  fit = [fitR, fitP, fitRnew, fitPnew];
  % Select the best N individual
  [fit, idx] = sort(fit,'ascend');
  fit        = fit(1:N);
  X          = X(idx(1:N),:);
  % Best update
  if fit(1) < fitG
    fitG = fit(1);
    Xgb  = X(1,:);
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (PRO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
PRO.sf = Sf; 
PRO.ff = sFeat; 
PRO.nf = length(Sf);
PRO.c  = curve; 
PRO.f  = feat; 
PRO.l  = label;
end

