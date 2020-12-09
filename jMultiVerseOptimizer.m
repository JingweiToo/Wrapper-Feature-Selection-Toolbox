%[2015]-"Multi-verse optimizer: A nature-inspired algorithm for global
%optimization"

% (9/12/2020)

function MVO = jMultiVerseOptimizer(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
p     = 6;      % control TDR
Wmax  = 1;      % maximum WEP
Wmin  = 0.2;    % minimum WEP
type  = 1;      

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'p'), p = opts.p; end 
if isfield(opts,'Wmin'), Wmin = opts.Wmin; end 
if isfield(opts,'Wmax'), Wmax = opts.Wmax; end 
if isfield(opts,'ty'), type = opts.ty; end
if isfield(opts,'thres'), thres = opts.thres; end

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
% Pre
fit  = zeros(1,N); 
fitG = inf;

curve = inf;
t = 1; 
% Iterations
while t <= max_Iter
  % Calculate inflation rate
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best universe
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  % Sort universe from best to worst
  [fitSU, idx] = sort(fit,'ascend'); 
  X_SU         = X(idx,:); 
  % Elitism (first 1 is elite)
  X(1,:) = X_SU(1,:);
  % Either 1-norm or 2-norm 
  if type == 1  
    % Normalize inflation rate using 2-norm
    NI = fitSU ./ sqrt(sum(fitSU .^ 2)); 
  elseif type == 2
    % Normalize inflation rate using 1-norm
    NI = fitSU / sum(fitSU);
  end
  % Normalize inverse inflation rate using 1-norm
  inv_fitSU = 1 ./ (1 + fitSU); 
  inv_NI    = inv_fitSU / sum(inv_fitSU);
  % Wormhole Existence probability (3.3), increases from 0.2 to 1
  WEP = Wmin + t * ((Wmax - Wmin) / max_Iter);
  % Travelling disrance rate (3.4), descreases from 0.6 to 0
  TDR = 1 - ((t ^ (1 / p)) / (max_Iter ^ (1 / p)));
  % Start with 2 since first is elite
  for i = 2:N
    % Define black hole
    idx_BH = i;
    for d = 1:dim
      % White/black hole tunnels & exchange object of universes (3.1)
      r1 = rand();
      if r1 < NI(i)
        % Random select k with roulette wheel
        idx_WH       = jRouletteWheelSelection(inv_NI);
        % Position update
        X(idx_BH, d) = X_SU(idx_WH, d);
      end
      % Local changes for universes (3.2)
      r2 = rand(); 
      if r2 < WEP    
        r3 = rand(); 
        r4 = rand();
        if r3 < 0.5
          X(i,d) = Xgb(d) + TDR * ((ub - lb) * r4 + lb);
        else
          X(i,d) = Xgb(d) - TDR * ((ub - lb) * r4 + lb);
        end
      else
        X(i,d) = X(i,d);
      end
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (MVO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
MVO.sf = Sf; 
MVO.ff = sFeat;
MVO.nf = length(Sf); 
MVO.c  = curve;
MVO.f  = feat;
MVO.l  = label;
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



