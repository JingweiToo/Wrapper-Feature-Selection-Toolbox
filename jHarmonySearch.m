%[2001]-A New Heuristic Optimization Algorithm: Harmony Search"

% (9/12/2020)

function HS = jHarmonySearch(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
PAR   = 0.05;   % pitch adjusting rate
HMCR  = 0.7;    % harmony memory considering rate
bw    = 0.2;    % bandwidth

if isfield(opts,'N'), HMS = opts.N; end % harmony memory size
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'PAR'), PAR = opts.PAR; end 
if isfield(opts,'HMCR'), HMCR = opts.HMCR; end 
if isfield(opts,'bw'), bw = opts.bw; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial (13)
X   = zeros(HMS,dim); 
for i = 1 : HMS
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end
% Fitness 
fit  = zeros(1, HMS); 
fitG = inf;
for i = 1 : HMS
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Worst solution
[fitW, idx_W] = max(fit);
% Pre
Xnew = zeros(HMS, dim); 

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
while t <= max_Iter 
	for i = 1 : HMS
    for d = 1:dim
      % Harmony memory considering rate 
      if rand() < HMCR
        % Random select 1 harmony memory 
        k         = randi([1, HMS]);
        % Update new harmony using harmony memory
        Xnew(i,d) = X(k,d);
      else
        % Randomize a new harmony
        Xnew(i,d) = lb + (ub - lb) * rand();
      end
      % Pitch adjusting rate 
      if rand() < PAR
        r = rand();
        if r > 0.5 
          Xnew(i,d) = X(i,d) + rand() * bw;
        else
          Xnew(i,d) = X(i,d) - rand() * bw;
        end
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1 : HMS
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Update worst solution
    if Fnew < fitW
      fit(idx_W)    = Fnew; 
      X(idx_W,:)    = Xnew(i,:);
      % New worst solution
      [fitW, idx_W] = max(fit); 
    end
    % Global update
    if Fnew < fitG
      fitG = Fnew;
      Xgb  = Xnew(i,:);
    end
  end 
  curve(t) = fitG;
  fprintf('\nIteration %d Best (HS)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
HS.sf = Sf; 
HS.ff = sFeat;
HS.nf = length(Sf);
HS.c  = curve; 
HS.f  = feat; 
HS.l  = label;
end





