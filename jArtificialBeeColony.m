%[2007]-"A powerful and efficient algorithm for numerical function 
%optimization: artificial bee colony (ABC) algorithm"

% (9/12/2020)

function ABC = jArtificialBeeColony(feat,label,opts)
% Parameters
lb        = 0;
ub        = 1; 
thres     = 0.5; 
max_limit = 5;     % Maximum limits allowed

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'max'), max_limit = opts.max; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Divide into employ and onlooker bees
N   = N / 2; 
% Initial 
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness (9)
fit  = zeros(1,N);
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Best food source 
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:); 
  end
end
% Pre
limit = zeros(N,1); 
V     = zeros(N,dim); 

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iteration
while t <= max_Iter
  % {1} Employed bee phase
  for i = 1:N
    % Choose k randomly, but not equal to i
    k = [1 : i-1, i+1 : N]; 
    k = k(randi([1, numel(k)]));   
    for d = 1:dim
      % Phi in [-1,1]
      phi    = -1 + 2 * rand(); 
      % Position update (6)
      V(i,d) = X(i,d) + phi * (X(i,d) - X(k,d)); 
    end
    % Boundary
    XB = V(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    V(i,:) = XB; 
  end
  % Fitness
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,(V(i,:) > thres),opts);   
    % Compare neighbor bee 
    if Fnew <= fit(i)
      % Update bee & reset limit counter
      X(i,:)   = V(i,:);
      fit(i)   = Fnew;
      limit(i) = 0;
    else
      % Update limit counter 
      limit(i) = limit(i) + 1;
    end
  end
  % Minimization problem (5)
  Ifit = 1 ./ (1 + fit);
  % Convert probability (7)
  prob = Ifit / sum(Ifit);
  
  % {2} Onlooker bee phase 
  i = 1;
  m = 1;
  while m <= N
    if rand() < prob(i)
      % Choose k randomly, but not equal to i
      k = [1 : i-1, i+1 : N]; 
      k = k(randi([1, numel(k)]));     
      for d = 1:dim
        % Phi in [-1,1]
        phi    = -1 + 2 * rand(); 
        % Position update (6)
        V(i,d) = X(i,d) + phi * (X(i,d) - X(k,d)); 
      end
      % Boundary
      XB = V(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      V(i,:) = XB;
      % Fitness
      Fnew = fun(feat,label,(V(i,:) > thres),opts); 
      % Greedy selection
      if Fnew <= fit(i)
        X(i,:)   = V(i,:); 
        fit(i)   = Fnew;
        limit(i) = 0;
        % Re-compute new probability (5,7)
        Ifit = 1 ./ (1 + fit); 
        prob = Ifit / sum(Ifit);
      else
        limit(i) = limit(i) + 1;
      end
      m = m + 1; 
    end
    % Reset i 
    i = i + 1; 
    if i > N
      i = 1; 
    end
  end
  
  % {3} Scout bee phase
  for i = 1:N
    if limit(i) == max_limit
      for d = 1:dim
        % Produce new bee (8)
        X(i,d) = lb + (ub - lb) * rand();
      end
      % Reset Limit
      limit(i) = 0;
      % Fitness
      fit(i)   = fun(feat,label,(X(i,:) > thres),opts);
    end
    % Best food source 
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:); 
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (ABC)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
ABC.sf = Sf; 
ABC.ff = sFeat;
ABC.nf = length(Sf); 
ABC.c  = curve;
ABC.f  = feat; 
ABC.l  = label;
end


