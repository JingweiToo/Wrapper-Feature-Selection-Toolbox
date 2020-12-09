%[2019]-"Henry gas solubility optimization: A novel physics-based
%algorithm"

% (8/12/2020)

function HGSO = jHenryGasSolubilityOptimization(feat,label,opts)
% Parameters
lb      = 0;
ub      = 1; 
thres   = 0.5; 
num_gas = 2;      % number of gas types / cluster
K       = 1;      % constant
alpha   = 1;      % influence of other gas
beta    = 1;      % constant 
L1      = 5E-3; 
L2      = 100; 
L3      = 1E-2;
Ttheta  = 298.15;
eps     = 0.05; 
c1      = 0.1;
c2      = 0.2; 
 
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'Nc'), num_gas = opts.Nc; end 
if isfield(opts,'K'), K = opts.K; end 
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'beta'), beta = opts.beta; end 
if isfield(opts,'L1'), L1 = opts.L1; end
if isfield(opts,'L2'), L2 = opts.L2; end
if isfield(opts,'L3'), L3 = opts.L3; end
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction;
% Number of dimensions
dim = size(feat,2); 
% Number of gas in Nc cluster
Nn  = ceil(N / num_gas); 
% Initial (6)
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
	end
end
% Henry constant & E/R constant (7)
H = zeros(num_gas,1); 
C = zeros(num_gas,1); 
P = zeros(num_gas,Nn);
for j = 1:num_gas
  H(j) = L1 * rand();
  C(j) = L3 * rand();
  for i = 1:Nn
    % Partial pressure (7)
    P(j,i) = L2 * rand();
  end
end
% Divide the population into Nc type of gas cluster
Cx = cell(num_gas,1); 
for j = 1:num_gas
  if j ~= num_gas
    Cx{j} = X(((j - 1) * Nn) + 1 : j * Nn, :);
  else
    Cx{j} = X(((num_gas - 1) * Nn + 1 : N), :);
  end
end
% Fitness of each cluster
Cfit  = cell(num_gas,1); 
fitCB = ones(1,num_gas); 
Cxb   = zeros(num_gas,dim); 
fitG  = inf;
for j = 1:num_gas
  for i = 1:size(Cx{j},1)
    Cfit{j}(i,1) = fun(feat,label,(Cx{j}(i,:) > thres),opts);
    % Update best gas
    if Cfit{j}(i) < fitCB(j)
      fitCB(j) = Cfit{j}(i);
      Cxb(j,:) = Cx{j}(i,:);
    end
    % Update global best
    if Cfit{j}(i) < fitG
      fitG = Cfit{j}(i);
      Xgb  = Cx{j}(i,:);
    end
  end
end
% Pre 
S     = zeros(num_gas,Nn); 

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iterations
while t <= max_Iter
  % Compute temperature (8)
  T = exp(-t / max_Iter); 
  for j = 1:num_gas
    % Update henry coefficient (8)
    H(j) = H(j) * exp(-C(j) * ((1 / T) - (1 / Ttheta)));
    for i = 1:size(Cx{j},1)
      % Update solubility (9)
      S(j,i) = K * H(j) * P(j,i);
      % Compute gamma (10)
      gamma  = beta * exp(-((fitG + eps) / (Cfit{j}(i) + eps)));
      % Flag change between - & +
      if rand() > 0.5
        F = -1;
      else
        F = 1; 
      end
      for d = 1:dim     
        % Random constant
        r = rand();
        % Position update (10)
        Cx{j}(i,d) = Cx{j}(i,d) + F * r * gamma * ...
          (Cxb(j,d) - Cx{j}(i,d)) + F * r * alpha * ...
          (S(j,i) * Xgb(d) - Cx{j}(i,d));
      end
      % Boundary
      XB = Cx{j}(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      Cx{j}(i,:) = XB;
    end
  end
  % Fitness
  for j = 1:num_gas
    for i = 1:size(Cx{j},1)
      % Fitness
      Cfit{j}(i,1) = fun(feat,label,(Cx{j}(i,:) > thres),opts); 
    end
  end
  % Select the worst solution (11)
  Nw = round(N * (rand() * (c2 - c1) + c1));
  % Convert cell to array
  XX = cell2mat(Cx); 
  FF = cell2mat(Cfit);
  [~, idx] = sort(FF,'descend');
  % Update position of worst solution (12)
  for i = 1:Nw
    for d = 1:dim
      XX(idx(i),d) = lb + rand() * (ub - lb);
    end
    % Fitness
    FF(idx(i)) = fun(feat,label,(XX(idx(i),:) > thres),opts);
  end
  % Divide the population into Nc type of gas cluster back
  for j = 1:num_gas
    if j ~= num_gas
      Cx{j}   = XX(((j - 1) * Nn) + 1 : j * Nn, :); 
      Cfit{j} = FF(((j - 1) * Nn) + 1 : j * Nn);
    else
      Cx{j}   = XX(((num_gas - 1) * Nn + 1 : N), :); 
      Cfit{j} = FF((num_gas - 1) * Nn + 1 : N);
    end
  end  
  % Update best solution
  for j = 1:num_gas
    for i = 1:size(Cx{j},1)
      % Update best gas
      if Cfit{j}(i) < fitCB(j)
        fitCB(j) = Cfit{j}(i);
        Cxb(j,:) = Cx{j}(i,:);
      end
      % Update global best
      if Cfit{j}(i) < fitG
        fitG = Cfit{j}(i);
        Xgb  = Cx{j}(i,:);
      end
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (HGSO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
HGSO.sf = Sf; 
HGSO.ff = sFeat; 
HGSO.nf = length(Sf); 
HGSO.c  = curve;
HGSO.f  = feat; 
HGSO.l  = label;
end


