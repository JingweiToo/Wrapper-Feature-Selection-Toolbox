%[2006]-"A GA-based feature selection and parameters optimization for
%support vector machines"

% (9/12/2020)

function GA = jGeneticAlgorithmTour(feat,label,opts)
% Parameters 
CR        = 0.8;   % crossover rate
MR        = 0.01;  % mutation rate
Tour_size = 3;     % tournament size

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'CR'), CR = opts.CR; end
if isfield(opts,'MR'), MR = opts.MR; end
if isfield(opts,'Ts'), Tour_size = opts.Ts; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);
% Initial 
X   = jInitialization(N,dim); 
% Fitness 
fit  = zeros(1,N); 
fitG = inf; 
for i = 1:N
  fit(i) = fun(feat,label,X(i,:),opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Pre
curve = zeros(1,max_Iter); 
curve(1) = fitG; 
t = 2;
% Generations
while t <= max_Iter
  % Preparation  
  Xc1   = zeros(1,dim);
  Xc2   = zeros(1,dim); 
  fitC1 = ones(1,1);
  fitC2 = ones(1,1);
  z     = 1;
  for i = 1:N
    if rand() < CR
      % Select two parents 
      k1 = jTournamentSelection(fit,Tour_size,N);
      k2 = jTournamentSelection(fit,Tour_size,N);
      % Store parents 
      P1 = X(k1,:); 
      P2 = X(k2,:);
      % Single point crossover
      ind = randi([1, dim - 1]);
      % Crossover between two parents
      Xc1(z,:) = [P1(1:ind), P2(ind + 1:dim)]; 
      Xc2(z,:) = [P2(1:ind), P1(ind + 1:dim)]; 
      % Mutation
      for d = 1:dim
        % First child
        if rand() < MR
          Xc1(z,d) = 1 - Xc1(z,d);
        end
        % Second child
        if rand() < MR
          Xc2(z,d) = 1 - Xc2(z,d);
        end        
      end
      % Fitness
      fitC1(1,z) = fun(feat,label,Xc1(z,:),opts);
      fitC2(1,z) = fun(feat,label,Xc2(z,:),opts);
      z = z + 1;
    end
  end
  % Merge population
  XX = [X; Xc1; Xc2];
  FF = [fit, fitC1, fitC2]; 
  % Select N best solution 
  [FF, idx] = sort(FF,'ascend');
  X         = XX(idx(1:N),:);
  fit       = FF(1:N);
  % Best agent
  if fit(1) < fitG
    fitG = fit(1);
    Xgb  = X(1,:);
  end
  % Save
  curve(t) = fitG; 
  fprintf('\nGeneration %d Best (GA Tournament)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos(Xgb == 1); 
sFeat = feat(:,Sf); 
% Store results
GA.sf = Sf; 
GA.ff = sFeat; 
GA.nf = length(Sf);
GA.c  = curve; 
GA.f  = feat;
GA.l  = label;
end


%// Tournament Selection //
function Index = jTournamentSelection(fit,Tour_size,N)
% Random positions based on position & Tournament Size
Tour_idx  = randsample(N,Tour_size);
% Select ftiness value based on position selected by tournament 
Tour_fit  = fit(Tour_idx); 
% Get position of best ftiness value (win tournament)
[~, idx]  = min(Tour_fit);
% Store the position
Index     = Tour_idx(idx);
end


function X = jInitialization(N,dim)
% Initialize X vectors
X = zeros(N,dim);
for i = 1:N
  for d = 1:dim 
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
end

