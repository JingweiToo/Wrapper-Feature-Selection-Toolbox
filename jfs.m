% Wrapper Feature Selection Toolbox by Jingwei Too - 9/12/2020

function model = jfs(type,feat,label,opts)
switch type
  % 2020
  case 'mpa'  ; fun = @jMarinePredatorsAlgorithm; 
  case 'gndo' ; fun = @jGeneralizedNormalDistributionOptimization;
  case 'sma'  ; fun = @jSlimeMouldAlgorithm; 
  case 'eo'   ; fun = @jEquilibriumOptimizer;
  case 'mrfo' ; fun = @jMantaRayForagingOptimization; 
  % 2019  
  case 'aso'  ; fun = @jAtomSearchOptimization; 
  case 'hho'  ; fun = @jHarrisHawksOptimization; 
  case 'hgso' ; fun = @jHenryGasSolubilityOptimization; 
  case 'pfa'  ; fun = @jPathFinderAlgorithm; 
  case 'pro'  ; fun = @jPoorAndRichOptimization; 
  % 2018 
  case 'boa'  ; fun = @jButterflyOptimizationAlgorithm;
  case 'epo'  ; fun = @jEmperorPenguinOptimizer; 
  case 'tga'  ; fun = @jTreeGrowthAlgorithm; 
  % 2017
  case 'abo'  ; fun = @jArtificialButterflyOptimization; 
  case 'ssa'  ; fun = @jSalpSwarmAlgorithm; 
  case 'sbo'  ; fun = @jSatinBowerBirdOptimization; 
  case 'wsa'  ; fun = @jWeightedSuperpositionAttraction; 
  % 2016
  case 'ja'   ; fun = @jJayaAlgorithm; 
  case 'csa'  ; fun = @jCrowSearchAlgorithm;
  case 'sca'  ; fun = @jSineCosineAlgorithm; 
  case 'woa'  ; fun = @jWhaleOptimizationAlgorithm;
  % 2015
  case 'alo'  ; fun = @jAntLionOptimizer; 
  case 'hlo'  ; fun = @jHumanLearningOptimization; 
  case 'mbo'  ; fun = @jMonarchButterflyOptimization; 
  case 'mfo'  ; fun = @jMothFlameOptimization;
  case 'mvo'  ; fun = @jMultiVerseOptimizer; 
  case 'tsa'  ; fun = @jTreeSeedAlgorithm;  
  % 2014 
  case 'gwo'  ; fun = @jGreyWolfOptimizer; 
  case 'sos'  ; fun = @jSymbioticOrganismsSearch; 
  % 2012
  case 'fpa'  ; fun = @jFlowerPollinationAlgorithm;
  case 'foa'  ; fun = @jFruitFlyOptimizationAlgorithm; 
  % 2009 - 2010 
  case 'ba'   ; fun = @jBatAlgorithm; 
  case 'fa'   ; fun = @jFireflyAlgorithm; 
  case 'cs'   ; fun = @jCuckooSearchAlgorithm; 
  case 'gsa'  ; fun = @jGravitationalSearchAlgorithm;
  % Traditional
  case 'abc'  ; fun = @jArtificialBeeColony; 
  case 'hs'   ; fun = @jHarmonySearch;         
  case 'de'   ; fun = @jDifferentialEvolution; 
  case 'aco'  ; fun = @jAntColonyOptimization;
  case 'acs'  ; fun = @jAntColonySystem; 
  case 'pso'  ; fun = @jParticleSwarmOptimization; 
  case 'gat'  ; fun = @jGeneticAlgorithmTour; 
  case 'ga'   ; fun = @jGeneticAlgorithm; 
  case 'sa'   ; fun = @jSimulatedAnnealing;
end
tic;
model = fun(feat,label,opts); 
% Computational time
t = toc;

model.t = t;
fprintf('\n Processing Time (s): %f % \n',t); fprintf('\n');
end


