# Jx-WFST : A Wrapper Feature Selection Toolbox
---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/5dc2bdb4-ce4b-4e0e-bd6e-0237ff6ddde1/f9a9e760-64b9-4e31-9903-dffcabdf8be6/images/1607601518.JPG)


## Introduction

* This toolbox offers more than 40 wrapper feature selection methods
* The < A_Main.m file > provides the examples of how to apply these methods on benchmark dataset 
* Source code of these methods are written based on pseudocode & paper


* Main goals of this toolbox are:
    + Knowledge sharing on wrapper feature selection  
    + Assists others in data mining projects

## Usage
The main function *jfs* is adopted to perform feature selection. You may switch the algorithm by changing the 'pso' to [other abbreviations](/README.md#list-of-available-wrapper-feature-selection-methods)
* If you wish to use particle swarm optimization ( see example 1 ) then you may write
```code
FS = jfs('pso',feat,label,opts);
```
* If you want to use slime mould algorithm ( see example 2 ) then you may write
```code
FS = jfs('sma',feat,label,opts);
```

## Input
* *feat*   : feature vector matrix ( Instance *x* Features )
* *label*  : label matrix ( Instance *x* 1 )
* *opts*   : parameter settings
    + *N* : number of solutions / population size ( *for all methods* )
    + *T* : maximum number of iterations ( *for all methods* )
    + *k* : *k*-value in *k*-nearest neighbor 


## Output
* *Acc*  : accuracy of validation model
* *FS*   : feature selection model ( It contains several results )
    + *sf* : index of selected features
    + *ff* : selected features
    + *nf* : number of selected features
    + *c*  : convergence curve
    + *t*  : computational time (s)
    

## Notation
Some methods have their specific parameters ( example: PSO, GA, DE ), and if you do not set them then they will be defined as default settings
* you may open the < m.file > to view or change the parameters
* you may use *opts* to set the parameters of method ( see example 1 or refer [here](/Description.md) )
* you may also change the < jFitnessFunction.m file > 


### Example 1 : Particle Swarm Optimization ( PSO ) 
```code 
% Common parameter settings
opts.k  = 5;      % Number of k in K-nearest neighbor
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of PSO
opts.c1 = 2;
opts.c2 = 2;
opts.w  = 0.9;

% Load dataset
load ionosphere.mat;

% Ratio of validation data
ho = 0.2;
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 

% Perform feature selection 
FS = jfs('pso',feat,label,opts);

% Define index of selected features
sf_idx = FS.sf;

% Accuracy  
Acc = jknn(feat(:,sf_idx),label,opts); 

% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations'); 
ylabel('Fitness Value');
title('PSO');
```

### Example 2 : Slime Mould Algorithm ( SMA ) 
```code
% Common parameter settings
opts.k  = 5;      % Number of k in K-nearest neighbor
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations

% Load dataset
load ionosphere.mat; 

% Ratio of validation data
ho = 0.2;
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 

% Perform feature selection 
FS = jfs('sma',feat,label,opts);

% Define index of selected features
sf_idx = FS.sf;

% Accuracy  
Acc = jknn(feat(:,sf_idx),label,opts); 

% Plot convergence
plot(FS.c); grid on; 
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('SMA');
```

### Example 3 : Whale Optimization Algorithm ( WOA )
```code
% Common parameter settings
opts.k  = 5;      % Number of k in K-nearest neighbor
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameter of WOA
opts.b = 1;

% Load dataset
load ionosphere.mat; 

% Ratio of validation data
ho = 0.2;
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 

% Perform feature selection 
FS = jfs('woa',feat,label,opts);

% Define index of selected features
sf_idx = FS.sf;

% Accuracy  
Acc = jknn(feat(:,sf_idx),label,opts); 

% Plot convergence
plot(FS.c); grid on; 
xlabel('Number of Iterations'); 
ylabel('Fitness Value'); 
title('WOA');
```


## Requirement

* MATLAB 2014 or above 
* Statistics and Machine Learning Toolbox


## List of available wrapper feature selection methods
* Note that the methods are altered so that they can be used in feature selection tasks 
* The extra parameters represent the parameter(s) other than population size and maximum number of iterations
* Click on the name of method to view the extra parameter(s)
* Use the *opts* to set the specific parameter(s)


| No. | Abbreviation | Name                                                                                        | Year | Extra Parameters |
|-----|--------------|---------------------------------------------------------------------------------------------|------|------------------|
| 43  | 'mpa'        | [Marine Predators Algorithm](/Description.md#marine-predators-algorithm-mpa)                | 2020 | Yes              |
| 42  | 'gndo'       | Generalized Normal Distribution Optimization                                                | 2020 | No               |
| 41  | 'sma'        | Slime Mould Algorithm                                                                       | 2020 | No               |
| 40  | .mrfo'       | [Manta Ray Foraging Optimization](/Description.md#manta-ray-foraging-optimization-mrfo)     | 2020 | Yes              |
| 39  | 'eo'         | [Equilibrium Optimizer](/Description.md#equilibrium-optimizer-eo)                           | 2020 | Yes              |
| 38  | 'aso'        | [Atom Search Optimization](/Description.md#atom-search-optimization-aso)                    | 2019 | Yes              |
| 37  | 'hgso'       | [Henry Gas Solubility Optimization](/Description.md#henry-gas-solubility-optimization-hgso) | 2019 | Yes              |
| 36  | 'hho'        | Harris Hawks Optimization                                                                   | 2019 | No               |
| 35  | 'pfa'        | Path Finder Algorithm                                                                       | 2019 | No               |
| 34  | 'pro'        | [Poor And Rich Optimization](/Description.md#poor-and-rich-optimization-pro)                | 2019 | Yes              |
| 33  | 'boa'        | [Butterfly Optimization Algorithm](/Description.md#butterfly-optimization-algorithm-boa)    | 2018 | Yes              |
| 32  | 'epo'        | [Emperor Penguin Optimizer](/Description.md#emperor-penguin-optimizer-epo)                  | 2018 | Yes              |
| 31  | 'tga'        | [Tree Growth Algorithm](/Description.md#tree-growth-algorithm-tga)                          | 2018 | Yes              |
| 30  | 'abo'        | [Artificial Butterfly Optimization](/Description.md#artificial-butterfly-optimization-abo)  | 2017 | Yes              |
| 29  | 'ssa'        | Salp Swarm Algorithm                                                                        | 2017 | No               |
| 28  | 'wsa'        | [Weighted Superposition Attraction](/Description.md#weighted-superposition-attraction-wsa)  | 2017 | Yes              |
| 27  | 'sbo'        | [Satin Bower Bird Optimization](/Description.md#satin-bower-bird-optimization-sbo)          | 2017 | Yes              |
| 26  | 'ja'         | Jaya Algorithm                                                                              | 2016 | No               |
| 25  | 'csa'        | [Crow Search Algorithm](/Description.md#crow-search-algorithm-csa)                          | 2016 | Yes              |
| 24  | 'sca'        | [Sine Cosine Algorithm](/Description.md#sine-cosine-algorithm-sca)                          | 2016 | Yes              |
| 23  | 'woa'        | [Whale Optimization Algorithm](/Description.md#whale-optimization-algorithm-woa)            | 2016 | Yes              |
| 22  | 'alo'        | Ant Lion Optimizer                                                                          | 2015 | No               |
| 21  | 'hlo'        | [Human Learning Optimization](/Description.md#human-learning-optimization-hlo)              | 2015 | Yes              |
| 20  | 'mbo'        | [Monarch Butterfly Optimization](/Description.md#monarch-butterfly-optimization-mbo)        | 2015 | Yes              |  
| 19  | 'mfo'        | [Moth Flame Optimization](/Description.md#moth-flame-optimization-mfo)                      | 2015 | Yes              |
| 18  | 'mvo'        | [Multiverse Optimizer](/Description.md#multi-verse-optimizer-mvo)                           | 2015 | Yes              |
| 17  | 'tsa'        | [Tree Seed Algorithm](/Description.md#tree-seed-algorithm-tsa)                              | 2015 | Yes              |
| 16  | 'gwo'        | Grey Wolf Optimizer                                                                         | 2014 | No               |
| 15  | 'sos'        | Symbiotic Organisms Search                                                                  | 2014 | No               |
| 14  | 'fpa'        | [Flower Pollination Algorithm](/Description.md#flower-pollination-algorithm-fpa)            | 2012 | Yes              |
| 13  | 'foa'        | Fruitfly Optimization Algorithm                                                             | 2012 | No               |
| 12  | 'ba'         | [Bat Algorithm](/Description.md#bat-algorithm-ba)                                           | 2010 | Yes              |
| 11  | 'fa'         | [Firefly Algorithm](/Description.md#firefly-algorithm-fa)                                   | 2010 | Yes              |
| 10  | 'cs'         | [Cuckoo Search Algorithm](/Description.md#cuckoo-search-cs)                                 | 2009 | Yes              |
| 09  | 'gsa'        | [Gravitational Search Algorithm](/Description.md#gravitational-search-algorithm-gsa)        | 2009 | Yes              |
| 08  | 'abc'        | [Artificial Bee Colony](/Description.md#artificial-bee-colony-abc)                          | -    | Yes              |
| 07  | 'hs'         | [Harmony Search](/Description.md#harmony-search-hs)                                         | -    | Yes              |
| 06  | 'de'         | [Differential Evolution](/Description.md#differential-evolution-de)                         | 1997 | Yes              |
| 05  | 'aco'        | [Ant Colony Optimization](/Description.md#ant-colony-optimization-aco)                      | -    | Yes              |
| 04  | 'acs'        | [Ant Colony System](/Description.md#ant-colony-system-acs)                                  | -    | Yes              |
| 03  | 'pso'        | [Particle Swarm Optimization](/Description.md#particle-swarm-optimization-pso)              | -    | Yes              |
| 02  | 'ga' / 'gat' | [Genetic Algorithm](/Description.md#genetic-algorithm-ga)                                   | -    | Yes              |
| 01  | 'sa'         | [Simulated Annealing](/Description.md#simulated-annealing-sa)                               | -    | Yes              |
 
    


