# Wrapper Feature Selection Toolbox
---
> "Toward talent scientist: Sharing and learning together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

## Description

* This toolbox offers more than 40 wrapper feature selection methods
    + The < A_Main.m file > provides the demostrations on benchmark dataset. 


* Main goals of this toolbox are:
    + Knowledge sharing on wrapper feature selection  
    + Assists others in data mining projects

### Example 1
```code 
%% Particle Swarm Optimization (PSO) 
clear, clc, close;

% Parameters settings
opts.k  = 5; 
ho      = 0.2;
opts.N  = 10;     
opts.T  = 100;   
opts.c1 = 2;
opts.c2 = 2;
opts.w  = 0.9;

% Prepare data
load ionosphere.mat; 
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 

% Feature selection 
FS = jfs('pso',feat,label,opts);

% Define index of selected features
sf_idx = FS.sf;

% Accuracy  
Acc = jknn(feat(:,sf_idx),label,opts);

```

## Requirement

* MATLAB 2014 or above 
* Statistics and Machine Learning Toolbox

## List of available methods
* Note that the methods are altered so that they can be used in feature selection tasks. 
* The extra parameters represent the parameter(s) other than population size and maximum number of iteration
* Click on the name of method to view the detailed parameters
* Use the *opts* to set the specific parameters

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
 
    


