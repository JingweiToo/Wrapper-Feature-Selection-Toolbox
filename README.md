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

| No. | Abbreviation | Name                                         | Year | Extra Parameters |
|-----|--------------|----------------------------------------------|------|------------------|
| 43  | MPA          | Marine Predators Algorithm                   | 2020 | Yes              |
| 42  | GNDO         | Generalized Normal Distribution Optimization | 2020 | No               |
| 41  | SMA          | Slime Mould Algorithm                        | 2020 | No               |
| 40  | MRFO         | Manta Ray Foraging Optimization              | 2020 | Yes              |
| 39  | EO           | Equilibrium Optimizer                        | 2020 | No               |
| 38  | ASO          | Atom Search Optimization                     | 2019 | Yes              |
| 37  | HGSO         | Henry Gas Solubility Optimization            | 2019 | Yes              |
| 36  | HHO          | Harris Hawks Optimization                    | 2019 | No               |
| 35  | PFA          | Path Finder Algorithm                        | 2019 | No               |
| 34  | PRO          | Poor And Rich Optimization                   | 2019 | Yes              |
| 33  | BOA          | Butterfly Optimization Algorithm             | 2018 | Yes              |
| 32  | EPO          | Emperor Penguin Optimizer                    | 2018 | Yes              |
| 31  | TGA          | Tree Growth Algorithm                        | 2018 | Yes              |
| 30  | ABO          | Artificial Butterfly Optimization            | 2017 | Yes              |
| 29  | SSA          | Salp Swarm Algorithm                         | 2017 | No               |
| 28  | WSA          | Weighted Superposition Attraction            | 2017 | Yes              |
| 27  | SBO          | Satin Bower Bird Optimization                | 2017 | Yes              |
| 26  | JA           | Jaya Algorithm                               | 2016 | No               |
| 25  | CSA          | Crow Search Algorithm                        | 2016 | Yes              |
| 24  | SCA          | Sine Cosine Algorithm                        | 2016 | No               |
| 23  | WOA          | Whale Optimization Algorithm                 | 2016 | Yes              |
| 22  | ALO          | Ant Lion Optimizer                           | 2015 | No               |
| 21  | HLO          | Human Learning Optimization                  | 2015 | Yes              |
| 20  | MBO          | Monarch Butterfly Optimization               | 2015 | Yes              |  
| 19  | MFO          | Moth Flame Optimization                      | 2015 | Yes              |
| 18  | MVO          | Multiverse Optimizer                         | 2015 | Yes              |
| 17  | TSA          | Tree Seed Algorithm                          | 2015 | Yes              |
| 16  | GWO          | Grey Wolf Optimizer                          | 2014 | No               |
| 15  | SOS          | Symbiotic Organisms Search                   | 2014 | No               |
| 14  | FPA          | Flower Pollination Algorithm                 | 2012 | Yes              |
| 13  | FOA          | Fruitfly Optimization Algorithm              | 2012 | No               |
| 12  | BA           | Bat Algorithm                                | 2010 | Yes              |
| 11  | FA           | Firefly Algorithm                            | 2010 | Yes              |
| 10  | CS           | Cuckoo Search Algorithm                      | 2009 | Yes              |
| 09  | GSA          | Gravitational Search Algorithm               | 2009 | Yes              |
| 08  | ABC          | Artificial Bee Colony                        | 2007 | Yes              |
| 07  | HS           | Harmony Search                               | 2001 | Yes              |
| 06  | DE           | Differential Evolution                       | 1997 | Yes              |
| 05  | ACO          | Ant Colony Optimization                      | -    | Yes              |
| 04  | ACS          | Ant Colony System                            | -    | Yes              |
| 03  | PSO          | Particle Swarm Optimization                  | -    | Yes              |
| 02  | GA           | Genetic Algorithm                            | -    | Yes              |
| 01  | SA           | Simulated Annealing                          | -    | Yes              |
 
    


