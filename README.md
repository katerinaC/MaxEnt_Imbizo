### Imbizo Project: Maximum Entropy Model Project for fMRI Data 

This repository represents part of the code used for my Imbizo.africa final 
project.

Part of the project was run in Mathematica software, however, might be included 
into python scripts in the future. 

#### Running the script
To run a main script, open your command line, navigate to the directory and run 
it according to the arguments.

##### Running example:
`python main_max_ent.py --input /Users/UserName/Folder/fmri_data
 --output /Users/UserName/Folder/fmri_data_output --threshold 0.`
 
#### Notes
As part of the model is now incorporated in Mathematica, user has to provide
corresponding lambdas (Lagrange multipliers). 

The script works for 6 regions. If different number is necessary - change `repeat`
in func `calculate_prob_max_ent` in `max_ent.py`