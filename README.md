# Fake News Detection Using Deep Learning   

ECE884 Spring Semester 2021 Final Project

## Authors

Namratha Shah: shahnamr@msu.edu    
Sneha Mhaske: mhaskesn@msu.edu
Nathaniel Hawkins: hawki235@msu.edu  

**Corresponding Authors**: Inquiries can be directed at any of the three authors listed above.

## Project Overview

## Dependencies

This project primarily utilizes libraries standard to the Jupyter python installation. The additional requirements needed for this project are listed in `requirements.txt`. To install these dependencies from the command line, run the following

`pip install -r requirements.txt`

## Repository Contents  

* `data/Kaggle` - This directory contains the raw dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). There are two `zip` compressed csv files in this directory, one for _True_ and _Fake_ news respectively. The remainder of the data directory will be filled out following text preprocessing (`notebooks/1_text-preprocessing.ipynb`) and creation of embedding from processed text (`src/2_create_embeddings.py`) for the baseline models.    
* `doc/` - This directory contains deliverables for our final project. This includes a PDF of our final poster and a PDF of the slides from our final presentation. Evaluation forms for our group memebers were individually emailed to the professor.   
* `notebooks/` - The source code for this project are separated into two separate directories. Code that can be interactively run from the Jupyter notebook environment lives in this directory. These notebooks include machine learning model implementation and exploratory data analysis primarily. These notebooks allow an individual to not only reproduce our results, but further explore the underlying code, models, datasets, etc. in further detail.   
* `results/` - Results from the execution of our code.
* `slurm_outs/` - SLURM job script outputs should be directed here to avoid cluttering the main directory with excessive output files.   
* `src/` - Source code for our project that is intended to be run from the command line or via a job script submission (i.e., SLURM). These scripts typically have longer run times and should be reviewed prior to executing on a local machine in order to avoid user issues.

## Reproducing Results

All scripts in our project are numbered to indicate the order in which they should be run to reproduce our results. Note from above: the source code is divided into two directories for this projct.

## Summary of Main Results



## Author Contributions

NH, SM, NS all contributed equally to project inseption, algorithm selection and study outline, and creation of deliverables (i.e., poster, final slideshow, repository). NH, SM, NS all contributed equally to data selection, preprocessing, and exploratory analyses. SM coded the implementation of the transformer models. NS coded the implementation of the LSTM models. NH coded the implementation of the baseline machine learning models. NH, SM, NS all contributed equally to interpretation of results and determining conclusions from said results.