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

All scripts in our project are numbered to indicate the order in which they should be run to reproduce our results. Note from above: the source code is divided into two directories for this project. The following is the intended order that our source code can be run to fully reproduce the results of our project.

1. `notebooks/0_data-viewer.ipynb` - This will load in our dataset from Kaggle to pandas dataframe objects. From here, a user can explore the dataset, examine individual articles, etc. This script can be run at any point that the user would like to further explore the data used for our project.
2. `notebooks/1_text-preprocessing.ipynb` - Loads in the dataset, implements a standard text preprocessing pipeline common in the literature, and outputs a dataframe of processed article titles, article texts, and the associated label (binary label for _True_ or _Fake_).
3. `src/2_create_embeddings.py` - Using the preprocessed text from the previous step, several different embedding matrices are created using the flairNLP library and output as numpy array objects to the `data/` directory. Embeddings are made for the title alone, body of the article alone, the average of these two features, and the concatenation of these two features. 
4. **Baseline Models**: The baseline models are broken down into four sepearate scripts in order to take advantage of job-level parallelism to faster compute the results of > 300 baseline models. All baseline models are implemented in the `src/` directory and are named with a proceeding **3** (e.g., `src/3_avg-baselines.py`). Each baseline script has an associated SLURM job script that can be submitted to a compatible compute cluster. The following job scripts will implement all baseline models and contain the command line code for executing on a local machine if desired.
    1. `src/3_avg-baselines.sb` - Arithmetic average of the embeddings for the article title and article body.
    2. `src/3_body-baselines.sb` - Using the embedding for the article body alone.
    3. `src/3_concat-baselines.sb` - Concatenation of the embeddings for the article title and article body.
    4. `src/3_title-baselines.sb` - Using the embedding for the article title alone.
5. `notebooks/4_compile-baselines.ipynb` - Compile the results files for the above baseline models into csv files for convenient post-processing and analysis.
6. `notebooks/5_lstms.ipynb` - Our implementation of both LSTM and Bi-LSTM.
7. **Transformer Models**: We implemented two separate transformer models. Each has its own code for execution:
    1. `notebooks/6_BERT-Classifier.ipynb` - Implementation of our fine tuned BERT model. Associated `.py` file: `src/6_FakeNews_BERT_16.py`.
    2. `notebooks/6_GPT2-Classification.ipynb` - Implementation of our GPT2 model. Associated `.py` file: `src/6_FakeNews_GPT2.py`.


The execution of the above code in the order outlined will fully reproduce the results of our work.


## Summary of Main Results



## Author Contributions

NH, SM, NS all contributed equally to project inseption, algorithm selection and study outline, and creation of deliverables (i.e., poster, final slideshow, repository). NH, SM, NS all contributed equally to data selection, preprocessing, and exploratory analyses. SM coded the implementation of the transformer models. NS coded the implementation of the LSTM models. NH coded the implementation of the baseline machine learning models. NH, SM, NS all contributed equally to interpretation of results and determining conclusions from said results.