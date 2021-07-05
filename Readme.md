# AlgoDistinctSolutions

The scope of this repository is to offer a tool in which you can for a given competitive programming problem to split it in K different solutions in terms of algorthmic approach as well as implementation.

It uses AlgoLabel repository (https://github.com/NLCP/AlgoLabel) for it's implementation of various preprocessing steps as well as the integration of some state of the art deep learning models.

## Manual annotated dataset
To validate the method, a dataset which contains 10 problems from infoarena was manually annotated. The raw data as well as the embeddings generated for this problem are available here:

    Raw Dataset(https://drive.google.com/file/d/1mSRGL57389is7r5U70b_5yZ6whTeuWz2/view?usp=sharing)

    Embeddings(https://drive.google.com/file/d/11AUF2HklrWND4_eF18VpMHvYVJTCTyOv/view?usp=sharing)

Besides this, there is a script available which downloads those 2 files. The script can be run through an argument available in the main.py.

## Prerequirements

1. You need to download g++
2. You need to download cppchecker
3. In the algolabel folder you need to download the safe model. Go into AlgoLabel/safe and run python3 -m downloader.py -b .
4. You need to install the libraries from requirements.txt. 

## How to run it

The main.py offers an argument parser which can fulfill multiple operations:
1. Download the embeddings and the raw dataset
2. Transform the dataset into a format which can be used by AlgoLabel
3. Compute the embeddings(tf-idf, w2v, safe) of the preprocessed dataset.
4. Compute the evalution based on the embeddings. The results from the evaluation step are saved in the Data/Validation in csv format.

## Note
The entire repository was tested just on WSL2 using the Ubuntu distro. 





