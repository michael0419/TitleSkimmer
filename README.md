# Title Skimmer
by Michael Cai

A machine learning project developed for the Cuny Tech Prep program

## Goal of Title Skimmer:  
The main objective for this project is to develop a tool that creators can use to improve the quality of their titles. Users may also use this tool to see how a machine learning model can potentially classify their content when it gets published online.  

So far Title Skimmer meets this need by providing a classification of based on the title, this way a user can make sure their title can accurately reflect the genre or type of content. 

In addition, Title Skimmer can help inspire new titles/headlines by searching similar titles based on meanings from its dataframe.  

## Important Info:
This repository holds the code required to run the full Title Skimmer flask application**  
**due to Github limitations, some files were too large so 2 folders need to be downloaded  
Required folders "models/" and "data/" are available [here](https://www.dropbox.com/sh/blq7sri150vxrcv/AACgJeSzvIos7VIOBIzY_z9za?dl=0)  

Ram usage: ~4GB  
Total Disk space usage: ~2GB  

## There is a lightweight online version!  
It ONLY supports the Multinomial Naive Bayes Model  
[GitHub Link](https://github.com/michael0419/TitleSkimmerLite)  
[Try It Out On Heroku!](https://title-skimmer-lite.herokuapp.com/)  

## About the models usage:

Title Skimmer showcases the use of 3 machine learning models (Universal Sentence Encoder, Multinomial Naive Bayes, and Deep Learning) to classify titles  
Title Skimmer also uses the Universal Sentence Encoder in conjunction with a encoded version of the news category dataset to suggest similar titles  

## My prep work for the project:
[Link to Multinomial Naive Bayes model and data exploratory on GitHub](https://github.com/michael0419/TitleSkimmerDataAnalytics)  
[Link to Neural Networking model development and similar title function on Google Colab](https://colab.research.google.com/drive/1FaBL1lfuHU6BNvgmFagze9G5HnCFiEND?usp=sharing)

## Data Source:
Models trained uses the following datasets:
- [News Category Dataset from the Huffington Post on Kaggle](https://www.kaggle.com/rmisra/news-category-dataset)

## Google Slides Presentation
[Presentation Link](https://docs.google.com/presentation/d/1-LspBr_Ed0GoAbrr4bAJACHjYOCxewUnFdwcbLJ6fSk/edit?usp=sharing)

## TODO: clean up code...
