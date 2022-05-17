# Final assignment (selfassigned) - 
 
Link to github of this assignment: https://github.com/sarah-hvid/Final_assignment

## Assignment description
The purpose of this assignment is to investigate whether it is possible to classify images of Pokemon by generation. The hypothesis is that there is stilistic changes in the design of Pokemon by generation that a model can distinguish. 

## Methods
The main problem of classifying the Pokemon, is the small size of the dataset. There is a limited number of Pokemon available from each generation. This also makes it difficult to train a brand new model on the data. For this reason, transfer learning is used. The ```VGG16``` model is loaded without the top layer. A new classifier layer is then trained on the Pokemon images. Data augmentation was also included in the model to mimic more data. The user may specify whether to use data augmentation, which kernel regularizer to use, the number of epochs and the batchsize. The results of the model run is written to the ```output``` folder. A plot is saved of the loss and accuracy history of the model. A text file of the classification report along with the confusion matrix is also saved.

## Usage
In order to run the scripts, certain modules need to be installed. These can be found in the ```requirements.txt``` file. The folder structure must be the same as in this GitHub repository (ideally, clone the repository).
The data used in the assignment is the ```Pokemon Image Dataset``` from Kaggle. The data must be downloaded from this website: https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types. The images must be unzipped from within the ```data``` folder in order to replicate the results. This should result in the following structure: ```data``` folder, ```images``` folder, ```images``` folder, all images. The CSV file should be placed in the data folder. The information of the generation of the Pokemon is not included in the Kaggle CSV file. This data is therefore scraped from Wikipedia: https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon. The current working directory when running the script must be the one that contains the ```data```, ```output``` and ```src``` folder. It is important that the user runs the ```gather_data.py``` script first.
Examples of how to run the scripts from the command line: 

Gathering the data:
- python src/gather_data.py
    
Classifying the data:
- python src/model.py -d_a 1

With all specifications:
- python src/model.py -d_a 0 -ke_re l1_l2 -epochs 15 -batch_size 10
  
Examples of the outputs of the scripts can be found in the ```output``` folder. 

## Results
The results of the classification of the Pokemon images are quite consistent across all specifications. The f1-score ranges between 0.30 and 0.40. Some specifications resulted in higher f1-scores, but only for that particular split of the data. Therefore, the results are better than chance, but not great despite data augmentation and kernel regularization. Looking at the confusion matrix, some results appeared to be consistent across specifications and data splits; generation 7 is easy to classify, generation 2 and 6 is hard and other generations are often predicted as being generation 1 and 5. Overall, the classifier is not great. Despite the attempts to improve the model, the results remain the same. This suggests that the images overall are very difficult to classify by generation. 
 