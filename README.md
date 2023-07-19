# SPECTRA
Solar Panel Evaluation through Computer Vision and Advanced Techniques for Reliable Analysis

cd server

Special thanks to Keunhong Park for developing and open-sourcing part of this website code.
https://keunhong.com/

Sebastian-Schuchmann
https://github.com/Sebastian-Schuchmann/ChurrosSamosaClassifier

### Tools used

This was built with PyTorch/FastAI for the Machine Learning part and Flask as a Webserver. For containerization and easy deployment I use Docker. The dataset in use is the [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) but I chose to only use two of the foods contained there.

## How to use

You can easily use this as a template for your own models.

### Train your own Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sebastian-Schuchmann/ChurrorsSamosaClassifier/blob/main/Train_a_Food_Model.ipynb)

Go ahead and open the Notebook in Colab with a simple press of a button! Most steps will be described in detail there. Just follow along!

You can decide which two foods you want to classify by changing:

```python
#Deciding which two foods we want to classify
labelA = 'samosa'
labelB = 'churros'
```

Of course it is also possible to train all the foods contained in the [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) but in order to achieve this you have to modify the code a bit.

At the end of the notebook it will download an export.pkl file, which is your model. 

### Deploy your own Model

Deploying your on model is a easy as replacing the model (server/export.pkl) with your own model. Of course, it makes sense to also modify the HTML/CSS a bit to your liking.

Command to launch the container:
```bash
docker build -t churros_samosa_classifier . && docker run --rm -it -p 5000:5000 churros_samosa_classifier
```