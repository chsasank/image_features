# Deep Learning Image Embeddings/Features That Work

You are looking for generic image features for
1. Image classification
2. Image retrieval
3. Image similarity and so on.

Sometimes, you are not looking for latest and greatest. You just need something that just works. With image_features, you can extract such deep learning based features from images in a single line of code:

```python
from image_features import image_features
features = image_features(['your_image_1.png', 'your_image_2.jpg'])
```

You can use these features to train a scikit-learn classification model:

```python
from sklearn import linear_model
from image_features import image_features
X_train = image_features(['your_image_1.png', 'your_image_2.jpg'])
y_train = ['cat', 'dog']
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
```

Package internally uses PyTorch and imagenet pretrained deep learning model like [resnet50](https://arxiv.org/abs/1512.03385) (default).

## Install

```
pip install -U git+https://github.com/chsasank/image_features.git
```

## Tutorial

I have written an [accompanying tutorial](https://chsasank.github.io/deep-learning-image-features.html) to help you get started. 

![](https://storage.googleapis.com/public-sasank/image_features_tutorial.png)

## Aim

* Inspired by [face_recognition](https://github.com/ageitgey/face_recognition) and how it just works most of the time.
* Simple yet fairly complete implementation.
* If there is enough interest in this, put on pypi