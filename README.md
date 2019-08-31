# Deep Learning Image Embeddings/Features That Work

You are looking for generic image features for
1. Image classification
2. Image retrieval

and so on.

Sometimes, you are not looking for latest and greatest. You just need image
features that just work.

Idea is to create a simple python package to do this:

```python
from image_features import image_features
embedding = image_features(['your_image_1.png', 'your_image_2.jpg'])
```

So that you can use the following:


```python
from sklearn import linear_model
from image_features import image_features
X_train = image_features(['your_image_1.png', 'your_image_2.jpg'])
y_train = ['cat', 'dog']
clf = linear_model.linear_model.LogisticRegressionCV()
clf.fit(X_train, y_train)
```

Package internally uses pretrained deep learning model like resnet50 (default).

## Aim

* Inspired by [face_recognition](https://github.com/ageitgey/face_recognition) and how it just works most of the time.
* Simple yet fairly complete implementation.
* If there is enough interest in this, can put on pypi