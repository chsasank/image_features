# Deep Learning Image Embeddings/Features That Work

You are looking for generic image features for
1. Image classification
2. Image retrieval

and so on.

Sometimes, you are not looking for latest and greatest. You just need image
features that just work.

Idea is to create a simple python package to do this:

```python
from image_embedding import image_embedding
embedding = image_embedding('your_image.png')
```


## Aim

* Inspired by [face_recognition](https://github.com/ageitgey/face_recognition) and how it just works most of the time.
* Minimal dependencies (no torch/tf etc.)