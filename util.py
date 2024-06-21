from fastai.vision.all import *


def predict(img, learner, labels):
    img = PILImage.create(img)
    pred, _, probs = learner.predict(img)
    return pred, {labels[i]: float(probs[i] for i in range(len(labels)))}