#install keras
pip install keras-ocr

#import liberaries
import matplotlib.pyplot as plt
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of example images
#put the url between the quotation marks
images = [
    keras_ocr.tools.read(url) for url in [
        'https://drive.google.com/file/d/15CxYX0Vi97vCDAeNcDy2jb44tPZv-AyG/view?usp=sharing',
        'https://drive.google.com/file/d/1tdm4MAWkwXLdWvwJfHyZvNt_4jG5zo_E/view?usp=sharing',
        'https://drive.google.com/file/d/1qErogVFms0tmGfGY3icsdn4Ke9vXQLWU/view?usp=sharing'
    ]
]


# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(60, 60))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
