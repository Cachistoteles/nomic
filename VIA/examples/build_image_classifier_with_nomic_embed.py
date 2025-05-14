from datasets import load_dataset
from nomic import embed
import numpy as np
from PIL import Image
from typing import Union, List


class ImageClassifierFromTextQuery:
    '''Builds an image classifier that returns True when the image is similar
    to a text query. This is determined by the embedding similarity score
    between the image and the text query. If this similarity score is above
    a certain threshold, the classifier returns True.'''
    def __init__(self, text_query: str, threshold: float):
        self.threshold = threshold

        text_emb = embed.text(
            [text_query],
            task_type="search_query",
            model="nomic-embed-text-v1.5"
        )["embeddings"]
        self.text_emb: np.ndarray = np.array(text_emb)
        
    def predict(self, image: List[Union[str, Image.Image]]) -> List[bool]:
        image_emb = embed.image(image)["embeddings"]
        image_emb = np.array(image_emb)

        similarity = np.dot(self.text_emb, image_emb.T)
        return np.squeeze(similarity > self.threshold).tolist()


print("Building classifier")
classifier = ImageClassifierFromTextQuery(
    text_query="a tiny white ball", threshold=0.058
)

print("Loading dataset")
dataset = load_dataset('frgfm/imagenette', '160px')['train']

# First three should return True, last two should return False
ids = [7450, 6828, 7464, 2343, 3356]
images = [dataset[i]["image"] for i in ids]

print("Predicting")
predictions = classifier.predict(images)
assert predictions == [True, True, True, False, False], (
    f"Predictions should be [True, True, True, False, False], "
    f"got {predictions}"
)
for i, pred in enumerate(predictions):
    print(f"Prediction for image {ids[i]}: {pred}")

sport_dataset = load_dataset('nihaludeen/sports_classification', split="train")
sport_images = sport_dataset["image"]
predictions = classifier.predict(sport_images)
positive_predictions = [i for i, pred in enumerate(predictions) if pred]
sport_images[positive_predictions[0]].show()
