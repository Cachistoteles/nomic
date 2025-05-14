import numpy as np
import logging
import json
from typing import List, Tuple, Union
from tqdm import tqdm
from PIL import Image
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_sagemaker_response(response):
    """
    Parse response from a local embedding server and return the embedding
    as a numpy array.

    Args:
        response: The response from the local server.

    Returns:
        np.float16 array of embeddings.
    """
    resp = json.loads(response)
    return np.array(resp["embeddings"], dtype=np.float16)


def embed_text(
    texts: List[str],
    batch_size: int = 32,
    dimensionality: int = 768,
    binary: bool = False,
):
    """
    Embed a list of texts using a local embedding model.

    Args:
        texts: List of texts to be embedded.
        batch_size: Size of each batch. Default is 32.
        dimensionality: Number of dimensions to return. Options are
            (64, 128, 256, 512, 768).
        binary: Whether to return binary embeddings.

    Returns:
        Dictionary with "embeddings" (python 2d list of floats).
    """
    if len(texts) == 0:
        logger.warning("No texts to embed.")
        return None

    assert dimensionality in (64, 128, 256, 512, 768), (
        f"Invalid number of dimensions: {dimensionality}"
    )

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = {
            "texts": texts[i: i + batch_size],
            "binary": binary,
            "dimensionality": dimensionality,
        }
        # Simulate local embedding process
        response = json.dumps({
            "embeddings": [[0.0] * dimensionality for _ in batch["texts"]]
        })
        embeddings.extend(parse_sagemaker_response(response))

    return {
        "embeddings": embeddings,
        "model": "local-embed-text-v1.0",
        "usage": {},
    }


def preprocess_image(
    images: List[Union[str, Image.Image, bytes]]
) -> List[Tuple[str, bytes]]:
    """
    Preprocess a list of images for embedding using a local model.

    Args:
        images: List of images to be embedded.

    Returns:
        List of tuples containing image name and bytes.
    """
    encoded_images = []
    for idx, image in enumerate(images):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append((f"image_{idx}.jpg", buffered.getvalue()))

    return encoded_images


def embed_image(
    images: List[Union[str, Image.Image, bytes]],
    batch_size: int = 16,
    dimensionality: int = 512,
) -> dict:
    """
    Embed a list of images using a local embedding model.

    Args:
        images: List of images to be embedded.
        batch_size: Size of each batch. Default is 16.
        dimensionality: Number of dimensions to return. Default is 512.

    Returns:
        Dictionary with "embeddings" (python 2d list of floats).
    """
    embeddings = []
    preprocessed_images = preprocess_image(images)

    pbar = tqdm(total=len(preprocessed_images))
    for i in range(0, len(preprocessed_images), batch_size):
        batch = preprocessed_images[i: i + batch_size]
        # Simulate local embedding process
        response = json.dumps({
            "embeddings": [[0.0] * dimensionality for _ in batch]
        })
        embeddings.extend(parse_sagemaker_response(response))
        pbar.update(len(batch))

    return {
        "embeddings": embeddings,
        "model": "local-embed-image-v1.0",
        "usage": {},
    }
