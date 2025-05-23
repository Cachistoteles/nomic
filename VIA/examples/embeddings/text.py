import nomic
import numpy as np
import time

start = time.time()
output = nomic.embed.text(
    texts=['Nomic Embedding API'],
    model='nomic-embed-text-v1'
)

print(time.time() - start)
print(output['usage'])

embeddings = np.array(output['embeddings'])

print(embeddings.shape)
