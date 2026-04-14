import json
import numpy as np

with open("project\contrastive_method\panose_vectors.json", "r", encoding="utf-8") as f:
    panose_json = json.load(f)

FONT_NAMES = list(panose_json.keys())
FONT_VECTORS = np.array(list(panose_json.values()))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) )


def find_font(pred, top=10):

    sims = [cosine_similarity(pred, font_vec) for font_vec in FONT_VECTORS ]
    sims = np.array(sims)
    inds = np.argsort(sims)[::-1]

    results = [
        {
            "name": FONT_NAMES[i],
            "sim": float(sims[i])
        }
        for i in inds 
    ]
    # print("Косинусное")
    # print(results)
    return results[:10]
