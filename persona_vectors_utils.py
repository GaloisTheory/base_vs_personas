"""
Analysis utilities for persona vectors.

Higher-level analysis that builds on the raw vectors from persona_vectors.py.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA


def compute_axes(
    vectors_dict: dict[str, torch.Tensor],
    assistant_personas: list[str],
) -> dict:
    """
    Compute assistant axis, PC1, and per-persona projections.

    The assistant axis is defined as the normalized difference between the mean
    of assistant-like persona vectors and the mean of all other (role) vectors,
    after mean-centering.

    Args:
        vectors_dict: mapping persona name -> activation vector (d_model,)
        assistant_personas: names to treat as the "assistant" cluster

    Returns dict with:
        names:          list of persona names (same order as projections)
        assistant_proj: (n_personas,) projection onto assistant axis
        pc1_proj:       (n_personas,) projection onto PC1
        cosine_sim:     |cos(assistant_axis, PC1)|
        r_squared:      squared Pearson r between the two projection vectors
        var_explained:  fraction of total variance captured by PC1
    """
    names = list(vectors_dict.keys())
    vectors = torch.stack([vectors_dict[n].float() for n in names])

    # Mean-center
    vectors = vectors - vectors.mean(dim=0)

    # Split into assistant vs role indices
    assistant_idx = [i for i, n in enumerate(names) if n in assistant_personas]
    role_idx = [i for i, n in enumerate(names) if n not in assistant_personas]

    # Assistant axis
    mean_assistant = vectors[assistant_idx].mean(dim=0)
    mean_roles = vectors[role_idx].mean(dim=0)
    assistant_axis = mean_assistant - mean_roles
    assistant_axis = assistant_axis / assistant_axis.norm()

    # PCA
    vectors_np = vectors.numpy()
    pca = PCA()
    pca.fit(vectors_np)
    pc1 = torch.from_numpy(pca.components_[0]).float()
    pc1 = pc1 / pc1.norm()

    # Per-persona projections
    assistant_proj = (vectors @ assistant_axis).numpy()
    pc1_proj = (vectors @ pc1).numpy()

    # Cosine similarity between the two axes
    cosine_sim = float(torch.dot(assistant_axis, pc1).abs())

    # R² between the two projection vectors
    correlation = np.corrcoef(assistant_proj, pc1_proj)[0, 1]
    r_squared = correlation ** 2

    return {
        "names": names,
        "assistant_proj": assistant_proj,
        "pc1_proj": pc1_proj,
        "cosine_sim": cosine_sim,
        "r_squared": r_squared,
        "var_explained": pca.explained_variance_ratio_[0],
    }
