"""
Visualization utilities for persona vectors.

Provides interactive Plotly visualizations for:
- Cosine similarity heatmaps (mean-centered)
- PCA projections colored by assistant axis
- Variance explained plots
"""

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from sklearn.decomposition import PCA
from torch import Tensor

DEFAULT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful"]


def compute_cosine_similarity_matrix_centered(
    persona_vectors: dict[str, Tensor],
) -> tuple[Tensor, list[str]]:
    """
    Compute pairwise cosine similarity between mean-centered persona vectors.

    Args:
        persona_vectors: Dict mapping persona name to activation vector

    Returns:
        Tuple of (similarity matrix, list of persona names)
    """
    names = list(persona_vectors.keys())
    vectors = t.stack([persona_vectors[name].float() for name in names])

    # Mean-center the vectors
    vectors = vectors - vectors.mean(dim=0)

    # Normalize and compute cosine similarity
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)
    cos_sim = vectors_norm @ vectors_norm.T

    return cos_sim, names


def plot_cosine_similarity_heatmap(
    persona_vectors: dict[str, Tensor],
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Plot cosine similarity heatmap of mean-centered persona vectors.

    Args:
        persona_vectors: Dict mapping persona name to activation vector
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    cos_sim, names = compute_cosine_similarity_matrix_centered(persona_vectors)

    fig = px.imshow(
        cos_sim.numpy(),
        x=names,
        y=names,
        title="Persona Cosine Similarity (Mean-Centered)",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="equal",
    )
    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(tickangle=45),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def _compute_pca_data(
    persona_vectors: dict[str, Tensor],
    default_personas: list[str] = DEFAULT_PERSONAS,
) -> dict:
    """
    Compute PCA decomposition and assistant axis for persona vectors.

    Returns dict with: names, pca_coords, projections, pca, cumulative_var
    """
    names = list(persona_vectors.keys())
    vectors = t.stack([persona_vectors[name].float() for name in names])

    # Mean-center
    vectors = vectors - vectors.mean(dim=0)
    vectors_np = vectors.numpy()

    # Compute assistant axis: mean(default) - mean(roles)
    default_indices = [i for i, n in enumerate(names) if n in default_personas]
    role_indices = [i for i, n in enumerate(names) if n not in default_personas]

    if default_indices and role_indices:
        mean_default = vectors[default_indices].mean(dim=0)
        mean_roles = vectors[role_indices].mean(dim=0)
        assistant_axis = mean_default - mean_roles
        assistant_axis = assistant_axis / assistant_axis.norm()
    else:
        assistant_axis = t.zeros(vectors.shape[1])

    # PCA
    pca_full = PCA()
    pca_full.fit(vectors_np)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors_np)

    # Project onto assistant axis
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)
    projections = (vectors_norm @ assistant_axis).numpy()

    return {
        "names": names,
        "pca_coords": pca_coords,
        "projections": projections,
        "pca": pca,
        "cumulative_var": cumulative_var,
    }


def plot_pca_projection(
    persona_vectors: dict[str, Tensor],
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Plot 2D PCA projection of persona vectors colored by assistant axis.

    Args:
        persona_vectors: Dict mapping persona name to activation vector
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    data = _compute_pca_data(persona_vectors)

    fig = px.scatter(
        x=data["pca_coords"][:, 0],
        y=data["pca_coords"][:, 1],
        text=data["names"],
        color=data["projections"],
        color_continuous_scale="RdBu",
        title="Persona Space (PCA) - Colored by Assistant Axis",
        labels={
            "x": f"PC1 ({data['pca'].explained_variance_ratio_[0]:.1%})",
            "y": f"PC2 ({data['pca'].explained_variance_ratio_[1]:.1%})",
            "color": "Assistant Axis",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(width=900, height=700)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_pca_variance_explained(
    persona_vectors: dict[str, Tensor],
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Plot cumulative variance explained by PCA components.

    Args:
        persona_vectors: Dict mapping persona name to activation vector
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    data = _compute_pca_data(persona_vectors)
    cumulative_var = data["cumulative_var"]
    components = np.arange(1, len(cumulative_var) + 1)

    fig = px.line(
        x=components,
        y=cumulative_var,
        title="Cumulative Variance Explained by PCA Components",
        labels={"x": "Number of Components", "y": "Cumulative Variance Explained"},
        markers=True,
    )

    # Add threshold lines
    fig.add_hline(y=0.90, line_dash="dash", line_color="gray",
                  annotation_text="90%", annotation_position="right")
    fig.add_hline(y=0.99, line_dash="dash", line_color="gray",
                  annotation_text="99%", annotation_position="right")

    fig.update_layout(width=800, height=500)

    if save_path:
        fig.write_html(save_path)

    return fig
