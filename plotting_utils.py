"""
Visualization utilities for persona vectors.

Provides interactive Plotly visualizations for:
- Cosine similarity heatmaps (mean-centered)
- PCA projections colored by assistant axis
- Variance explained plots
- Cross-model comparison (PCA overlay, cross-similarity)
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


def plot_pca_comparison(
    vectors_a: dict[str, Tensor],
    vectors_b: dict[str, Tensor],
    label_a: str = "Model A",
    label_b: str = "Model B",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Overlay two sets of persona vectors on one PCA scatter plot.

    Fits PCA jointly on both sets (mean-centered together), then plots with
    different marker shapes per model and labels per persona.

    Args:
        vectors_a: Dict mapping persona name to vector (model A)
        vectors_b: Dict mapping persona name to vector (model B)
        label_a: Display name for model A
        label_b: Display name for model B
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    # Use shared persona names (intersection)
    shared_names = [n for n in vectors_a if n in vectors_b]

    vecs_a = t.stack([vectors_a[n].float() for n in shared_names])
    vecs_b = t.stack([vectors_b[n].float() for n in shared_names])
    combined = t.cat([vecs_a, vecs_b], dim=0)

    # Mean-center jointly
    combined = combined - combined.mean(dim=0)
    combined_np = combined.numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined_np)

    n = len(shared_names)
    coords_a, coords_b = coords[:n], coords[n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords_a[:, 0], y=coords_a[:, 1],
        mode="markers+text", text=shared_names, textposition="top center",
        name=label_a,
        marker=dict(size=12, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=coords_b[:, 0], y=coords_b[:, 1],
        mode="markers+text", text=shared_names, textposition="bottom center",
        name=label_b,
        marker=dict(size=12, symbol="diamond"),
    ))

    # Draw lines connecting same persona across models
    for i in range(n):
        fig.add_trace(go.Scatter(
            x=[coords_a[i, 0], coords_b[i, 0]],
            y=[coords_a[i, 1], coords_b[i, 1]],
            mode="lines", line=dict(color="gray", width=0.5, dash="dot"),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Persona Space Comparison: {label_a} vs {label_b}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        width=1000, height=700,
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_cosine_cross_similarity(
    vectors_a: dict[str, Tensor],
    vectors_b: dict[str, Tensor],
    label_a: str = "Model A",
    label_b: str = "Model B",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Plot heatmap of cosine similarity between model A and model B persona vectors.

    Each cell (i, j) shows cosine similarity between model A's persona i
    and model B's persona j. Vectors are mean-centered jointly before comparison.

    Args:
        vectors_a: Dict mapping persona name to vector (model A)
        vectors_b: Dict mapping persona name to vector (model B)
        label_a: Display name for model A
        label_b: Display name for model B
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    shared_names = [n for n in vectors_a if n in vectors_b]

    vecs_a = t.stack([vectors_a[n].float() for n in shared_names])
    vecs_b = t.stack([vectors_b[n].float() for n in shared_names])

    # Mean-center jointly
    combined_mean = t.cat([vecs_a, vecs_b], dim=0).mean(dim=0)
    vecs_a = vecs_a - combined_mean
    vecs_b = vecs_b - combined_mean

    # Normalize
    vecs_a_norm = vecs_a / vecs_a.norm(dim=1, keepdim=True)
    vecs_b_norm = vecs_b / vecs_b.norm(dim=1, keepdim=True)

    # Cross-similarity: (n_personas_a, n_personas_b)
    cross_sim = (vecs_a_norm @ vecs_b_norm.T).numpy()

    fig = px.imshow(
        cross_sim,
        x=[f"{n} ({label_b})" for n in shared_names],
        y=[f"{n} ({label_a})" for n in shared_names],
        title=f"Cross-Model Cosine Similarity: {label_a} vs {label_b}",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="equal",
    )
    fig.update_layout(
        width=900,
        height=900,
        xaxis=dict(tickangle=45),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


MARKER_SYMBOLS = ["circle", "diamond", "square", "cross", "x", "triangle-up"]


def plot_pca_comparison_nway(
    all_vectors: list[dict[str, Tensor]],
    labels: list[str],
    save_path: Optional[Path] = None,
) -> go.Figure:
    """
    Overlay N sets of persona vectors on one joint PCA scatter plot.

    Fits PCA jointly on all sets (mean-centered together), plots with
    distinct marker shapes per model and connecting lines per persona.

    Args:
        all_vectors: List of dicts mapping persona name to vector
        labels: Display name for each model
        save_path: Optional path to save HTML file

    Returns:
        Plotly Figure object
    """
    # Use shared persona names (intersection of all sets)
    shared_names = list(all_vectors[0].keys())
    for vecs in all_vectors[1:]:
        shared_names = [n for n in shared_names if n in vecs]

    # Stack all vectors: list of (n_personas, d_model)
    per_model = [t.stack([v[n].float() for n in shared_names]) for v in all_vectors]
    combined = t.cat(per_model, dim=0)

    # Mean-center jointly
    combined = combined - combined.mean(dim=0)
    combined_np = combined.numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined_np)

    n = len(shared_names)
    coords_per_model = [coords[i * n : (i + 1) * n] for i in range(len(all_vectors))]

    fig = go.Figure()
    for idx, (label, model_coords) in enumerate(zip(labels, coords_per_model)):
        fig.add_trace(go.Scatter(
            x=model_coords[:, 0], y=model_coords[:, 1],
            mode="markers+text", text=shared_names, textposition="top center",
            name=label,
            marker=dict(size=12, symbol=MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]),
        ))

    # Draw lines connecting same persona across models
    for i in range(n):
        xs = [coords_per_model[m][i, 0] for m in range(len(all_vectors))]
        ys = [coords_per_model[m][i, 1] for m in range(len(all_vectors))]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines", line=dict(color="gray", width=0.5, dash="dot"),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Persona Space Comparison: {' / '.join(labels)}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        width=1100, height=750,
    )

    if save_path:
        fig.write_html(save_path)

    return fig
