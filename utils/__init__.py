from utils.persona_vectors import (
    PersonaVectorConfig,
    ActivationExtractor,
    generate_responses,
    extract_persona_vectors,
    save_responses,
    load_responses,
    save_persona_vectors,
    load_persona_vectors,
    model_slug,
    PERSONAS,
    EVAL_QUESTIONS,
    DEFAULT_PERSONAS,
)
from utils.persona_vectors_utils import compute_axes
from utils.plotting_utils import (
    compute_cosine_similarity_matrix_centered,
    plot_cosine_similarity_heatmap,
    plot_pca_projection,
    plot_pca_variance_explained,
    plot_pca_comparison,
    plot_cosine_cross_similarity,
    plot_pca_comparison_nway,
)
from utils.transcript_projection import (
    discover_transcripts,
    format_conversation_raw,
    find_assistant_spans,
    project_transcript,
)
