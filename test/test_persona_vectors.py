"""
Tests for Persona Vector Calculator

Primary acceptance criteria: computed vectors must match cached expected vectors
with cosine similarity > 0.99 for all personas.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch as t
import torch.nn.functional as F

from utils.persona_vectors import (
    PersonaVectorConfig,
    ActivationExtractor,
    extract_persona_vectors,
    load_responses,
    load_persona_vectors,
    PERSONAS,
    EVAL_QUESTIONS,
)


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test_fixtures"
RESPONSES_CACHE_FILE = TEST_DATA_DIR / "responses_cache.json"
EXPECTED_VECTORS_FILE = TEST_DATA_DIR / "persona_vectors_layer40.pt"


def cosine_similarity(v1: t.Tensor, v2: t.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


class TestPersonaVectorConfig:
    """Tests for configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PersonaVectorConfig()
        assert config.model_name == "google/gemma-3-27b-it"
        assert config.layer_fraction == 0.65
        assert config.max_tokens == 256
        assert config.temperature == 0.7

    def test_layer_computation(self):
        """Test layer computation from fraction."""
        config = PersonaVectorConfig(layer_fraction=0.65)
        # For a 62-layer model, 62 * 0.65 + 0.5 = 40.8 -> 40
        assert config.get_extraction_layer(62) == 40

    def test_openrouter_model_override(self):
        """Test API model can be overridden."""
        config = PersonaVectorConfig(
            model_name="google/gemma-3-27b-it",
            openrouter_model="different/model"
        )
        assert config.openrouter_model == "different/model"


class TestResponseLoading:
    """Tests for response cache loading."""

    def test_load_responses_format(self):
        """Test that cached responses load correctly."""
        if not RESPONSES_CACHE_FILE.exists():
            pytest.skip("Response cache file not found")

        responses = load_responses(RESPONSES_CACHE_FILE)

        # Check format
        assert isinstance(responses, dict)
        assert len(responses) > 0

        # Check key format is (persona, question) tuple
        key = list(responses.keys())[0]
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)

    def test_responses_cover_all_personas(self):
        """Test that responses exist for all expected personas."""
        if not RESPONSES_CACHE_FILE.exists():
            pytest.skip("Response cache file not found")

        responses = load_responses(RESPONSES_CACHE_FILE)
        personas_in_responses = {k[0] for k in responses.keys()}

        expected_personas = set(PERSONAS.keys())
        assert personas_in_responses == expected_personas, (
            f"Missing personas: {expected_personas - personas_in_responses}"
        )


class TestExpectedVectors:
    """Tests for expected vector file."""

    def test_load_expected_vectors(self):
        """Test that expected vectors load correctly."""
        if not EXPECTED_VECTORS_FILE.exists():
            pytest.skip("Expected vectors file not found")

        vectors = load_persona_vectors(EXPECTED_VECTORS_FILE)

        assert isinstance(vectors, dict)
        assert len(vectors) > 0

        # Check shape
        sample_vec = list(vectors.values())[0]
        assert sample_vec.shape == (5376,), f"Expected shape (5376,), got {sample_vec.shape}"


@pytest.fixture(scope="module")
def extractor():
    """Create an activation extractor for testing."""
    config = PersonaVectorConfig()
    return ActivationExtractor(config)


@pytest.fixture(scope="module")
def cached_responses():
    """Load cached responses."""
    return load_responses(RESPONSES_CACHE_FILE)


@pytest.fixture(scope="module")
def expected_vectors():
    """Load expected vectors."""
    return load_persona_vectors(EXPECTED_VECTORS_FILE)


class TestPersonaVectorExtraction:
    """
    Main acceptance test: extracted vectors must match expected vectors.
    """

    def test_persona_vectors_match_expected(
        self,
        extractor: ActivationExtractor,
        cached_responses: dict,
        expected_vectors: dict,
    ):
        """
        PRIMARY TEST: Verify computed vectors match expected vectors.

        This is the main acceptance test. All persona vectors must have
        cosine similarity > 0.99 with the expected vectors.
        """
        # Extract vectors at layer 40 (matching expected)
        layer = 40

        computed_vectors = extract_persona_vectors(
            extractor=extractor,
            personas=PERSONAS,
            questions=EVAL_QUESTIONS,
            responses=cached_responses,
            layer=layer,
        )

        # Check all expected personas were computed
        missing = set(expected_vectors.keys()) - set(computed_vectors.keys())
        assert not missing, f"Missing personas: {missing}"

        # Check cosine similarity for each persona
        failures = []
        for persona_name, expected_vec in expected_vectors.items():
            computed_vec = computed_vectors[persona_name]

            # Convert to float32 for comparison
            expected_vec = expected_vec.float()
            computed_vec = computed_vec.float()

            sim = cosine_similarity(computed_vec, expected_vec)

            if sim < 0.99:
                failures.append((persona_name, sim))

        if failures:
            failure_msg = "\n".join(
                f"  {name}: cosine_sim = {sim:.6f}" for name, sim in failures
            )
            pytest.fail(
                f"Persona vectors do not match expected (threshold: 0.99):\n{failure_msg}"
            )

    def test_vector_shapes(
        self,
        extractor: ActivationExtractor,
        cached_responses: dict,
    ):
        """Test that extracted vectors have correct shape."""
        layer = 40

        # Extract for just one persona to save time
        test_persona = {"default": PERSONAS["default"]}

        computed_vectors = extract_persona_vectors(
            extractor=extractor,
            personas=test_persona,
            questions=EVAL_QUESTIONS,
            responses=cached_responses,
            layer=layer,
        )

        vec = computed_vectors["default"]
        assert vec.shape == (5376,), f"Expected shape (5376,), got {vec.shape}"

    def test_vector_norms_reasonable(
        self,
        extractor: ActivationExtractor,
        cached_responses: dict,
    ):
        """Test that vector norms are in reasonable range."""
        layer = 40

        test_persona = {"default": PERSONAS["default"]}

        computed_vectors = extract_persona_vectors(
            extractor=extractor,
            personas=test_persona,
            questions=EVAL_QUESTIONS,
            responses=cached_responses,
            layer=layer,
        )

        vec = computed_vectors["default"]
        norm = vec.norm().item()

        # Norms should be reasonable (not zero, within expected range)
        # Expected norms for bfloat16 activations from Gemma 27B are ~60000
        assert 1000 < norm < 100000, f"Vector norm {norm} seems unreasonable"


class TestCrossValidation:
    """
    Cross-validate freshly-generated vectors (from data/) against reference vectors (from for_tests/).

    Uses a looser threshold (0.90) than the deterministic extraction test because:
    - API responses are stochastic (temperature=0.7)
    - data/ vectors come from a different set of API responses than the reference
    - Averaging over 18 questions stabilizes the mean, but not perfectly
    """

    GENERATED_VECTORS_FILE = (
        Path(__file__).parent.parent / "data" / "persona_vectors" / "example_test_gemma-3-27b-it_layer40.pt"
    )
    CROSS_VAL_THRESHOLD = 0.90

    @pytest.fixture(scope="class")
    def generated_vectors(self):
        if not self.GENERATED_VECTORS_FILE.exists():
            pytest.skip(f"Generated vectors not found at {self.GENERATED_VECTORS_FILE} — run example.py first")
        return load_persona_vectors(self.GENERATED_VECTORS_FILE)

    @pytest.fixture(scope="class")
    def reference_vectors(self):
        if not EXPECTED_VECTORS_FILE.exists():
            pytest.skip("Reference vectors not found")
        return load_persona_vectors(EXPECTED_VECTORS_FILE)

    def test_cross_validation_cosine_similarity(
        self,
        generated_vectors: dict,
        reference_vectors: dict,
    ):
        """
        Verify that freshly-generated persona vectors are close to the reference vectors.

        Each persona's generated vector (from new API responses) should have
        cosine similarity > 0.90 with the corresponding reference vector.
        """
        # Check all reference personas exist in generated set
        missing = set(reference_vectors.keys()) - set(generated_vectors.keys())
        assert not missing, f"Missing personas in generated vectors: {missing}"

        failures = []
        similarities = {}
        for persona_name, ref_vec in reference_vectors.items():
            gen_vec = generated_vectors[persona_name]

            ref_vec = ref_vec.float()
            gen_vec = gen_vec.float()

            sim = cosine_similarity(gen_vec, ref_vec)
            similarities[persona_name] = sim

            if sim < self.CROSS_VAL_THRESHOLD:
                failures.append((persona_name, sim))

        # Print all similarities for diagnostic purposes
        print("\nCross-validation cosine similarities:")
        for name, sim in sorted(similarities.items(), key=lambda x: x[1]):
            marker = " *** FAIL ***" if sim < self.CROSS_VAL_THRESHOLD else ""
            print(f"  {name:20s}: {sim:.4f}{marker}")

        if failures:
            failure_msg = "\n".join(
                f"  {name}: cosine_sim = {sim:.6f}" for name, sim in failures
            )
            pytest.fail(
                f"Cross-validation failed (threshold: {self.CROSS_VAL_THRESHOLD}):\n{failure_msg}"
            )

    def test_generated_vectors_complete(self, generated_vectors: dict):
        """Verify generated vectors contain all personas."""
        expected_personas = set(PERSONAS.keys())
        actual_personas = set(generated_vectors.keys())
        missing = expected_personas - actual_personas
        assert not missing, f"Missing personas: {missing}"
        assert len(actual_personas) == len(PERSONAS), (
            f"Expected {len(PERSONAS)} personas, got {len(actual_personas)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
