import pytest

from fyp_sim.corpus import generate_video_corpus


def test_determinism():
    """Same seed and config should produce identical results."""
    n = 10
    seed = 42
    cfg = {}

    videos1 = generate_video_corpus(n, seed, cfg)
    videos2 = generate_video_corpus(n, seed, cfg)

    assert len(videos1) == n
    assert videos1 == videos2

    # Verify deep equality of fields
    for v1, v2 in zip(videos1, videos2, strict=True):
        assert v1.video_id == v2.video_id
        assert v1.topic_category == v2.topic_category
        assert v1.duration_s == v2.duration_s
        assert v1.tags == v2.tags


def test_sensitivity():
    """Different seeds should produce different results."""
    n = 10
    cfg = {}

    videos1 = generate_video_corpus(n, 42, cfg)
    videos2 = generate_video_corpus(n, 43, cfg)

    assert videos1 != videos2


def test_config_constraints():
    """Verify generated values respect config bounds."""
    n = 50
    seed = 123
    cfg = {
        "generator": {
            "duration": {"dist": "uniform", "min": 100, "max": 100},
            "sentiment": {"weights": {1.0: 1.0}},
            "viewpoint": {"weights": {0.5: 1.0}},
            "topic": {"weights": {"politics": 1.0}},
            "tags": {"vocab": ["a", "b", "c"], "per_video": {"min": 1, "max": 1}},
        }
    }

    videos = generate_video_corpus(n, seed, cfg)

    for v in videos:
        assert v.duration_s == 100
        assert v.sentiment_score == 1.0
        assert v.viewpoint_score == 0.5
        assert v.topic_category == "politics"
        assert len(v.tags) == 1
        assert v.tags[0] in ["a", "b", "c"]


def test_taxonomy_validation():
    """Should validate topics against taxonomy if provided."""

    class MockTaxonomy:
        TOPIC_CATEGORIES = ("tech", "art")

    n = 1
    seed = 1

    # Valid config
    cfg_valid = {"generator": {"topic": {"weights": {"tech": 1.0}}}}
    videos = generate_video_corpus(n, seed, cfg_valid, taxonomy=MockTaxonomy)
    assert videos[0].topic_category == "tech"

    # Invalid config
    cfg_invalid = {"generator": {"topic": {"weights": {"sports": 1.0}}}}
    with pytest.raises(ValueError, match="not found in taxonomy"):
        generate_video_corpus(n, seed, cfg_invalid, taxonomy=MockTaxonomy)


def test_empty_config_defaults():
    """Should work with empty config and fallback to defaults."""
    n = 10
    seed = 1
    cfg = {}

    videos = generate_video_corpus(n, seed, cfg)
    assert len(videos) == 10
    for v in videos:
        assert isinstance(v.video_id, int)
        assert isinstance(v.duration_s, int)
        # Default fallback for topic is likely "undefined" if no taxonomy found/provided
        # But if we don't mock taxonomy, and it is imported inside generator, it might use real taxonomy or None.
        # In generator.py: `tx = taxonomy if taxonomy else default_taxonomy`
        # `try: from fyp_sim import taxonomy as default_taxonomy`
        # So it uses real taxonomy by default.
        # Real taxonomy has "politics", etc.
        # So topic should be one of them.
        pass


def test_n_zero_negative():
    """Handle edge cases for n."""
    assert generate_video_corpus(0, 1, {}) == []

    with pytest.raises(ValueError):
        generate_video_corpus(-1, 1, {})
