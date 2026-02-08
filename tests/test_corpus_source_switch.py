import pytest

from fyp_sim.corpus import build_corpus
from fyp_sim.models import Video


def test_source_file_default():
    """Default source='file' loads from video_pool list."""
    cfg = {
        "video_pool": [
            {
                "video_id": 1,
                "topic_category": "t",
                "viewpoint_score": 0.5,
                "sentiment_score": 0.0,
                "duration_s": 10,
            }
        ]
    }
    corpus = build_corpus(cfg)
    assert len(corpus) == 1
    assert corpus[0].video_id == 1


def test_source_file_explicit():
    """Explicit source='file' loads from video_pool list."""
    cfg = {
        "corpus": {"source": "file"},
        "video_pool": [
            {
                "video_id": 2,
                "topic_category": "t",
                "viewpoint_score": 0.5,
                "sentiment_score": 0.0,
                "duration_s": 10,
            }
        ],
    }
    corpus = build_corpus(cfg)
    assert len(corpus) == 1
    assert corpus[0].video_id == 2


def test_source_generated():
    """source='generated' calls generator."""
    n = 5
    cfg = {
        "corpus": {
            "source": "generated",
            "n_videos": n,
            "seed": 123,
            "generator": {"duration": {"min": 10, "max": 10}},
        }
    }
    corpus = build_corpus(cfg)
    assert len(corpus) == n
    assert all(isinstance(v, Video) for v in corpus)
    # Check stable generation
    corpus2 = build_corpus(cfg)
    assert corpus == corpus2


def test_source_generated_missing_n():
    """Must raise error if n_videos missing."""
    cfg = {"corpus": {"source": "generated"}}
    with pytest.raises(ValueError, match="n_videos"):
        build_corpus(cfg)


def test_source_unknown():
    """Unknown source raises ValueError."""
    cfg = {"corpus": {"source": "mystery_source"}}
    with pytest.raises(ValueError, match="Unknown corpus source"):
        build_corpus(cfg)
