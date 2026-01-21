from fyp_sim.models import User, UserPhenotype, Video


def test_video_has_duration():
    v = Video(
        video_id=1,
        topic_category="Education",
        viewpoint_score=0.5,
        sentiment_score=0.0,
        duration_s=42,
    )
    assert v.duration_s == 42


def test_user_has_phenotype():
    u = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.7,
        interest_vector={"Education": 0.9, "Entertainment": 0.4},
        sentiment_threshold=0.2,
    )
    assert u.phenotype == UserPhenotype.SAMPLER


def test_video_has_tags_default_empty():
    v = Video(
        video_id=2,
        topic_category="Entertainment",
        viewpoint_score=0.3,
        sentiment_score=0.5,
        duration_s=15,
    )
    assert v.tags == ()


def test_video_accepts_tags():
    v = Video(
        video_id=3,
        topic_category="News",
        viewpoint_score=0.8,
        sentiment_score=-0.1,
        duration_s=10,
        tags=("breaking", "world"),
    )
    assert v.tags == ("breaking", "world")
