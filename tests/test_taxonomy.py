from fyp_sim.taxonomy import TOPIC_CATEGORIES, TOPIC_LABELS, normalise_token


def test_categories_are_unique():
    assert len(TOPIC_CATEGORIES) == len(set(TOPIC_CATEGORIES))


def test_labels_cover_all_categories():
    assert set(TOPIC_LABELS.keys()) == set(TOPIC_CATEGORIES)


def test_normalise_token_examples():
    assert normalise_token("Health & Wellness") == "health_wellness"
    assert normalise_token("DIY & Life hacks") == "diy_life_hacks"
