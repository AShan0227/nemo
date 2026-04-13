"""Tests for Dempster-Shafer fusion."""

from src.screen.fusion import EvidenceMass, dempster_combine, fuse_multiple


def test_agreeing_sources():
    m1 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    m2 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    combined, conflict = dempster_combine(m1, m2)
    assert combined.beliefs["button"] > 0.9
    assert conflict < 0.1


def test_conflicting_sources():
    m1 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    m2 = EvidenceMass({"text": 0.9, "unknown": 0.1})
    combined, conflict = dempster_combine(m1, m2)
    assert conflict > 0.5


def test_multiple_fusion():
    m1 = EvidenceMass({"button": 0.7, "unknown": 0.3})
    m2 = EvidenceMass({"button": 0.6, "unknown": 0.4})
    m3 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    combined, _ = fuse_multiple(m1, m2, m3)
    assert combined.beliefs["button"] > 0.8
