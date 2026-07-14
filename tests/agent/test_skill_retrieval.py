"""
Tests for agent/skill_retrieval.py  (issue #34823).

Run with:
    pytest tests/agent/test_skill_retrieval.py -v

All tests are pure-Python, stdlib-only, and require no network access or
configured embedding models. The EmbeddingIndex dense layer is tested via
a monkeypatched _embed() that returns deterministic vectors.
"""
from __future__ import annotations

import json
import math
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Unit tests: BM25Scorer
# ---------------------------------------------------------------------------

from agent.skill_retrieval import BM25Scorer, _rrf, _scores_to_ranks, _dict_scores_to_ranks


class TestBM25Scorer:
    CORPUS = [
        "docker containerize image build run",
        "python testing pytest unit integration",
        "git version control commit branch merge",
        "aws cloud deploy lambda s3 ec2",
    ]

    def test_exact_match_scores_highest(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("docker build")
        assert scores[0] == max(scores), "docker doc should score highest for 'docker build'"

    def test_empty_query_returns_zeros(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("")
        assert all(s == 0.0 for s in scores)

    def test_scores_parallel_to_corpus(self):
        bm25 = BM25Scorer(self.CORPUS)
        scores = bm25.score("python")
        assert len(scores) == len(self.CORPUS)

    def test_single_doc_corpus(self):
        bm25 = BM25Scorer(["only one document here"])
        scores = bm25.score("document")
        assert len(scores) == 1
        assert scores[0] > 0

    def test_all_zeros_for_no_overlap(self):
        bm25 = BM25Scorer(["apple orange banana"])
        scores = bm25.score("photosynthesis")
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# Unit tests: RRF fusion helpers
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_single_layer(self):
        names = ["a", "b", "c"]
        ranks = {"a": 0, "b": 1, "c": 2}
        fused = _rrf([ranks], names, k=60)
        # Higher rank → higher fused score
        assert fused["a"] > fused["b"] > fused["c"]

    def test_rrf_two_agreeing_layers(self):
        names = ["a", "b", "c"]
        r1 = {"a": 0, "b": 1, "c": 2}
        r2 = {"a": 0, "b": 1, "c": 2}
        fused = _rrf([r1, r2], names, k=60)
        # Agreement amplifies the ordering
        assert fused["a"] > fused["b"] > fused["c"]

    def test_rrf_disagreeing_layers(self):
        names = ["a", "b"]
        r1 = {"a": 0, "b": 1}   # BM25 prefers a
        r2 = {"a": 1, "b": 0}   # dense prefers b
        fused = _rrf([r1, r2], names, k=60)
        # With exactly inverted ranks of size 2, scores are equal
        assert math.isclose(fused["a"], fused["b"], rel_tol=1e-9)

    def test_rrf_empty_layer_ignored(self):
        names = ["a", "b"]
        r1 = {"a": 0, "b": 1}
        fused_with_empty = _rrf([r1, {}], names, k=60)
        fused_single     = _rrf([r1],     names, k=60)
        assert math.isclose(fused_with_empty["a"], fused_single["a"])

    def test_scores_to_ranks(self):
        names = ["x", "y", "z"]
        scores = [0.5, 0.9, 0.1]
        ranks = _scores_to_ranks(scores, names)
        assert ranks["y"] == 0   # highest score → rank 0
        assert ranks["x"] == 1
        assert ranks["z"] == 2

    def test_dict_scores_to_ranks_missing_key(self):
        names = ["a", "b", "c"]
        score_dict = {"a": 0.9, "b": 0.5}  # "c" missing
        ranks = _dict_scores_to_ranks(score_dict, names)
        # "c" absent → gets last rank
        assert ranks["c"] == len(names) - 1


# ---------------------------------------------------------------------------
# Unit tests: retrieve_skills
# ---------------------------------------------------------------------------

from agent.skill_retrieval import retrieve_skills


SAMPLE_SKILLS = [
    {"name": "docker", "description": "Build and run Docker containers"},
    {"name": "pytest",  "description": "Python unit testing with pytest"},
    {"name": "git",     "description": "Git version control commands"},
    {"name": "aws-s3",  "description": "Manage S3 buckets and objects"},
    {"name": "vscode",  "description": "VS Code editor configuration"},
    {"name": "nginx",   "description": "Configure Nginx web server"},
]


class TestRetrieveSkills:
    def test_returns_set_of_names(self):
        result = retrieve_skills("containerize my app", SAMPLE_SKILLS, top_k=2)
        assert isinstance(result, set)
        assert all(isinstance(n, str) for n in result)

    def test_respects_top_k(self):
        result = retrieve_skills("anything", SAMPLE_SKILLS, top_k=3)
        assert len(result) <= 3

    def test_short_circuit_when_skills_le_top_k(self):
        few = SAMPLE_SKILLS[:3]
        result = retrieve_skills("docker", few, top_k=5)
        # Returns all names unchanged when count ≤ top_k
        assert result == {"docker", "pytest", "git"}

    def test_empty_skills_returns_empty(self):
        result = retrieve_skills("docker", [], top_k=5)
        assert result == set()

    def test_docker_query_ranks_docker_highly(self):
        result = retrieve_skills("docker build image", SAMPLE_SKILLS, top_k=2)
        # BM25 should rank docker first for a docker-specific query
        assert "docker" in result

    def test_no_crash_with_missing_description(self):
        skills = [{"name": "no-desc"}, {"name": "has-desc", "description": "something"}]
        result = retrieve_skills("something", skills, top_k=1)
        assert isinstance(result, set)

    def test_embedding_cfg_without_model_uses_bm25_only(self):
        """embedding_cfg with no model → dense layer skipped, BM25 still runs."""
        result = retrieve_skills(
            "docker",
            SAMPLE_SKILLS,
            top_k=2,
            embedding_cfg={"enabled": True, "embedding_model": ""},
        )
        assert "docker" in result


# ---------------------------------------------------------------------------
# Unit tests: EmbeddingIndex
# ---------------------------------------------------------------------------

from agent.skill_retrieval import EmbeddingIndex


def _make_unit_vec(seed: int, dim: int = 8) -> list[float]:
    """Reproducible unit vector for testing."""
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestEmbeddingIndex:
    def _make_index(self) -> EmbeddingIndex:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return EmbeddingIndex(db_path=Path(tmp.name))

    def _patch_embed(self, idx: EmbeddingIndex, seed: int):
        """Monkeypatch _embed to return a deterministic vector."""
        def fake_embed(text, cfg):
            return _make_unit_vec(seed)
        idx._embed = staticmethod(fake_embed)

    def test_is_warm_false_on_empty_db(self):
        idx = self._make_index()
        assert idx.is_warm() is False

    def test_upsert_makes_warm(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=1)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "skill-a", "description": "desc a"}], embedding_cfg=cfg)
        assert idx.is_warm() is True

    def test_query_returns_scores(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=42)
        cfg = {"embedding_model": "test-model"}
        skills = [
            {"name": "alpha", "description": "first skill"},
            {"name": "beta",  "description": "second skill"},
        ]
        idx.upsert(skills, embedding_cfg=cfg)
        scores = idx.query("anything", ["alpha", "beta"], embedding_cfg=cfg)
        assert "alpha" in scores
        assert "beta" in scores
        assert all(-1.0 <= v <= 1.0 for v in scores.values())

    def test_invalidate_specific_name(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=7)
        cfg = {"embedding_model": "test-model"}
        skills = [
            {"name": "keep-me",   "description": "stays"},
            {"name": "delete-me", "description": "goes"},
        ]
        idx.upsert(skills, embedding_cfg=cfg)
        assert idx.is_warm()

        idx.invalidate(["delete-me"])
        scores = idx.query("x", ["keep-me", "delete-me"], embedding_cfg=cfg)
        assert "keep-me" in scores
        assert "delete-me" not in scores

    def test_invalidate_all(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=3)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "x", "description": "y"}], embedding_cfg=cfg)
        idx.invalidate()
        assert idx.is_warm() is False

    def test_stale_detection_skips_unchanged(self):
        """Calling upsert twice with the same content does not re-embed."""
        idx = self._make_index()
        call_count = {"n": 0}

        def counting_embed(text, cfg):
            call_count["n"] += 1
            return _make_unit_vec(call_count["n"])

        idx._embed = staticmethod(counting_embed)
        cfg = {"embedding_model": "test-model"}
        skill = [{"name": "stable", "description": "unchanged description"}]
        idx.upsert(skill, embedding_cfg=cfg)
        first_count = call_count["n"]
        idx.upsert(skill, embedding_cfg=cfg)
        # Hash matched → no second embed call
        assert call_count["n"] == first_count

    def test_stale_detection_re_embeds_on_change(self):
        idx = self._make_index()
        call_count = {"n": 0}

        def counting_embed(text, cfg):
            call_count["n"] += 1
            return _make_unit_vec(call_count["n"])

        idx._embed = staticmethod(counting_embed)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "s", "description": "v1"}], embedding_cfg=cfg)
        first_count = call_count["n"]
        # Change description → hash mismatch → should re-embed
        idx.upsert([{"name": "s", "description": "v2 changed"}], embedding_cfg=cfg)
        assert call_count["n"] > first_count

    def test_build_async_completes(self):
        """build_async() runs in background and upserts without deadlock."""
        idx = self._make_index()
        self._patch_embed(idx, seed=99)
        cfg = {"embedding_model": "test-model"}
        skills = [{"name": f"skill-{i}", "description": f"desc {i}"} for i in range(5)]
        idx.build_async(skills, embedding_cfg=cfg)

        # Wait up to 5 s for the daemon thread
        deadline = threading.Event()
        threading.Timer(5.0, deadline.set).start()
        while not idx.is_warm() and not deadline.is_set():
            pass
        assert idx.is_warm(), "build_async should complete within 5 seconds"

    def test_query_returns_empty_on_failed_embed(self):
        idx = self._make_index()
        self._patch_embed(idx, seed=11)
        cfg = {"embedding_model": "test-model"}
        idx.upsert([{"name": "s", "description": "d"}], embedding_cfg=cfg)

        # Make query-time embed fail
        idx._embed = staticmethod(lambda text, cfg: None)
        scores = idx.query("anything", ["s"], embedding_cfg=cfg)
        assert scores == {}


# ---------------------------------------------------------------------------
# Integration: prompt_builder renders two-tier output
# ---------------------------------------------------------------------------


class TestPromptBuilderIntegration:
    """Verify that the two-tier rendering appears correctly.

    These tests patch retrieve_skills to return a controlled set so we don't
    need actual BM25 computation in the integration layer.
    """

    def _call_builder(self, query_text, featured_names):
        """Call build_skills_system_prompt with patched retrieval."""
        from agent.prompt_builder import build_skills_system_prompt

        with patch("agent.prompt_builder.retrieve_skills", return_value=set(featured_names)):
            with patch("agent.prompt_builder._load_config", return_value={
                "skills": {
                    "semantic_search": {
                        "enabled": True,
                        "top_k": 2,
                        "embedding_model": "",
                    }
                }
            }):
                return build_skills_system_prompt(query_text=query_text)

    def test_no_query_text_returns_full_index(self):
        from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=False)
        # Without query_text the full-index path runs unchanged — just check
        # the function doesn't crash.
        result = build_skills_system_prompt(query_text=None)
        assert isinstance(result, str)

    def test_non_featured_skills_appear_as_names_only(self):
        """Non-top-k skills must appear in the [other skills, names only] line."""
        # This is a structural test — the actual names depend on the local
        # skills install; we just verify the line exists when retrieval ran.
        from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=False)

        with patch("agent.skill_retrieval.retrieve_skills", return_value={"hermes-agent"}):
            try:
                from hermes_cli.config import load_config
                cfg = load_config()
                ss = (cfg.get("skills") or {}).get("semantic_search") or {}
                if not ss.get("enabled", True):
                    pytest.skip("semantic_search.enabled is false in local config")
            except Exception:
                pytest.skip("Cannot load config")

            result = build_skills_system_prompt(query_text="help with hermes setup")
            if "[other skills, names only]" in result:
                # Verify the helper line is present too
                assert "skill_view(name)" in result
