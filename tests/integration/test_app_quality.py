"""
Integration tests focused on functionality and output quality.
"""
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import soundfile as sf

from app import (
    FORGEConfig,
    FORGEFeedback,
    extract_loops,
    generate_drum_oneshots,
    generate_vocal_chops,
    get_audio_hash,
)


def _load_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sample_rate


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


@pytest.mark.integration
class TestOutputQuality:
    """Validate outputs are sane and consistent."""

    def test_loop_outputs_have_audio(self, sample_audio_file_long, mock_gradio_progress):
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=1.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress,
        )

        assert len(loops) > 0
        for loop in loops:
            audio, sample_rate = _load_mono(loop["path"])
            assert sample_rate == FORGEConfig.SAMPLE_RATE
            assert _rms(audio) > 1e-4
            assert np.max(np.abs(audio)) <= 1.0 + 1e-3

    def test_vocal_chops_with_silence_gaps(self, tmp_path, mock_gradio_progress):
        sample_rate = FORGEConfig.SAMPLE_RATE
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        tone = 0.4 * np.sin(2 * np.pi * 440 * t)
        silence = np.zeros(int(sample_rate * 0.3))
        audio = np.concatenate([tone, silence, tone])

        audio_path = tmp_path / "chop_source.wav"
        sf.write(audio_path, audio, sample_rate)

        chops = generate_vocal_chops(
            audio_path=str(audio_path),
            mode="silence",
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.2,
            progress=mock_gradio_progress,
        )

        assert len(chops) >= 2
        for chop_path in chops:
            chop_audio, _ = _load_mono(chop_path)
            assert _rms(chop_audio) > 1e-4

    def test_drum_oneshots_from_impulses(self, tmp_path, mock_gradio_progress):
        sample_rate = FORGEConfig.SAMPLE_RATE
        audio = np.zeros(int(sample_rate * 1.0))
        for hit_time in (0.1, 0.4, 0.7):
            start = int(hit_time * sample_rate)
            audio[start:start + 20] = np.linspace(1.0, 0.0, 20)

        audio_path = tmp_path / "drum_hits.wav"
        sf.write(audio_path, audio, sample_rate)

        oneshots = generate_drum_oneshots(
            audio_path=str(audio_path),
            min_duration=0.05,
            max_duration=0.3,
            progress=mock_gradio_progress,
        )

        assert len(oneshots) >= 2
        for oneshot_path in oneshots:
            oneshot_audio, _ = _load_mono(oneshot_path)
            assert _rms(oneshot_audio) > 1e-4

    def test_audio_hash_is_stable(self, sample_audio_file):
        hash_a = get_audio_hash(sample_audio_file)
        hash_b = get_audio_hash(sample_audio_file)
        assert hash_a == hash_b


@pytest.mark.integration
class TestFeedback:
    """Validate feedback output."""

    def test_feedback_saved_to_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        FORGEConfig.setup_directories()

        result = FORGEFeedback.save_feedback(
            feature="Loop Extraction",
            rating=4,
            comments="Looks good",
            email="test@example.com",
        )

        assert "saved successfully" in result.lower()
        feedback_files = list(Path("feedback").glob("feedback_*.json"))
        assert len(feedback_files) == 1

        with open(feedback_files[0], "r") as handle:
            payload = json.load(handle)

        assert payload["feature"] == "Loop Extraction"
        assert payload["rating"] == 4
        assert payload["comments"] == "Looks good"
        assert payload["email"] == "test@example.com"

