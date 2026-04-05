import datetime
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from utilities.device_utils import resolve_device
from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator


class KVoiceWalk:
    def __init__(
        self,
        target_audio: Path,
        target_text: str,
        other_text: str,
        voice_folder: str,
        interpolate_start: bool,
        population_limit: int,
        starting_voice: str,
        output_name: str,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        print(f"Using device: {self.device}")
        self.target_audio = target_audio
        self.target_text = target_text
        self.other_text = other_text
        self.initial_selector = InitialSelector(
            str(target_audio),
            target_text,
            other_text,
            voice_folder=voice_folder,
            device=self.device,
        )
        voices: list[torch.Tensor] = []
        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)
        self.speech_generator = SpeechGenerator(device=self.device)
        self.fitness_scorer = FitnessScorer(str(target_audio), device=self.device)
        self.voice_generator = VoiceGenerator(voices, starting_voice, device=self.device)
        self.starting_voice = self.voice_generator.starting_voice
        self.output_name = output_name

    def random_walk(self, step_limit: int, log_interval: int = 100):
        if log_interval < 1:
            log_interval = 1

        best_voice = self.starting_voice
        best_results = self.score_voice(self.starting_voice)

        interactive = sys.stdout.isatty()
        started_at = time.time()

        if interactive:
            loop = tqdm(range(step_limit), desc="Random Walk", unit="step")
            progress_write = loop.write
        else:
            loop = range(step_limit)
            progress_write = print

        progress_write(
            f'Target Sim:{best_results.get("target_similarity", 0.0):.3f}, '
            f'Self Sim:{best_results.get("self_similarity", 0.0):.3f}, '
            f'Feature Sim:{best_results.get("feature_similarity", 0.0):.2f}, '
            f'Score:{best_results.get("score", 0.0):.2f}'
        )

        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f'{self.output_name}_{self.target_audio.stem}_{now.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)

        for i in loop:
            diversity = random.uniform(0.01, 0.15)
            voice = self.voice_generator.generate_voice(best_voice, diversity, device=self.device)

            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice, min_similarity)

            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice
                progress_write(
                    f'Step:{i:<6} Target Sim:{best_results["target_similarity"]:.3f} '
                    f'Self Sim:{best_results["self_similarity"]:.3f} '
                    f'Feature Sim:{best_results["feature_similarity"]:.3f} '
                    f'Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}'
                )
                torch.save(
                    best_voice,
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.pt',
                )
                sf.write(
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
                    best_results["audio"],
                    24000,
                )

            if not interactive and ((i + 1) % log_interval == 0 or i == step_limit - 1):
                elapsed = time.time() - started_at
                print(
                    f'Progress: step {i + 1}/{step_limit} | '
                    f'best_score={best_results.get("score", 0.0):.2f} | '
                    f'best_target_sim={best_results.get("target_similarity", 0.0):.3f} | '
                    f'elapsed={elapsed:.1f}s'
                )

        elapsed = time.time() - started_at
        print(f"Random Walk Final Results for {self.output_name}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Random Walk pt and wav files ---> {results_dir}")

    def score_voice(self, voice: torch.Tensor, min_similarity: float = 0.0) -> dict[str, Any]:
        """Using a harmonic mean calculation to provide a score for the voice in similarity"""
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str, Any] = {
            "audio": audio
        }
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(audio, audio2, target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        return results
