"""
Reward utility functions for GRPO / RL training.

Mirrors Unsloth's built-in reward functions for R1-style reasoning training.
All functions follow the TRL reward-function signature:
    fn(prompts, completions, **kwargs) -> List[float]
"""

import re
from typing import List, Optional, Any


# ─── Format Rewards ───────────────────────────────────────────────────────────

def reward_format_think_answer(completions: List[Any], **kwargs) -> List[float]:
    """
    Reward 1.0 if the completion contains <think>...</think><answer>...</answer>.
    Partial reward 0.5 if only one tag pair is present.
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_think or has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def reward_format_xml_tags(completions: List[Any], tags: List[str] = None, **kwargs) -> List[float]:
    """
    Generic XML tag format reward. Rewards 1/len(tags) per correctly found tag pair.
    Default tags: ['think', 'answer']
    """
    if tags is None:
        tags = ["think", "answer"]
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        score = 0.0
        for tag in tags:
            if re.search(rf"<{tag}>.*?</{tag}>", text, re.DOTALL):
                score += 1.0 / len(tags)
        rewards.append(score)
    return rewards


def reward_no_repetition(completions: List[Any], window: int = 10, **kwargs) -> List[float]:
    """
    Penalizes repetitive n-grams. Returns 1.0 if no repeated window-grams, 
    scales down linearly with repetition ratio.
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        words = text.split()
        if len(words) < window:
            rewards.append(1.0)
            continue
        ngrams = [tuple(words[i:i+window]) for i in range(len(words) - window + 1)]
        unique_ratio = len(set(ngrams)) / len(ngrams)
        rewards.append(unique_ratio)
    return rewards


def reward_length_penalty(
    completions: List[Any],
    min_length: int = 50,
    max_length: int = 500,
    **kwargs,
) -> List[float]:
    """
    Rewards completions in the [min_length, max_length] word range.
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        n = len(text.split())
        if min_length <= n <= max_length:
            rewards.append(1.0)
        elif n < min_length:
            rewards.append(n / min_length)
        else:
            rewards.append(max(0.0, 1.0 - (n - max_length) / max_length))
    return rewards


# ─── Correctness Rewards ──────────────────────────────────────────────────────

def reward_exact_match(
    completions: List[Any],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward 1.0 if the answer inside <answer>...</answer> exactly matches ground truth.
    """
    rewards = []
    for comp, gt in zip(completions, ground_truths):
        text = _extract_text(comp)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match and match.group(1).strip().lower() == gt.strip().lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_math_answer(
    completions: List[Any],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """
    Math-specific reward: extracts numeric answer from <answer> tag
    and compares with tolerance to ground truth.
    """
    rewards = []
    for comp, gt in zip(completions, ground_truths):
        text = _extract_text(comp)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        try:
            predicted = float(re.sub(r"[^\d.\-]", "", match.group(1).strip()))
            expected = float(re.sub(r"[^\d.\-]", "", gt.strip()))
            if abs(predicted - expected) < 1e-4:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def reward_code_compiles(
    completions: List[Any],
    language: str = "python",
    **kwargs,
) -> List[float]:
    """
    Reward 1.0 if the code block inside the completion compiles without syntax errors.
    Currently supports Python only.
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        code_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        code = code_match.group(1) if code_match else text
        try:
            compile(code, "<string>", "exec")
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


# ─── Composite Rewards ────────────────────────────────────────────────────────

def build_reward_fn(*fns, weights: List[float] = None):
    """
    Compose multiple reward functions into one weighted sum.

    Example:
        reward_fn = build_reward_fn(
            reward_format_think_answer,
            reward_no_repetition,
            weights=[0.7, 0.3],
        )
    """
    if weights is None:
        weights = [1.0 / len(fns)] * len(fns)
    assert len(weights) == len(fns), "weights must match number of functions"

    def composite(completions, **kwargs):
        total = [0.0] * len(completions)
        for fn, w in zip(fns, weights):
            scores = fn(completions, **kwargs)
            for i, s in enumerate(scores):
                total[i] += w * s
        return total

    return composite


# ─── R1-Style Default Reward ──────────────────────────────────────────────────

def r1_reward_fn(completions: List[Any], ground_truths: List[str] = None, **kwargs) -> List[float]:
    """
    Default R1-style combined reward used in Unsloth's GRPO notebooks:
      - 50% format reward (think + answer tags)
      - 30% no repetition
      - 20% exact match (if ground_truths provided), else length reward
    """
    fmt = reward_format_think_answer(completions, **kwargs)
    rep = reward_no_repetition(completions, **kwargs)

    if ground_truths is not None:
        corr = reward_exact_match(completions, ground_truths, **kwargs)
    else:
        corr = reward_length_penalty(completions, **kwargs)

    return [0.5 * f + 0.3 * r + 0.2 * c for f, r, c in zip(fmt, rep, corr)]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(completion: Any) -> str:
    """Extract string from various TRL completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat format: list of dicts with 'content' key
        parts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
            else:
                parts.append(str(msg))
        return " ".join(parts)
    return str(completion)
