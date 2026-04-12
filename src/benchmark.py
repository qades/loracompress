"""
Benchmarking framework to compare two models' outputs.
"""
import torch
import numpy as np
from transformers import set_seed
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from model_loader import get_device


@dataclass
class ComparisonResult:
    """Result of comparing two model outputs."""
    match: bool
    output_a: str
    output_b: str
    token_match: bool
    token_similarity: float
    logits_mse: float
    logits_max_diff: float
    generated_tokens_a: List[int]
    generated_tokens_b: List[int]
    
    def to_dict(self):
        return asdict(self)


class ModelComparator:
    """Compare outputs from two models given the same inputs."""
    
    def __init__(self, model_a, model_b, tokenizer, device: str = "auto"):
        self.model_a = model_a
        self.model_b = model_b
        self.tokenizer = tokenizer
        self.device = get_device(device)
        
        # Set models to eval mode
        self.model_a.eval()
        self.model_b.eval()
    
    @torch.no_grad()
    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        seed: int = 42,
        return_logits: bool = True,
    ) -> ComparisonResult:
        """
        Compare outputs from both models for a given prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            seed: Random seed for reproducibility
            return_logits: Whether to compute and return logits comparison
        """
        set_seed(seed)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "top_k": top_k if do_sample else 50,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate from model A
        set_seed(seed)
        outputs_a = self.model_a.generate(
            **inputs,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Generate from model B
        set_seed(seed)
        outputs_b = self.model_b.generate(
            **inputs,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Decode outputs
        generated_tokens_a = outputs_a.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        generated_tokens_b = outputs_b.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        
        output_text_a = self.tokenizer.decode(generated_tokens_a, skip_special_tokens=True)
        output_text_b = self.tokenizer.decode(generated_tokens_b, skip_special_tokens=True)
        
        # Check exact token match
        token_match = generated_tokens_a == generated_tokens_b
        
        # Calculate token similarity (Jaccard-like metric for sequences)
        token_similarity = self._sequence_similarity(generated_tokens_a, generated_tokens_b)
        
        # Compare logits if requested
        logits_mse = float('inf')
        logits_max_diff = float('inf')
        
        if return_logits and hasattr(outputs_a, 'scores') and hasattr(outputs_b, 'scores'):
            logits_mse, logits_max_diff = self._compare_logits(outputs_a.scores, outputs_b.scores)
        
        # Determine overall match
        match = token_match and logits_mse < 1e-5
        
        return ComparisonResult(
            match=match,
            output_a=output_text_a,
            output_b=output_text_b,
            token_match=token_match,
            token_similarity=token_similarity,
            logits_mse=logits_mse,
            logits_max_diff=logits_max_diff,
            generated_tokens_a=generated_tokens_a,
            generated_tokens_b=generated_tokens_b,
        )
    
    def _sequence_similarity(self, seq_a: List[int], seq_b: List[int]) -> float:
        """Calculate similarity between two token sequences."""
        if len(seq_a) == 0 and len(seq_b) == 0:
            return 1.0
        if len(seq_a) == 0 or len(seq_b) == 0:
            return 0.0
        
        # Use LCS (Longest Common Subsequence) based similarity
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs_length = dp[m][n]
        return 2 * lcs_length / (len(seq_a) + len(seq_b))
    
    def _compare_logits(self, scores_a: Tuple[torch.Tensor], scores_b: Tuple[torch.Tensor]) -> Tuple[float, float]:
        """Compare logits from two generations."""
        mse_list = []
        max_diff_list = []
        
        min_len = min(len(scores_a), len(scores_b))
        
        for i in range(min_len):
            logits_a = scores_a[i][0]  # [vocab_size]
            logits_b = scores_b[i][0]  # [vocab_size]
            
            mse = torch.mean((logits_a - logits_b) ** 2).item()
            max_diff = torch.max(torch.abs(logits_a - logits_b)).item()
            
            mse_list.append(mse)
            max_diff_list.append(max_diff)
        
        avg_mse = np.mean(mse_list) if mse_list else float('inf')
        avg_max_diff = np.mean(max_diff_list) if max_diff_list else float('inf')
        
        return avg_mse, avg_max_diff
    
    def run_benchmark(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> Dict:
        """Run benchmark on multiple prompts and aggregate results."""
        results = []
        matches = 0
        token_matches = 0
        similarities = []
        mse_list = []
        
        for prompt in prompts:
            result = self.compare(prompt, **generation_kwargs)
            results.append({
                "prompt": prompt,
                **result.to_dict()
            })
            
            if result.match:
                matches += 1
            if result.token_match:
                token_matches += 1
            similarities.append(result.token_similarity)
            mse_list.append(result.logits_mse)
        
        summary = {
            "total_prompts": len(prompts),
            "exact_matches": matches,
            "match_rate": matches / len(prompts),
            "token_matches": token_matches,
            "token_match_rate": token_matches / len(prompts),
            "avg_token_similarity": np.mean(similarities),
            "min_token_similarity": np.min(similarities),
            "avg_logits_mse": np.mean(mse_list),
            "max_logits_mse": np.max(mse_list),
            "all_pass": matches == len(prompts),
        }
        
        return {
            "summary": summary,
            "results": results,
        }


def print_comparison_result(result: ComparisonResult, prompt: str = ""):
    """Pretty print a comparison result."""
    print("=" * 80)
    print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    print("-" * 80)
    print(f"Match: {'✓ YES' if result.match else '✗ NO'}")
    print(f"Token Match: {'✓ YES' if result.token_match else '✗ NO'}")
    print(f"Token Similarity: {result.token_similarity:.4f}")
    print(f"Logits MSE: {result.logits_mse:.6e}")
    print(f"Logits Max Diff: {result.logits_max_diff:.6e}")
    print("-" * 80)
    print(f"Output A: {result.output_a[:200]}..." if len(result.output_a) > 200 else f"Output A: {result.output_a}")
    print(f"Output B: {result.output_b[:200]}..." if len(result.output_b) > 200 else f"Output B: {result.output_b}")
    print("=" * 80)
