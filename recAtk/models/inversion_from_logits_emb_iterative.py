import copy
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from recAtk.models.config import InversionConfig
from recAtk.models.inversion_from_logits_emb import InversionFromLogitsEmbModel
import ipdb

class InversionFromLogitsEmbModelIterative(InversionFromLogitsEmbModel):
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,  
    ) -> torch.Tensor:
        """
        Generate the inverted prompt using iterative self-correction.
        In each iteration, multiple candidates are generated (via beam search),
        and the one with the highest cosine similarity to the frozen embedding is selected.
        """
        if num_recursive_steps is None:
            num_recursive_steps = getattr(self.config, "num_gen_recursive_steps", 5)
        device = next(self.parameters()).device

        # 1. Compute the frozen embedding (i.e., the embedding of the target prompt)
        if "frozen_embeddings" not in inputs or inputs["frozen_embeddings"] is None:
            with torch.no_grad():
                frozen_emb = self.call_embedding_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            inputs["frozen_embeddings"] = frozen_emb.to(device)

        # 2. Initialize the hypothesis: use the original prompt as the initial hypothesis
        hypothesis_ids = inputs["input_ids"]
        inputs["hypothesis_input_ids"] = hypothesis_ids
        inputs["hypothesis_attention_mask"] = inputs["attention_mask"]
        with torch.no_grad():
            hypo_emb = self.call_embedding_model(
                input_ids=hypothesis_ids,
                attention_mask=inputs["attention_mask"],
            )
        inputs["hypothesis_embedding"] = hypo_emb.to(device)

        best_scores = None

        # 3. Iterative self-correction
        for step in range(num_recursive_steps):
            inputs["input_ids"] = inputs["hypothesis_input_ids"]
            inputs["attention_mask"] = inputs["hypothesis_attention_mask"]

            # Call the parent class's generate method; generation_kwargs must include num_beams and num_return_sequences
            generation_kwargs["num_beams"] = 5
            generation_kwargs["num_return_sequences"] = 5
            candidate_prompts = super().generate(inputs, generation_kwargs)
            # Candidate_prompts shape: [batch * num_candidates, seq_length]
            batch_size = inputs["input_ids"].shape[0]
            num_candidates = candidate_prompts.shape[0] // batch_size
            candidate_prompts = candidate_prompts.view(batch_size, num_candidates, -1)

            # Generate attention masks for the candidates
            candidate_attention_mask = (candidate_prompts != self.embedder_tokenizer.pad_token_id).long()
            candidate_prompts_flat = candidate_prompts.view(batch_size * num_candidates, -1)
            candidate_attention_mask_flat = candidate_attention_mask.view(batch_size * num_candidates, -1)

            # Compute embeddings for each candidate prompt
            with torch.no_grad():
                candidate_embeddings = self.call_embedding_model(
                    input_ids=candidate_prompts_flat,
                    attention_mask=candidate_attention_mask_flat,
                )
            candidate_embeddings = candidate_embeddings.view(batch_size, num_candidates, -1)

            # Compute cosine similarity between candidate embeddings and the frozen embedding
            frozen_emb_expanded = inputs["frozen_embeddings"].unsqueeze(1).expand_as(candidate_embeddings)
            cos_sim = torch.nn.functional.cosine_similarity(candidate_embeddings, frozen_emb_expanded, dim=2)

            # Select the best candidate (highest similarity) for each example in the batch
            best_indices = cos_sim.argmax(dim=1)  # [batch]
            best_candidate_prompts = []
            best_candidate_embeddings = []
            best_cos_sim = []
            for i in range(batch_size):
                best_idx = best_indices[i]
                best_candidate_prompts.append(candidate_prompts[i, best_idx])
                best_candidate_embeddings.append(candidate_embeddings[i, best_idx])
                best_cos_sim.append(cos_sim[i, best_idx])
                
            best_candidate_prompts = torch.stack(best_candidate_prompts, dim=0)  # [batch, seq_length]
            best_candidate_embeddings = torch.stack(best_candidate_embeddings, dim=0)  # [batch, hidden_dim]
            best_cos_sim = torch.stack(best_cos_sim, dim=0)  # [batch]

            avg_cos = best_cos_sim.mean().item()
            #print(f"Iteration {step+1}: Average cosine similarity = {avg_cos:.8f}")

            if best_scores is not None and torch.all(torch.isclose(best_cos_sim, best_scores, atol=1e-5)):
                #print(f"Convergence reached at iteration {step+1}. Early stopping.")
                hypothesis_ids = best_candidate_prompts
                break
            
            # Update hypothesis with the best candidates for next iteration
            best_scores = best_cos_sim
            inputs["hypothesis_input_ids"] = best_candidate_prompts
            inputs["hypothesis_attention_mask"] = (best_candidate_prompts != self.embedder_tokenizer.pad_token_id).long()
            inputs["hypothesis_embedding"] = best_candidate_embeddings
            hypothesis_ids = best_candidate_prompts

        return hypothesis_ids