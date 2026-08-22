# Presentation-ready project summary

## Research Question

Do language models actually use conversational context to detect sarcasm, or do they mostly rely on superficial cues inside the comment itself?

## Dataset

- Balanced Reddit sarcasm dataset.
- 10,000 total examples.
- 5,000 sarcastic and 5,000 not sarcastic.
- The dataset was balanced to avoid majority-class bias.

## Main Results for Slides

| Group | Method | Setting | Accuracy | Macro F1 | Sarcastic F1 |
|---|---|---|---:|---:|---:|
| Baseline | TF-IDF + Logistic Regression | context_plus_comment | 0.6500 | 0.6499 | 0.6436 |
| Baseline | TF-IDF + Logistic Regression | comment_only | 0.6430 | 0.6426 | 0.6301 |
| Baseline | TF-IDF + Logistic Regression | context_only | 0.5425 | 0.5425 | 0.5395 |
| Large language model | Qwen2.5-0.5B-Instruct + LoRA | context_plus_comment | 0.6550 | 0.6537 | 0.6326 |
| Representation analysis | Sentence Transformer | separate_context_comment_embeddings_concat | 0.6335 | 0.6334 | 0.6380 |
| Representation analysis | Sentence Transformer | joint_context_comment_embedding | 0.6175 | 0.6175 | 0.6185 |

## Random Context Ablation

| Group | Method | Setting | Accuracy | Macro F1 | Sarcastic F1 |
|---|---|---|---:|---:|---:|
| Ablation | tfidf_logistic_regression | comment_only | 0.6430 | 0.6426 | 0.6301 |
| Ablation | tfidf_logistic_regression | true_context_plus_comment | 0.6430 | 0.6430 | 0.6394 |
| Ablation | sentence_transformer_separate_concat | random_context_plus_comment_separate | 0.6370 | 0.6370 | 0.6348 |
| Ablation | sentence_transformer_separate_concat | true_context_plus_comment_separate | 0.6335 | 0.6334 | 0.6380 |
| Ablation | sentence_transformer_joint | true_context_plus_comment_joint | 0.6175 | 0.6175 | 0.6185 |
| Ablation | tfidf_logistic_regression | random_context_plus_comment | 0.6085 | 0.6083 | 0.5991 |
| Ablation | sentence_transformer_joint | random_context_plus_comment_joint | 0.5690 | 0.5689 | 0.5642 |

## Best Overall Result

- Best Accuracy: 0.6550
- Method: Qwen2.5-0.5B-Instruct + LoRA
- Setting: context_plus_comment
- Macro F1: 0.6537
- Sarcastic F1: 0.6326

## Main Findings

1. The comment alone is much stronger than the context alone.
2. Adding the true context gives a small improvement in some settings.
3. Random context hurts TF-IDF and joint Sentence Transformer, so the specific context can matter.
4. Separate Sentence Transformer embeddings performed better than joint embeddings, but random context did not hurt them. This suggests that this representation may rely mostly on the comment.
5. Qwen LoRA achieved the best overall Accuracy and Macro F1, but not the best Sarcastic F1.

## Error Analysis Counts

- TF-IDF context helped: 152 examples.
- TF-IDF random context hurt: 372 examples.
- Sentence Transformer joint context helped: 419 examples.
- Sentence Transformer separate true/random same prediction: 1661 examples.
- All main models wrong: 306 examples.

## Slide-ready Conclusion

Context can help sarcasm detection, but simply adding context is not enough.

Some models are sensitive to the specific conversational context, while others appear to rely mostly on the comment itself.

Therefore, context-aware sarcasm detection should be evaluated not only by adding context, but also by testing whether the model uses the correct context through ablation experiments.

## Future Work

1. Train Qwen 1.5B with LoRA.
2. Run random-context ablation also on Qwen.
3. Train for more epochs and with more examples.
4. Fine-tune an encoder-only model such as BERT, RoBERTa, or DeBERTa.
5. Add attention or token-level analysis.
6. Expand qualitative error analysis for the final August report.
