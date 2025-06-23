# Evaluating-LLM
This project runs, evaluates, and fine-tune a Large Language Model (LLM) in Hugging Face.

# Choice of language model
The goal is to select a language model capable of generating continuations for input prompts. **Meta's Llama 3 language model** was chosen for this task as it is a decoder-only, autoregressive transformer - meaning it generates output by predicting the next token in a sequence based on prior tokens. This makes it inherently well-suited to causual language modeling tasks, such as text continuation, where the model must produce coherent, left-to-right sequences.

Other open-sourced decoder-based models such as Falcon-7B and GPT-J-6B from Hugging Face were considered. However, these models have slower inference and no direct support for parameter-efficient fine-tuning via LoRa (Low-Rank Adaptation). LLama 3, on the other hand, offers strong performance on downstream tasks and is natively compatible with the **Unsloth framework**. Unsloth provides tools for 4-bit quantisation, helping reduce load time and increases inference speed.

# Evaluation measures
To assess the performance of the model before and after fine-tuning, two evaluation strategies are implemented:
1. Prompt continuation with log-probability and perplexity analysis.
2. Model's performance on a benchmark multiple-choice task using the HellaSwag dataset.

## Prompt-continuation & log-probability analysis
The first evaluation measure involved testing the model’s ability to generate a coherent and contextually appropriate continuation of a domain-specific prompt. The prompt selected is: "A vaccine is defined as a...". This prompt was chosen because it originates from the medical domain and thus served as a strong candidate for detecting the influence of fine-tuning on medical text. To evaluate the model’s response, both the generated text and the associated log probabilities were analysed at each generation step.

The log-probability of the token selected by the model at each step was then extracted and summed to compute the total log-probability of the full sequence. To obtain a normalised score, the perplexity of the generated text was computed. **Perplexity** reflects the model’s uncertainty when predicting a sequence – the lower the perplexity, the more confident the model. For deeper interpretability, the top-10 log-probabilities for each of the first 5 generated tokens was visualised. In these bar plots, the actual token selected by the model is highlighted in red. The plots revealed how strongly the model preferred the chosen token relative to other plausible alternatives.

The base model generated the following continuation:

"A vaccine is defined as a suspension of a killed or weakened organism or a portion of an organism, which produces immunity against a disease when administered to a person...” with a log probability of -25.87 and a Perplexity value of 2.24. These results served as a benchmark for comparison after fine-tuning. The generated text is coherent, and the low perplexity suggests strong confidence in the continuation.

## Benchmark evaluation using HellaSwag
The second evaluation strategy focused on the HellaSwag benchmark, a multiple-choice reasoning task available on the Hugging Face Datasets Hub. Each example in the dataset consists of a narrative context followed by four plausible sentence endings. The task is to select the most likely ending based on the model's understanding of language and world knowledge.

To evaluate the model’s performance, the following custom evaluation loop is implemented:

1. Concatenates the context with each of the four endings to form four full input sequences.
2. Computes the log-probability of each full sequence using the model’s output logits.
3. Selects the ending with the highest total log-probability as the predicted answer.
4. Compares this prediction to the ground-truth label provided in the dataset.
   
This method provides not only accuracy scores, but also insights into the model’s internal confidence via log-probability distributions. While the base model achieved only 1 out of 3 correct predictions, this task remains challenging due to the high plausibility of distractor endings. Importantly, the logprobabilities show how the model distributes confidence across all options. Even in incorrect cases, the correct answer was often ranked second, showing partial understanding.

# Fine-tuning corpus and expected effects
To adapt the model to a specialised domain, I selected the MedMCQA dataset, a multiple-choice question set derived from Indian medical entrance exams, available on Hugging Face. The goal was to explore whether fine-tuning the model on domainspecific content would lead to improved performance on medical prompt completions and affect the model’s internal reasoning over general multiple-choice tasks.

## Fine-tuning set-up & implementation
Each MedMCQA entry was reformatted into a prompt–completion pair, where the model is trained to predict the correct answer. Fine-tuning was performed on a subset of 100 training examples using the LoRA (Low-Rank Adaptation) method, which adds a small
number of trainable parameters to a frozen base model, greatly reducing memory requirements.

Key training configuration:
- Training steps: 20
- Batch size: 2 with gradient accumulation
- LoRA rank: 16
- Learning rate: 2e-4
  
The training loss steadily declined from 2.43 to 1.02, indicating effective learning on the small corpus.

## Expected effects on evaluation metrics

Based on the medical nature of the fine-tuning data, I hypothesised that the model would show different types of adaptation across the two evaluation measures:

- Prompt Continuation Task: I expect improvements in fluency and accuracy when continuing medical prompts (e.g., more specific terminology, reduced perplexity).
  
- HellaSwag Benchmark: Since HellaSwag is general-domain, I do not expect large gains in accuracy. However, I anticipate that fine-tuning would alter the internal log-probability distributions over the multiple-choice completions, revealing subtle effects in how the model ranks the different options.

This setup allows for a focused analysis of how domain-specific fine-tuning affects both in-domain (medical) and out-of-domain (general reasoning) tasks.

# Results and analysis

## Prompt continuation: pre- and post- fine-tuning
To measure the model's fluency and confidence in generating domain-relevant text, I prompted both the base and fine-tuned models with: "A vaccine is defined as a". This prompt was chosen due to its relevance to the MedMCQA medical corpus used for
fine-tuning. The generated continuations and associated metrics for both the base model and fine-tuned model are as follows:

*Base-model* 
- Continuation: "A vaccine is defined as a suspension of a killed or weakened organism or a portion of an organism, which produces immunity against a disease when administered to a person.”
- Log probability: -25.87
- Perplexity: 2.24

*Fine-tuned model*
- Continuation: “A vaccine is defined as a biological preparation that provides active acquired immunity to a particular disease. Vaccines are administered by injection or orally and are used to prevent infectious diseases.”
- Log probability: -20.90
- Perplexity: 1.92

The fine-tuned model's continuation is noticeably more detailed and medically precise. It includes phrases like "biological preparation" and "immunity”, which are characteristic of clinical definitions. In contrast, the base model’s output is simpler and less specific. Numerically, the fine-tuned model assigns a higher total log-probability and achieves lower perplexity, indicating greater confidence and fluency. These results validate the hypothesis that fine-tuning on a medical dataset helps steer the model toward domain-specific completions.

## Stochasticity in LLM
An important observation from the token-level log-probability visualisations is that the token ultimately selected by the model (shown in red) is not always the one with the highest probability. This illustrates a core property of autoregressive language models - they generate text stochastically by sampling from a probability distribution over possible next tokens, rather than deterministically selecting the highest probability option every time.

For instance, in the base model’s generation, the selected token was “suspension”, even though it had a lower probability (i.e., more negative log-probability) than “preparation” and “biological”. This stochastic selection allows the model to produce diverse and natural-sounding language, rather than repeating the most likely continuation every time. However, it also means that small changes in the sampling process - or even repeated runs with the same prompt - can lead to different outputs. This behaviour is not a sign of error but rather a fundamental feature of probabilistic generation in modern LLMs.

## HellaSwag benchmark evaluation: pre- vs post- fine-tuning
To evaluate general-domain reasoning and completion ability, both models were tested on the HellaSwag benchmark using three multiple-choice examples. For each example, the model was presented with a context and four plausible continuations. It was asked to select the most likely continuation based on total log-probability over the full sequence (context + ending).

Although both models produced the same number of correct predictions (1 out of 3), fine-tuning had a subtle but observable effect on the internal confidence of the model. While accuracy did not change, the fine-tuned model slightly increased its confidence for some of the correct answers (e.g., for example 2 and 3, the fine-tuned model’s log probability for the correct answer is -90.6 and -142.7, whereas for the base model, it’s log probability is -90.8 and -143.7 respectively). This suggests that fine-tuning adjusted the model's internal probability landscape - even on out-of-domain tasks like HellaSwag. This analysis shows that while fine-tuning improved domain-specific performance, it also influenced the model’s general reasoning behaviour in measurable ways.

# Conclusion 
Although benchmark accuracy remained unchanged, the model's shifting log-probabilities provide evidence that its internal representations were affected by fine-tuning. This is consistent with expectations and demonstrates the power and risks of
domain adaptation in large language models
