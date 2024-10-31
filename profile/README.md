**Model Summary**  
**Fritz**-2 is a Transformer with 2.7 billion parameters. It was trained using the same data sources as **Fritz**-1.5, augmented with a new data source that consists of various NLP synthetic texts and filtered websites (for safety and educational value). When assessed against benchmarks testing common sense, language understanding, and logical reasoning, **Fritz**-2 showcased a nearly state-of-the-art performance among models with less than 13 billion parameters.

Our model hasn't been fine-tuned through reinforcement learning from human feedback. The intention behind crafting this open-source model is to provide the research community with a non-restricted small model to explore vital safety challenges, such as reducing toxicity, understanding societal biases, enhancing controllability, and more.

**How to Use**  
**Fritz**-2 has been integrated in the transformers version 4.37.0, please ensure that you are using a version equal or higher than it.

**Fritz**-2 is known for having an attention overflow issue (with FP16). If you are facing this issue, please enable/disable autocast on the FritzAttention.forward() function.

**Intended Uses**  
Given the nature of the training data, the **Fritz**-2 model is best suited for prompts using the QA format, the chat format, and the code format.

**QA Format**:  
You can provide the prompt as a standalone question as follows:

Write a detailed analogy between mathematics and a lighthouse.
where the model generates the text after "." . To encourage the model to write more concise answers, you can also try the following QA format using "Instruct: <prompt>\nOutput:"

Instruct: Write a detailed analogy between mathematics and a lighthouse.  
Output: Mathematics is like a lighthouse. Just as a lighthouse guides ships safely to shore, mathematics provides a guiding light in the world of numbers and logic. It helps us navigate through complex problems and find solutions. Just as a lighthouse emits a steady beam of light, mathematics provides a consistent framework for reasoning and problem-solving. It illuminates the path to understanding and helps us make sense of the world around us.  
where the model generates the text after "Output:".

**Chat Format**:  
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?  
Bob: Well, have you tried creating a study schedule and sticking to it?  
Alice: Yes, I have, but it doesn't seem to help much.  
Bob: Hmm, maybe you should try studying in a quiet environment, like the library.  
Alice: ...  
where the model generates the text after the first "Bob:".

**Code Format**:  
```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   primes = []
   for num in range(2, n+1):
       is_prime = True
       for i in range(2, int(math.sqrt(num))+1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           primes.append(num)
   print(primes)
```
where the model generates the text after the comments.

**Notes**:  
**Fritz**-2 is intended for QA, chat, and code purposes. The model-generated text/code should be treated as a starting point rather than a definitive solution for potential use cases. Users should be cautious when employing these models in their applications.

Direct adoption for production tasks without evaluation is out of scope of this project. As a result, the **Fritz**-2 model has not been tested to ensure that it performs adequately for any production-level application. Please refer to the limitation sections of this document for more details.

If you are using transformers<4.37.0, always load the model with `trust_remote_code=True` to prevent side effects.

**Sample Code**:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/fritz-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/fritz-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

**Limitations of Fritz-2**  
- **Generate Inaccurate Code and Facts**: The model may produce incorrect code snippets and statements. Users should treat these outputs as suggestions or starting points, not as definitive or accurate solutions.
- **Limited Scope for Code**: Most of **Fritz**-2's training data is based in Python and common packages such as `typing, math, random, collections, datetime, itertools`. Users should verify API uses if scripts use other packages or languages.
- **Unreliable Responses to Instruction**: The model has not undergone instruction fine-tuning. As a result, it may struggle or fail to adhere to intricate or nuanced instructions provided by users.
- **Language Limitations**: The model is primarily designed for standard English. Informal English, slang, or other languages may pose challenges, leading to potential misinterpretations or errors.
- **Potential Societal Biases**: Despite training data safety measures, **Fritz**-2 may still reflect societal biases, especially when prompted or instructed in specific ways. Users should exercise caution and critical thinking.
- **Toxicity**: Although trained with carefully selected data, the model can produce harmful content if explicitly prompted. **Fritz**-2 is open-source to help the community develop strategies for reducing model toxicity post-pretraining.
- **Verbosity**: **Fritz**-2 often produces verbose or irrelevant responses due to its training data, which includes textbooks, leading to textbook-like responses.

**Training**  
- **Model**:
  - Architecture: Transformer-based model with next-word prediction
  - Context length: 2048 tokens
  - Dataset size: 250B tokens (NLP synthetic data + filtered web data)
  - Training tokens: 1.4T tokens
  - GPUs: 96xA100-80G
  - Training time: 14 days

- **Software**:
  - PyTorch
  - DeepSpeed
  - Flash-Attention

**License**  
The model is licensed under the MIT license.
