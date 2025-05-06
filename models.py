import torch
import torch.nn as nn
from transformers import GPT2Model

# Base GPT-2 classifier model
class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, num_classes=4, model_name="gpt2"):
        super(GPT2ForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.config = self.gpt2.config
        self.classifier = nn.Linear(self.config.n_embd, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # Use the last token's representation for classification
        sequence_output = hidden_states[:, -1, :]
        logits = self.classifier(sequence_output)
        return logits

# Prompt tuning model
class GPT2PromptTuning(nn.Module):
    def __init__(self, num_classes=4, model_name="gpt2", num_prompt_tokens=20):
        super(GPT2PromptTuning, self).__init__()
        self.num_classes = num_classes
        self.num_prompt_tokens = num_prompt_tokens
        
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.config = self.gpt2.config
        
        # Create trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_prompt_tokens, self.config.n_embd)
        )
        
        # Classifier head
        self.classifier = nn.Linear(self.config.n_embd, num_classes)
        
        # Freeze the GPT-2 model parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # Expand prompt embeddings to match batch size
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        
        # Get input embeddings
        inputs_embeds = self.gpt2.wte(input_ids)
        
        # Concatenate prompt embeddings with input embeddings
        combined_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        
        # Adjust attention mask to account for prompt tokens
        prompt_mask = torch.ones(batch_size, self.num_prompt_tokens, device=attention_mask.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use the last token's representation for classification
        sequence_output = hidden_states[:, -1, :]
        logits = self.classifier(sequence_output)
        return logits

# Partial fine-tuning model
class GPT2PartialFineTuning(nn.Module):
    def __init__(self, num_classes=4, model_name="gpt2", freeze_percentage=0.5):
        super(GPT2PartialFineTuning, self).__init__()
        self.num_classes = num_classes
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.config = self.gpt2.config
        self.classifier = nn.Linear(self.config.n_embd, num_classes)
        
        # Freeze lower layers (close to input) based on percentage
        num_layers = len(self.gpt2.h)
        num_to_freeze = int(num_layers * freeze_percentage)
        
        # Freeze embeddings
        for param in self.gpt2.wte.parameters():
            param.requires_grad = False
        for param in self.gpt2.wpe.parameters():
            param.requires_grad = False
        
        # Freeze the specified number of transformer layers
        for i in range(num_to_freeze):
            for param in self.gpt2.h[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen {num_to_freeze}/{num_layers} transformer layers")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        sequence_output = hidden_states[:, -1, :]
        logits = self.classifier(sequence_output)
        return logits