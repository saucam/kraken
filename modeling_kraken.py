import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TextClassificationPipeline
from configuration_kraken import KrakenConfig
import tokenizer_template_switch

class KrakenForCausalLM(PreTrainedModel):
    config_class = KrakenConfig

    def __init__(self, config):
        super().__init__(config)
        self.tokenizers = {key: AutoTokenizer.from_pretrained(name) for key, name in config.config_dict['tokenizers'].items()}
        self.models = {key: AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True) for key, name in config.config_dict['models'].items()}
        self.router_model = AutoModelForSequenceClassification.from_pretrained(config.config_dict['router'], trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.config_dict['router'], trust_remote_code=True)
        self.router = TextClassificationPipeline(model=self.router_model, tokenizer=self.tokenizer)
        self.models_indices = config.config_dict['class_indices']

    def tokenize_inputs(self, text, model_key):
        return self.tokenizers[model_key](text, return_tensors="pt")

    def determine_model(self, text):
        prediction = self.router(text)[0]["label"]
        model_decision_index = self.models_indices[prediction]
        model_keys = ['expert1', 'expert2', 'expert3', 'expert4']
        return model_keys[model_decision_index]
    
    def expert_tokenizer(self, text):
        model_key = self.determine_model(text)
        return self.tokenizers[model_key]

    def generate(self, input_ids, **generate_kwargs):
        # Tokenize the input_ids
        text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]

        msgs = tokenizer_template_switch.recover_chat_messages(text, self.tokenizer)

        # Check if the last element has the role 'assistant'
        if msgs and msgs[-1]['role'] == 'assistant':
            # Delete the last element
            msgs.pop()  

        # Determine the model key using the existing routing logic
        model_key = self.determine_model(text)
        
        # Retrieve the model from the dictionary
        model = self.models[model_key]

        mod_txt = self.tokenizers[model_key].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Tokenize accordingly to the best model
        tok = self.tokenizers[model_key](mod_txt, return_tensors="pt").input_ids  
        
        # Generate text using the retrieved model
        return model.generate(tok, **generate_kwargs)
    
