from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch
import os
import gc
import tqdm


class PartialForwardPassCausalLM():
    """
       Implements a model which does a partial forward pass upto some layer l in a given LLM
    """
    
    def __init__(self, model, layer_index):
        self.layer_index = layer_index
        """
        Start with an empty model
        copy the layers of the supplied model upto layer_index
        on forward pass, forward pass the input_ids through the model and return the output of the layer_index layer
        """
        self.model = torch.nn.Sequential()
        for i in range(layer_index):
            self.model.add_module(f"layer_{i}", model.layers[i])

    # forward pass the input_ids through the model and return the output of the layer_index layer
    # there should be an option to return hidden states of all layers
    def forward(self, input_ids, return_hidden_states_last_token=False):
        if return_hidden_states_last_token:
            return self.model(input_ids, output_hidden_states=True).hidden_states[-1]
        else:
            return self.model(input_ids)





def question_embeddings_model_specific(
        csv_path,
        model_name,
        batch_size=16,
):
    """
        Embed each input questions using input specific tokenizer
        get question specific embedding by taking mean embedding of all tokens
        return the embedding tensor
    """
    df = pd.read_csv(csv_path)
    questions = df["prompt"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    # extract the embedding layer of the model and create a new pytorch model with only the embedding layer
    embedding_layer = model.get_input_embeddings()
    # create a new pytorch model copy of the embedding layer.
    # forward pass the input_ids through the embedding layer and take the mean of the embedding for each question
    # return the embedding tensor
    # use bf16 precision
    embedding_model = torch.nn.Sequential(embedding_layer)
    embedding_model.eval()
    embedding_model.to(device)
    embedding_tensor = torch.tensor([], dtype=torch.bfloat16, device=device)
    for i in tqdm.tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i+batch_size]       
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device)
        forward_pass = embedding_model(input_ids)
        embedding_tensor = torch.cat((embedding_tensor, forward_pass.mean(dim=1).to(device)), dim=0)
        del inputs
        gc.collect()
        # if device == "cuda":
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()
        # elif device == "mps":
        #     torch.mps.empty_cache()
        #     torch.mps.synchronize()
        # else:
        #     torch.empty_cache()
        #     torch.synchronize()
    return embedding_tensor

def question_activations_last_token(
        csv_path,
        model_name,
        batch_size=16,
        use_only_last_layer=False,
        use_quantization=False,
):
    """
     The model is not a sentence transfomer but uses transformer architecture
     Do a forward pass on all the question tokens 
     return the activations of the last token as embedding for that question
     if use_only_last_layer is True, only use the activations of the last layer
     if use_only_last_layer is False, use the mean of the activations of all layers for the last token
     use bf16 precision
     batch using batch_size for the forward pass
    """
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
    df = pd.read_csv(csv_path)
    questions = df["prompt"].tolist()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    # use cuda or mps when available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    embedding_tensor = torch.tensor([], dtype=torch.bfloat16)
    for i in tqdm.tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i+batch_size]
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # Get the hidden states from the last layer
        last_hidden_states = outputs.hidden_states[-1]
        # Get the activations for the last token
        last_token_activations = last_hidden_states[:, -1, :]
        if use_only_last_layer:
            last_token_activations = last_token_activations
        else:
            # Stack all hidden states and take mean across layers
            all_hidden_states = torch.stack(outputs.hidden_states)
            last_token_activations = all_hidden_states[:, :, -1, :].mean(dim=0)
        embedding_tensor = torch.cat((embedding_tensor, last_token_activations.to('cpu')), dim=0)
        del inputs, outputs, last_token_activations
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if device == "mps":
            torch.mps.empty_cache()
    return embedding_tensor

def get_question_embeddings(
    csv_path="data/question_order.csv",
    model_name="all-mpnet-base-v2",
    batch_size=16,
):
    """
    Generate embeddings for questions in the order specified in the CSV file.

    Args:
        csv_path (str): Path to the question_order.csv file
        model_name (str): Name of the sentence-transformer model to use

    Returns:
        torch.Tensor: Tensor of shape (num_questions, embedding_dim) containing question embeddings
    """
    df = pd.read_csv(csv_path)
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    )
    model = SentenceTransformer(model_name,
                                model_kwargs= {
                                    "quantization_config": quantization_config,
                                    # "device": "cuda" if torch.cuda.is_available() else "cpu"
                                }
                                
    )
    questions = df["prompt"].tolist()
    embedding_tensor = torch.tensor(
        []
    )  # Initialize an empty tensor to store embeddings
    # for question in questions:
    #     embeddings = model.encode(question, show_progress_bar=True)
    #     embedding_tensor_i = torch.tensor(embeddings)
    #     embedding_tensor = torch.cat((embedding_tensor, embedding_tensor_i.unsqueeze(0)), dim=0)
    #     del question, embeddings
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()
    embedding_tensor = model.encode(questions, show_progress_bar=True, batch_size=batch_size)
    embedding_tensor = torch.tensor(embedding_tensor)  # Convert to tensor
    embedding_tensor = embedding_tensor.float()  # Ensure the tensor is of type float
    print(f"Generated embeddings tensor of shape: {embedding_tensor.shape}")
    return embedding_tensor


if __name__ == "__main__":
    # Set paths
    csv_path = "data/question_order.csv"

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()

    # Generate embeddings
    question_embeddings = question_embeddings_model_specific(csv_path, model_name=args.model_name, batch_size=64)

    # Save the tensor
    output_path = f"data/question_embeddings_{args.model_name.split('/')[1]}.pth"
    torch.save(question_embeddings, output_path)

    print(f"Generated embeddings tensor of shape: {question_embeddings.shape}")
    print(f"Saved embeddings to: {output_path}")
