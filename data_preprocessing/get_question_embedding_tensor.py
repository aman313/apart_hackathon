from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
import pandas as pd
import torch
import os
import gc


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

    # Generate embeddings
    question_embeddings = get_question_embeddings(csv_path, model_name="Qwen/Qwen1.5-4B-Chat", batch_size=4)

    # Save the tensor
    output_path = "../data/question_embeddings.pth"
    torch.save(question_embeddings, output_path)

    print(f"Generated embeddings tensor of shape: {question_embeddings.shape}")
    print(f"Saved embeddings to: {output_path}")
