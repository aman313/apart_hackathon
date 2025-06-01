import csv 

def reasoning_non_reasoning_eval_to_mf_dataset(eval_output_path, reasoning_model_id, non_reasoning_model_id):
    """
    input csv headers:
        query,score_match_reasoning,score_match_non_reasoning,id
    output csv headers:
        prompt_id,model_id,category_id,label,prompt,model_name,category
        (category_if can be ignored, so can category_id, fill them with default values or "cat" and cat_id)
        (query is the prompt, id maps to prompt_id)
        (each row in the first file should be mapped to two rows in the second file, one for the reasoning and one for the non-reasoning)
        (C maps to 1 , I maps to 0 from the model specfiic score row to the label column)
    """
    input_rows = []
    with open(eval_output_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        for row in reader:
            input_rows.append(row)
    
    # Create output rows
    output_rows = []
    for row in input_rows:
        query, score_reasoning, score_non_reasoning, prompt_id = row
        
        # Create row for reasoning model
        reasoning_row = [
            prompt_id,  # prompt_id
            reasoning_model_id,  # model_id
            "cat_id",  # category_id (default value)
            "1" if score_reasoning == "C" else "0",  # label (C->1, I->0)
            query,  # prompt
            "reasoning",  # model_name
            "cat"  # category (default value)
        ]
        
        # Create row for non-reasoning model
        non_reasoning_row = [
            prompt_id,  # prompt_id
            non_reasoning_model_id,  # model_id
            "cat_id",  # category_id (default value)
            "1" if score_non_reasoning == "C" else "0",  # label (C->1, I->0)
            query,  # prompt
            "non_reasoning",  # model_name
            "cat"  # category (default value)
        ]
        
        output_rows.extend([reasoning_row, non_reasoning_row])
    
    # Write output to a new file
    output_path = eval_output_path.replace('.csv', '_transformed.csv')
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['prompt_id', 'model_id', 'category_id', 'label', 'prompt', 'model_name', 'category'])
        # Write data rows
        writer.writerows(output_rows)
    
    return output_path

if __name__ == "__main__":
    reasoning_non_reasoning_eval_to_mf_dataset(
        "data/gsm8k_reasoning_non_reasoning_test.csv",
        "DeepSeek-R1-Distill-Qwen-32B",
        "Qwen2.5-32B-Instruct"
    )