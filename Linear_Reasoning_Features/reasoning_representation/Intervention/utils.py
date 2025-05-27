import json
import re
import torch
import datasets
import json
import re
import os
import random
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

random.seed(8888)
NUll_num = 0

gsm8k_prompt_template = """As an expert problem solver, solve step by step the following mathematical questions.

  Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
  A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. The final answer is 6.

  Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
  A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. The final answer is 5.

  Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
  A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. The final answer is 39.

  Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
  A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. The final answer is 8.

  Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
  A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. The final answer is 9.

  Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
  A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. The final answer is 29.

  Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
  A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. The final answer is 33.

  Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
  A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. The final answer is 8.

  Q: {TARGET_QUESTION}
  A: Let's think step by step. """

def load_prompt_template(ds_name, dataset_dir):
  
  if ds_name in ['GSM8k', 'GSM-symbolic', "MGSM"]:

    template = gsm8k_prompt_template
    template_no_cot = gsm8k_prompt_template
  
  elif ds_name == 'MMLU-Pro':

    dataset = load_from_disk(os.path.join(dataset_dir, 'mmlu-pro')) 

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    # load 5-shot prompts for each category
    prompts_cot = {c: '' for c in categories}
    prompts_no_cot = {c: '' for c in categories}

    print("len(dataset['validation']): ",len(dataset['validation']))
    for d in dataset['validation']:
        prompts_cot[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'
        prompts_no_cot[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + form_options(d['options']) + '\n' + f"The answer is ({d['answer']})." + '\n\n'
    
    template = prompts_cot
    template_no_cot = prompts_no_cot
  

  elif ds_name == 'C-Eval-H':

    categories = ['modern_chinese_history', 'ideological_and_moral_cultivation','logic', 'law', 'chinese_language_and_literature',
    'art_studies','professional_tour_guide','legal_professional','high_school_chinese','high_school_history','middle_school_history']

    # load 5-shot prompts for each category
    prompts_cot = {c: '' for c in categories}
    prompts_no_cot = {c: '' for c in categories}

    for c in categories:
        c_eval_h_item = pd.read_csv(os.path.join(dataset_dir, f'ceval-exam/dev/{c}_dev.csv'))
        c_eval_h_item_list = c_eval_h_item.to_dict(orient='records')
        
        for d in c_eval_h_item_list:
            prompts_cot[c] += '问题: ' + d['question'] + '\n' + form_options_ceval([d['A'], d['B'], d['C'], d['D']]) + '\n' + '让我们一步一步思考，'+d['explanation'] +'所以答案是' + d['answer']+ '\n\n'
            # prompts_no_cot[c] += '问题:' + ' ' + d['question'] + '\n' + form_options_ceval([d['A'], d['B'], d['C'], d['D']]) + '\n' + f"The answer is ({d['answer']})." + '\n\n'
        
        prompts_cot[c] = f"以下是中国关于{c}考试的单项选择题，请选出其中的正确答案。\n" + prompts_cot[c] 
        # prompts_no_cot[c] = "以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。" + prompts_no_cot[c]

    template = prompts_cot
    template_no_cot = None


#   elif ds_name == 'MGSM':
#     prompt = """As an expert problem solver, solve step by step the following mathematical questions. \n"""
#     for i in range(8):
#         prompt += f"Q: {MGSM_prompt_template[language][str(i)]['q']} \nA: {MGSM_prompt_template[language][str(i)]['a']}\n\n"

#     prompt += """Q: {TARGET_QUESTION} \nA: """

#     template = prompt
#     template_no_cot = prompt


  elif ds_name == 'PopQA': #0-shot on PopQA in original paper
    template = None
    template_no_cot = None

    
  return template, template_no_cot



# for gsm_symbolic and gsm8k
def extract_final_answer(model_resp: str) -> float:
    # Remove commas so for example 5,000 becomes 5000
    model_resp = model_resp.replace(",", "")
    # Find the last number
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)[-1]
    # Use float to ensure 3.0 and 3 are the same.
    return float(extracted_num)



def load_dataset(ds_name, dataset_dir, sample_num=3000, split='test'):
   
  if ds_name == 'MMLU-Pro':

    ds = load_from_disk(os.path.join(dataset_dir, 'mmlu-pro'))
    test_data = list(ds[split])


    ds_data = random.sample(test_data, sample_num) # 这里 sample_num = 3000

  elif ds_name == 'PopQA':

    ds = load_from_disk(os.path.join(dataset_dir, 'PopQA'))

    test_data = list(ds[split])

    ds_data = random.sample(test_data, sample_num) # 这里 sample_num = 1000 ?

    
  elif ds_name == 'MGSM': 

    mgsm_zh = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_zh.tsv'), sep='\t')
    mgsm_zh_test = mgsm_zh.values.tolist()
    mgsm_de = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_de.tsv'), sep='\t')
    mgsm_de_test = mgsm_de.values.tolist()
    mgsm_bn = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_bn.tsv'), sep='\t')
    mgsm_bn_test = mgsm_bn.values.tolist()
    mgsm_ja = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_ja.tsv'), sep='\t')
    mgsm_ja_test = mgsm_ja.values.tolist()
    mgsm_te = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_te.tsv'), sep='\t')
    mgsm_te_test = mgsm_te.values.tolist()
    mgsm_ru = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_ru.tsv'), sep='\t')
    mgsm_ru_test = mgsm_ru.values.tolist()
    mgsm_fr = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_fr.tsv'), sep='\t')
    mgsm_fr_test = mgsm_fr.values.tolist()
    mgsm_sw = pd.read_csv(os.path.join(dataset_dir, 'mgsm/mgsm_sw.tsv'), sep='\t')
    mgsm_sw_test = mgsm_sw.values.tolist()

    ds_data_list = mgsm_zh_test + mgsm_de_test + mgsm_bn_test + mgsm_ja_test + mgsm_te_test + mgsm_ru_test + mgsm_fr_test + mgsm_sw_test
    

    ds_data = [{'question': item[0], 'answer': item[1].replace(",", "")} for item in ds_data_list]


  elif ds_name == 'C-Eval-H': 
     
    c_eval_h = []
    for subject in ['modern_chinese_history', 'ideological_and_moral_cultivation','logic', 'law', 'chinese_language_and_literature',
    'art_studies','professional_tour_guide','legal_professional','high_school_chinese','high_school_history','middle_school_history']:
        c_eval_h_item = pd.read_csv(os.path.join(dataset_dir, f'ceval-exam/val/{subject}_val.csv'))
        c_eval_h_item_list = c_eval_h_item.to_dict(orient='records')
        for item in c_eval_h_item_list:
            item['subject'] = subject
        c_eval_h += c_eval_h_item_list
    
    ds_data = c_eval_h

  
  elif ds_name == 'GSM8k':
    ds = load_from_disk(os.path.join(dataset_dir, 'gsm8k/main'))
    ds_data = list(ds[split])
    for entry in ds_data:
        entry['final_answer'] = extract_final_answer(model_resp=entry['answer'])

  elif ds_name == 'GSM-symbolic':
    # 每个template有50个instance，共100个template
    with open(os.path.join(dataset_dir, f'gsm-symbolic_data/GSM_symbolic.jsonl'), 'r') as file:
        ds_data = []
        for line in file:
            ds_data.append(json.loads(line))  # 解析每一行的JSON对象

    print(len(ds_data))
    for entry in ds_data:
        entry['final_answer'] = extract_final_answer(model_resp=entry['answer'])

  elif ds_name == 'MBPP':
    mbpp_ds_full = load_from_disk(os.path.join(dataset_dir, 'mbpp/full'))
    mbpp_ds_full_test = list(mbpp_ds_full[split])
    ds_data = mbpp_ds_full_test


  return ds_data


def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str

def form_options_ceval(options: list):
    option_str = ''
    opts = ['A', 'B', 'C', 'D']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def generate_questions(model, tokenizer, questions, n_new_tokens=200):
    
    # inference on base model
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    input_length = inputs.input_ids.size(1)
    gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)

    gen_text = tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
    
    return gen_text

def set_act_modify_hooks(model, hs=True, mlp=True, attn=True, layer_name=None, model_layers_num=None, attn_name=None, mlp_name=None, direction=None, scale=0.1):
    """
    Works on LLaMA, OLMo, Gemma, Yi, Mistral getting activation values from certain positions
    """

    def modify_activation(name, direction, patch_input):
        def pre_hook(module, input):
            nonlocal direction, scale
            
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8) #direction需要是一个单位向量
            direction = direction.to('cuda')
            
        
            if "hs" in name:  
                if isinstance(input, tuple):
                    direction = direction.to(input[0].dtype)
                    projection = (input[0] @ direction).unsqueeze(-1) * direction
                    new_input = input[0] + scale * projection
                    input[0][:, :, :] = new_input[:, :, :]
                    # input[0] -= (input[0] @ direction).unsqueeze(-1) * direction 
                else:
                    direction = direction.to(input.dtype)
                    projection = (input @ direction).unsqueeze(-1) * direction
                    new_input = input + scale * projection
                    input[:, :, :] = new_input[:, :, :]

                
        def post_hook(module, input, output):
            nonlocal direction, scale
            
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to('cuda')
            
            if "attn" in name or "mlp" in name:
                if isinstance(output, tuple):
                    # print('output[0].shape: ',output[0].shape)
                    direction = direction.to(output[0].dtype)
                    projection = (output[0] @ direction).unsqueeze(-1) * direction
                    new_input = output[0] + scale * projection
                    output[0][:, :, :] = new_input[:, :, :]
     
                else:
                    direction = direction.to(output.dtype)
                    projection = (output @ direction).unsqueeze(-1) * direction
                    new_input = output + scale * projection
                    output[:, :, :] = new_input[:, :, :]  
                    
        if patch_input:
            return pre_hook
        else:
            return post_hook
    
    hooks = []
    attributes = layer_name.split(".")
    current_obj = model
    for attr in attributes:
        current_obj = getattr(current_obj, attr)
    layers_variable = current_obj

    for layer in range(model_layers_num):  
        if hs == True:
            hooks.append(layers_variable[layer].register_forward_pre_hook(modify_activation("hs_" + str(layer), direction=direction, patch_input=True)))
        if attn == True:
            hooks.append(getattr(layers_variable[layer], attn_name).register_forward_hook(modify_activation("attn_" + str(layer), direction=direction, patch_input=False)))
        if mlp == True:
            hooks.append(getattr(layers_variable[layer], mlp_name).register_forward_hook(modify_activation("mlp_" + str(layer), direction=direction, patch_input=False)))

    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
        
        
def generate_questions_in_hook(model, tokenizer, questions, ablation_dir, scale, layer_name, attn_name, mlp_name, model_layers_num, n_new_tokens=200):
    
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    input_length = inputs.input_ids.size(1)

    # 将所有token位置，所有layer位置的表征往 reasoning or memory的方向上去推动
    hooks = set_act_modify_hooks(model, hs=True, mlp=True, attn=True, layer_name=layer_name, model_layers_num=model_layers_num, attn_name=attn_name, mlp_name=mlp_name, direction=ablation_dir, scale=scale)
    gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)
    remove_hooks(hooks)

    gen_text = tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
    
    return gen_text

        
def evaluation_on_dataset(model, tokenizer, val_sampled_data=None, prompts_cot=None, prompts_no_cot=None, run_in_fewshot=True, run_in_cot=True, 
                          intervention=False, ablation_dir=None, layer_name = None, model_layers_num=None, attn_name = None, mlp_name = None, batch_size=4, scale=0.1, ds_name='MMLU-Pro'):
    
    queries_batch = []  
    entry_batch = []
    
    # per_category_accuracy = {c: [0, 0] for c in categories}
    
    for ix, entry in tqdm(enumerate(val_sampled_data)):
        
        if ds_name == 'MMLU-Pro':
            prefix_cot = prompts_cot[entry['category']]
            prefix_no_cot = prompts_no_cot[entry['category']]
            
            if run_in_fewshot:
                if run_in_cot:
                    query = prefix_cot + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + "\n\nA: Let's think step by step. " #cot
                else:
                    query = prefix_no_cot + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + "\n\nA: " #no cot
            else:
                query = 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n' + "\nA: "
                #query = 'Q: ' + entry['question'] + '\n' + '\nA: '
            
        elif ds_name in ['GSM8k', 'GSM-symbolic', 'MGSM']:
            prefix_cot = prompts_cot
            prefix_no_cot = prompts_no_cot
            
            if run_in_fewshot:
                if run_in_cot:
                    query = prompts_cot.format(TARGET_QUESTION=entry['question'])
                else:
                    query = prompts_no_cot.format(TARGET_QUESTION=entry['question'])
            else:
                query = 'Q: ' + entry['question'] + "\n\nA: "


        elif ds_name in ['PopQA']:
            query = 'Q: ' + entry['question'] + "\n\nA: "

        elif ds_name in ['C-Eval-H']:
            prefix_cot = prompts_cot[entry['subject']]
            prefix_no_cot = None
            
            if run_in_fewshot:
                if run_in_cot:
                    query = prefix_cot + '问题: ' + entry['question'] + '\n' + form_options([entry['A'], entry['B'], entry['C'], entry['D']]) + '\n' + '让我们一步一步思考，'
                # else:
                #     query = prefix_no_cot + '问题: ' + entry['question'] + '\n' + form_options([entry['A'], entry['B'], entry['C'], entry['D']]) + "\n\n答案: " 

        queries_batch.append(query)
        entry_batch.append(entry)

        if len(queries_batch) == batch_size or ix == len(val_sampled_data) - 1:
            
            if intervention:
                # generate in interventioning...
                responses = generate_questions_in_hook(model = model, tokenizer = tokenizer, questions = queries_batch, ablation_dir=ablation_dir, scale=scale, layer_name = layer_name, attn_name = attn_name, mlp_name = mlp_name, model_layers_num=model_layers_num, n_new_tokens=200)
            else:
                responses = generate_questions(model = model, tokenizer = tokenizer, questions = queries_batch, n_new_tokens=200)
                
            # metric calculating...
            
            if ds_name == 'MMLU-Pro': 
                for answer, entry in zip(responses, entry_batch):

                    entry['solution'] = answer
                    prediction = get_prediction(answer, ds_name)

                    if entry["answer"] == prediction:
                        # success += 1
                        # per_category_accuracy[entry['category']][0] += 1
                        entry['model_predict_correctness'] = True
                    else:
                        # fail += 1
                        # per_category_accuracy[entry['category']][1] += 1
                        entry['model_predict_correctness'] = False

            elif ds_name in ['GSM8k', 'GSM-symbolic']:  #还要再看看其他的数据形式
                for answer, entry in zip(responses, entry_batch):
                    
                    entry['solution'] = answer
                    prediction = get_prediction(answer, ds_name)

                    if entry["final_answer"] == prediction:
                        # success += 1
                        entry['model_predict_correctness'] = True
                    else:
                        # fail += 1
                        entry['model_predict_correctness'] = False

            elif ds_name in ['MGSM']:  
                for answer, entry in zip(responses, entry_batch):

                    prediction = get_prediction(answer, ds_name)

                    if float(entry['answer']) == prediction:
                        # success += 1
                        entry['model_predict_correctness'] = True
                    else:
                        # fail += 1
                        entry['model_predict_correctness'] = False
            
            elif ds_name in ['C-Eval-H']:  
                for answer, entry in zip(responses, entry_batch):

                    prediction = get_prediction(answer, ds_name)


                    if entry['answer'] == prediction:
                        # success += 1
                        entry['model_predict_correctness'] = True
                    else:
                        # fail += 1
                        entry['model_predict_correctness'] = False

            elif ds_name == 'PopQA':
                # compute accuracy for PopQA

                for answer, entry in zip(responses, entry_batch):

                    entry['solution'] = answer

                    possible_answers = json.loads(entry['possible_answers'])
                    print('possible_answers: ',possible_answers)        
                    is_correct = False
                    for pa in possible_answers:
                        if pa in answer or pa.lower() in answer or pa.capitalize() in answer:
                            is_correct = True

                    entry['model_predict_correctness'] = is_correct

                    
            queries_batch = []
            entry_batch = []
            
def compute_performance_on_reason_memory_subset(val_sampled_data=None, memory_indices=None, reason_indices=None, intervention=False, intervention_layer=None):
    
    val_sampled_memory_data = [val_sampled_data[i] for i in memory_indices]
    val_sampled_reason_data = [val_sampled_data[i] for i in reason_indices]

    correct_predictions = 0
    total_predictions = len(val_sampled_memory_data)

    for ix, entry in tqdm(enumerate(val_sampled_memory_data)):
        if entry['model_predict_correctness'] == True:
            correct_predictions += 1
    
    memory_accuracy = correct_predictions / total_predictions
    

    correct_predictions = 0
    total_predictions = len(val_sampled_reason_data)

    for ix, entry in tqdm(enumerate(val_sampled_reason_data)):
        if entry['model_predict_correctness'] == True:
            correct_predictions += 1

    reason_accuracy = correct_predictions / total_predictions
    
    if intervention:
        print(f"***Intervention in Layer {intervention_layer}, Memory Subset Accuracy: {memory_accuracy:.4f}")
        print(f"***Intervention in Layer {intervention_layer}, Reason Subset Accuracy: {reason_accuracy:.4f}")
    
    else:
        print(f"***Original performance of Memory Subset Accuracy: {memory_accuracy:.4f}")
        print(f"***Original performance of Reason Subset Accuracy: {reason_accuracy:.4f}")

def compute_performance_on_reason_subset(val_sampled_data=None, intervention=False, ds_name=None, intervention_layer=None):

    
    correct_predictions = 0
    total_predictions = len(val_sampled_data)

    for ix, entry in tqdm(enumerate(val_sampled_data)):
        if entry['model_predict_correctness'] == True:
            correct_predictions += 1

    reason_accuracy = correct_predictions / total_predictions
    
    if intervention:
        print(f"***Intervention in Layer {intervention_layer}, Reason Subset {ds_name} Accuracy: {reason_accuracy:.4f}")
    
    else:
        print(f"***Original performance of Reason Subset {ds_name} Accuracy: {reason_accuracy:.4f}")

        
def get_prediction(output=None, ds_name='MMLU-Pro'):
    
    if ds_name == 'MMLU-Pro':
        pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
        match = re.search(pattern, output)
        if match:
            #print('prediction success: ',match.group(1))
            return match.group(1)
        else:
            #print("extraction failed, do a random guess")
            global NUll_num  
            NUll_num += 1
            return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        
    elif ds_name in ['GSM8k', 'GSM-symbolic', 'MGSM']:
        
        model_resp = output 
        
        match = re.search(r"The final answer is (\d+\.?\d*)", model_resp)
        if match:
            return float(match.group(1))  # 返回数字作为 float 类型
        else:
            return None  # 如果没有找到匹配的数字，返回 None

        return float(extracted_num)

    elif ds_name in ['C-Eval-H']:
        
        text = output
        # 定义正则表达式，匹配 "所以答案是" 后面的选项
        match = re.search(r"所以答案是\s*([ABCD])", text)
        
        # 如果找到了匹配，返回结果，否则返回None
        if match:
            return match.group(1)
        else:
            return random.choice(['A', 'B', 'C', 'D'])


def get_candidate_directions(hs_cache_no_cot, model_layers_num, mlp_dim_num, reason_indices, memory_indices):

    candidate_directions = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float64, device='cuda')

    # calculating candidate reasoning features
    for layer in range(model_layers_num):
            
        hs_no_cot = hs_cache_no_cot[layer]

        #  we store the mean activations in high-precision to avoid numerical issues
        reason_hs_no_cot = hs_no_cot[reason_indices, :].to(torch.float64)
        #print('reason_hs_no_cot.shape: ',reason_hs_no_cot.shape) reason有点多，memory有点少，需要进一步把数据集做scale up    
        memory_hs_no_cot = hs_no_cot[memory_indices, :].to(torch.float64)

        mean_reason_hs_no_cot = reason_hs_no_cot.mean(dim=0)
        mean_memory_hs_no_cot = memory_hs_no_cot.mean(dim=0)

        mean_diff = mean_reason_hs_no_cot - mean_memory_hs_no_cot  #Reasoning features shape: [bsz, dims] 
        candidate_directions[layer] = mean_diff

    return candidate_directions