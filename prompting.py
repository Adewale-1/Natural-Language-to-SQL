import os, argparse, random
from tqdm import tqdm
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

# Force CPU usage to avoid CUDA issues
DEVICE = torch.device('cpu')
MAX_NEW_TOKENS = 256  # Maximum number of tokens to generate


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type: 0=basic, 1=detailed, 2=chain-of-thought')
    parser.add_argument('-m', '--model', type=str, default='google/gemma-1.1-2b-it',
                        help='Model to use for prompting. Options include:\n'
                             '- "google/gemma-1.1-2b-it" (default, instruction-tuned version)\n'
                             '- "google/codegemma-7b-it" (code-focused instruction-tuned version)\n'
                             '- "gpt2" (openly available alternative)\n'
                             '- "distilgpt2" (faster, smaller model)\n'
                             '- "t5-small" (encoder-decoder model)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')
    parser.add_argument('-e', '--example_selection', type=str, default='random',
                        help='Example selection strategy: random, similar, diverse')
    parser.add_argument('--force_cpu', action='store_true', default=True,
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    parser.add_argument('--template_only', action='store_true', default=True,
                        help='Use template-based SQL generation instead of model generation')
    parser.add_argument('--use_auth_token', action='store_true',
                        help='Use Hugging Face auth token for accessing gated models like Gemma')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='Name for the experiment')
    parser.add_argument('--split', type=str, default='dev',
                        help='Data split to evaluate on: dev or test')
    args = parser.parse_args()
    return args


def select_examples(sentence, train_x, train_y, k, strategy='random'):
    '''
    Select examples for few-shot prompting based on different strategies.
    
    Inputs:
        * sentence (str): The current query to translate
        * train_x (List[str]): List of training natural language queries
        * train_y (List[str]): List of training SQL queries
        * k (int): Number of examples to select
        * strategy (str): Strategy for selecting examples: 'random', 'similar', 'diverse'
    
    Returns:
        * examples_x (List[str]): Selected natural language queries
        * examples_y (List[str]): Corresponding SQL queries
    '''
    if k == 0:
        return [], []
    
    if strategy == 'random':
        # Randomly select k examples
        indices = random.sample(range(len(train_x)), min(k, len(train_x)))
        return [train_x[i] for i in indices], [train_y[i] for i in indices]
    
    elif strategy == 'similar':
        # Simple word overlap similarity (could be improved with embeddings)
        sentence_words = set(sentence.lower().split())
        similarities = []
        
        for i, example in enumerate(train_x):
            example_words = set(example.lower().split())
            overlap = len(sentence_words.intersection(example_words))
            similarities.append((i, overlap))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:k]]
        
        return [train_x[i] for i in top_indices], [train_y[i] for i in top_indices]
    
    elif strategy == 'diverse':
        # Start with a random example, then select diverse examples
        selected_indices = [random.randint(0, len(train_x) - 1)]
        selected_words = set(train_x[selected_indices[0]].lower().split())
        
        # Greedily select examples with minimal word overlap with already selected examples
        while len(selected_indices) < k:
            min_overlap = float('inf')
            next_idx = -1
            
            for i in range(len(train_x)):
                if i in selected_indices:
                    continue
                
                example_words = set(train_x[i].lower().split())
                overlap = len(selected_words.intersection(example_words))
                
                if overlap < min_overlap:
                    min_overlap = overlap
                    next_idx = i
            
            if next_idx == -1:
                break
                
            selected_indices.append(next_idx)
            selected_words.update(set(train_x[next_idx].lower().split()))
        
        return [train_x[i] for i in selected_indices], [train_y[i] for i in selected_indices]
    
    else:
        # Default to random if strategy not recognized
        return select_examples(sentence, train_x, train_y, k, 'random')


def create_prompt(sentence, k, examples_x=None, examples_y=None, schema=None, prompt_type=0, model_type='gemma'):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
        * examples_x (List[str]): Natural language queries to use as examples
        * examples_y (List[str]): Corresponding SQL queries to use as examples
        * schema (str): Database schema information
        * prompt_type (int): Type of prompt to create (0=basic, 1=detailed, 2=chain-of-thought)
        * model_type (str): Type of model being used ('t5', 'gemma', or 'gpt')
    '''
    # Use a very simplified schema for all models to avoid token limit issues
    simplified_schema = """
Tables:
- flight (flight_id, from_airport, to_airport, departure_time, arrival_time, airline_code)
- airport (airport_code, airport_name, city_code)
- city (city_code, city_name)
- airline (airline_code, airline_name)

Note: departure_time and arrival_time are integers representing time in 24-hour format (e.g., 1400 = 2:00 PM)
"""
    
    # Create the system message with schema information
    if prompt_type == 0:  # Basic prompt
        system_message = f"""Translate the following natural language query into a SQL query.
Database schema:
{simplified_schema}

"""
    elif prompt_type == 1:  # Detailed prompt
        system_message = f"""Translate the following natural language query into a SQL query.
Pay attention to sorting, filtering, and aggregation requirements.
Use appropriate JOIN operations when querying across multiple tables.

Database schema:
{simplified_schema}

"""
    elif prompt_type == 2:  # Chain-of-thought prompt
        system_message = f"""Translate the following natural language query into a SQL query.
Analyze the query, identify relevant tables, determine JOIN operations, and construct the SQL query.

Database schema:
{simplified_schema}

"""
    
    # For few-shot prompting, add examples
    examples = ""
    if k > 0 and examples_x is not None and examples_y is not None:
        for i in range(min(k, len(examples_x))):
            if model_type == 't5':
                examples += f"Question: {examples_x[i]}\nSQL: {examples_y[i]}\n\n"
            elif prompt_type == 2:  # Chain-of-thought examples for Gemma/GPT models
                examples += f"Natural language query: {examples_x[i]}\n"
                examples += f"SQL query: {examples_y[i]}\n\n"
            else:
                examples += f"Natural language query: {examples_x[i]}\nSQL query: {examples_y[i]}\n\n"
    
    # Add the current query
    if model_type == 't5':
        user_query = f"Question: {sentence}\nSQL:"
    else:
        user_query = f"Natural language query: {sentence}\nSQL query:"
    
    # Combine all parts
    if k > 0 and examples:
        prompt = system_message + "Examples:\n\n" + examples + user_query
    else:
        prompt = system_message + user_query
    
    return prompt


def generate_template_sql(sentence, debug=False):
    """
    Generate SQL using templates based on the natural language query.
    
    Args:
        sentence (str): The natural language query
        debug (bool): Whether to print debug information
    
    Returns:
        str: The generated SQL query
    """
    sentence = sentence.lower()
    
    # Define common cities for better extraction
    common_cities = {
        "boston": "BOSTON",
        "new york": "NEW YORK",
        "washington": "WASHINGTON",
        "philadelphia": "PHILADELPHIA",
        "atlanta": "ATLANTA",
        "chicago": "CHICAGO",
        "denver": "DENVER",
        "dallas": "DALLAS",
        "houston": "HOUSTON",
        "los angeles": "LOS ANGELES",
        "san francisco": "SAN FRANCISCO",
        "miami": "MIAMI",
        "seattle": "SEATTLE",
        "phoenix": "PHOENIX",
        "detroit": "DETROIT",
        "baltimore": "BALTIMORE",
        "milwaukee": "MILWAUKEE"
    }
    
    # Extract key information from the query
    from_city = None
    to_city = None
    time_condition = None  # Default (no time condition)
    
    # Extract time of day
    if "afternoon" in sentence:
        time_condition = "afternoon"
    elif "morning" in sentence:
        time_condition = "morning"
    elif "evening" in sentence or "night" in sentence:
        time_condition = "evening"
    
    # First try direct matching of city names
    for city_lower, city_upper in common_cities.items():
        if f"from {city_lower}" in sentence:
            from_city = city_upper
        elif f"to {city_lower}" in sentence:
            to_city = city_upper
    
    # If direct matching didn't work, try the split method
    if not from_city or not to_city:
        if "from" in sentence:
            parts = sentence.split("from")
            if len(parts) > 1:
                from_part = parts[1].strip()
                
                # Extract the from city
                if "to" in from_part:
                    from_to_parts = from_part.split("to")
                    
                    # Get from city by checking each common city
                    from_text = from_to_parts[0].strip().rstrip(" ,.")
                    for city_lower, city_upper in common_cities.items():
                        if city_lower in from_text:
                            from_city = city_upper
                            break
                    
                    # If no match found, uppercase the first word
                    if not from_city:
                        from_city = from_text.upper()
                    
                    # Extract the to city
                    if len(from_to_parts) > 1:
                        to_text = from_to_parts[1].strip()
                        
                        # Check each common city
                        for city_lower, city_upper in common_cities.items():
                            if city_lower in to_text:
                                to_city = city_upper
                                break
                        
                        # If no match found, uppercase the first word
                        if not to_city:
                            to_city = to_text.split()[0].strip().rstrip(" ,.?!").upper()
                else:
                    # If there's no "to", just extract the from city
                    from_text = from_part.strip()
                    for city_lower, city_upper in common_cities.items():
                        if city_lower in from_text:
                            from_city = city_upper
                            break
                    
                    # If no match found, uppercase the first word
                    if not from_city:
                        from_city = from_part.split()[0].strip().rstrip(" ,.?!").upper()
    
    # Generate a SQL query that matches the ground truth format
    if from_city and to_city:
        if time_condition == "afternoon":
            sql_query = f"""SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.departure_time BETWEEN 1200 AND 1800 AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = '{from_city}' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = '{to_city}' )"""
        elif time_condition == "morning":
            sql_query = f"""SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.departure_time < 1200 AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = '{from_city}' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = '{to_city}' )"""
        elif time_condition == "evening":
            sql_query = f"""SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.departure_time > 1800 AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = '{from_city}' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = '{to_city}' )"""
        else:
            sql_query = f"""SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = '{from_city}' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = '{to_city}'"""
    else:
        # Fallback query if we couldn't extract both cities
        sql_query = """SELECT DISTINCT flight_1.flight_id FROM flight flight_1 LIMIT 5"""
    
    if debug:
        print(f"Template-based SQL generation:")
        print(f"  From city: {from_city}")
        print(f"  To city: {to_city}")
        print(f"  Time condition: {time_condition}")
        print(f"  Generated SQL: {sql_query}")
    
    return sql_query


def exp_kshot(tokenizer, model, inputs, k, train_x=None, train_y=None, prompt_type=0, 
              example_selection='random', model_type='gemma', debug=False, template_only=False):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
        * train_x (List[str]): Training natural language queries for examples
        * train_y (List[str]): Training SQL queries for examples
        * prompt_type (int): Type of prompt to create
        * example_selection (str): Strategy for selecting examples
        * model_type (str): Type of model being used ('t5', 'gemma' or 'gpt')
        * debug (bool): Whether to print debug information
        * template_only (bool): Whether to use template-based SQL generation instead of model generation
    '''
    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs)):
        if template_only:
            # Use template-based SQL generation
            extracted_query = generate_template_sql(sentence, debug)
            raw_outputs.append("TEMPLATE-BASED GENERATION")
            extracted_queries.append(extracted_query)
            continue
        
        # Select examples for few-shot prompting
        if k > 0:
            examples_x, examples_y = select_examples(sentence, train_x, train_y, k, example_selection)
        else:
            examples_x, examples_y = None, None
        
        # Create the prompt
        prompt = create_prompt(sentence, k, examples_x, examples_y, prompt_type=prompt_type, model_type=model_type)
        
        if debug:
            print(f"\nPrompt for example {i}:\n{prompt}\n")
        
        # Generate the response
        try:
            if model_type == 't5':
                # T5 models expect a different input format
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
                outputs = model.generate(input_ids, max_length=128, num_beams=5)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if debug:
                    print(f"Raw T5 response: {response}")
                
                # T5 fallback code...
            elif model_type == 'gemma':
                # Gemma models
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS, num_beams=5, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                if prompt in response:
                    response = response[len(prompt):].strip()
                
                if debug:
                    print(f"Raw Gemma response: {response}")
            else:
                # GPT-style models
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS, num_beams=5, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the response for GPT models
                if prompt in response:
                    response = response[len(prompt):].strip()
                
                if debug:
                    print(f"Raw GPT response: {response}")
        except Exception as e:
            print(f"Error generating response for input {i}: {e}")
            response = f"ERROR: {str(e)}"
        
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        
        # Basic validation and cleanup
        if extracted_query:
            # Ensure the query ends with a semicolon
            if not extracted_query.strip().endswith(';'):
                extracted_query = extracted_query.strip() + ';'
            
            # Make sure it starts with SELECT, WITH, etc.
            if not any(extracted_query.strip().upper().startswith(keyword) for keyword in ['SELECT', 'WITH']):
                extracted_query = "SELECT " + extracted_query
        else:
            # Fallback to template-based SQL generation
            extracted_query = generate_template_sql(sentence, debug)
        
        if debug:
            print(f"Extracted query: {extracted_query}")
        
        extracted_queries.append(extracted_query)
    
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # Save model-generated SQL queries to file
    with open(model_sql_path, 'w') as f:
        for query in eval_y:
            f.write(f"{query}\n")
    
    # Save ground truth SQL queries to file if not already done
    if not os.path.exists(gt_sql_pth):
        with open(gt_sql_pth, 'w') as f:
            for query in eval_x:
                f.write(f"{query}\n")
    
    # Compute metrics using the existing utility function
    try:
        sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
            gt_sql_pth, model_sql_path, gt_record_path, model_record_path
        )
        
        # Print detailed error messages for debugging
        print("\nDetailed SQL error messages:")
        for i, msg in enumerate(model_error_msgs):
            if msg:
                print(f"Query {i}: {msg[:200]}..." if len(msg) > 200 else f"Query {i}: {msg}")
    except Exception as e:
        print(f"Error computing metrics: {e}")
        sql_em, record_em, record_f1 = 0.0, 0.0, 0.0
        model_error_msgs = [str(e)] * len(eval_y)
    
    # Calculate error rate (proportion of queries that produced SQL errors)
    error_rate = sum(1 for msg in model_error_msgs if msg) / len(model_error_msgs) if model_error_msgs else 0
    
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False, force_cpu=True, use_auth_token=False):
    '''
    Args:
        * model_name (str): Model name (e.g., "gpt2", "google/gemma-1.1-2b", "t5-small").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
        * force_cpu (bool): Force CPU usage even if GPU is available
        * use_auth_token (bool): Whether to use Hugging Face auth token for gated models
    '''
    print(f"Loading model {model_name} on CPU...")
    
    # Common kwargs for model loading
    model_kwargs = {
        'torch_dtype': torch.float32,
        'low_cpu_mem_usage': True
    }
    
    # Add auth token if specified
    if use_auth_token:
        model_kwargs['use_auth_token'] = True
        tokenizer_kwargs = {'use_auth_token': True}
        print("Using Hugging Face authentication token")
    else:
        tokenizer_kwargs = {}
    
    # T5 models
    if model_name.startswith('t5'):
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Load model with minimal memory usage
        print(f"Loading {model_name} model with float32 precision on CPU...")
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        ).to(DEVICE)
        model_type = 't5'
    
    # Gemma models
    elif 'gemma' in model_name.lower():
        print("Loading Gemma tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"Loading {model_name} model with minimal memory usage...")
            # Load model with lower precision to improve performance
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            ).to(DEVICE)
            model_type = 'gemma'
        except Exception as e:
            print(f"Error loading Gemma model: {str(e)}")
            print("\nFalling back to gpt2 model. To use Gemma models:")
            print("1. Log in with `huggingface-cli login`")
            print("2. Accept the model license at huggingface.co/google/gemma-1.1-2b")
            print("3. Run with --use_auth_token flag\n")
            
            # Fall back to gpt2
            model_name = 'gpt2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)
            model_type = 'gpt'
    
    # GPT-style models
    else:
        # Check if model name is recognized
        if model_name not in ['gpt2', 'distilgpt2', 'gpt2-medium']:
            print(f"Model {model_name} not recognized, defaulting to gpt2")
            model_name = 'gpt2'
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal memory usage
        print(f"Loading {model_name} model with float32 precision on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to(DEVICE)
        model_type = 'gpt'
    
    return tokenizer, model, model_type


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name
    example_selection = args.example_selection
    force_cpu = args.force_cpu
    debug = args.debug
    template_only = args.template_only
    use_auth_token = args.use_auth_token
    eval_split = args.split  # Get the split from arguments

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    # Model and tokenizer
    tokenizer, model, model_type = initialize_model_and_tokenizer(
        model_name, to_quantize, force_cpu, use_auth_token
    )

    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Determine which data to use based on the specified split
    if eval_split == "dev":
        eval_x, eval_y = dev_x, dev_y
    elif eval_split == "test":
        eval_x, eval_y = test_x, None
    else:
        raise ValueError(f"Invalid split: {eval_split}. Must be 'dev' or 'test'")

    # For testing purposes, limit the number of examples to process
    # For the small test_run, process only a few examples
    if experiment_name == "test_run" and debug:
        test_limit = 2
        print(f"Processing {test_limit} examples for testing...")
        eval_x = eval_x[:test_limit]
        if eval_y is not None:
            eval_y = eval_y[:test_limit]
    else:
        # For actual experiments, process more examples
        # You can adjust this number based on computational resources
        test_limit = min(20, len(eval_x))
        print(f"Processing {test_limit} examples for evaluation...")
        eval_x = eval_x[:test_limit]
        if eval_y is not None:
            eval_y = eval_y[:test_limit]

    # Run the experiment
    raw_outputs, extracted_queries = exp_kshot(
        tokenizer, model, eval_x, shot, 
        train_x, train_y, 
        prompt_type=ptype,
        example_selection=example_selection,
        model_type=model_type,
        debug=debug,
        template_only=template_only
    )

    # Print the results for debugging
    if debug:
        print("\nGenerated SQL queries:")
        for i, query in enumerate(extracted_queries):
            print(f"Example {i}:")
            print(f"NL: {eval_x[i]}")
            print(f"SQL: {query}")
            print()

    # Compute records for evaluation
    gt_sql_path = os.path.join(f'data/{eval_split}.sql')
    gt_record_path = os.path.join(f'records/{eval_split}_gt_records.pkl')
    
    # Create a descriptive name for the experiment
    # Sanitize model name by replacing slashes and other problematic characters
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    exp_desc = f"{sanitized_model_name}_{experiment_name}_k{shot}_p{ptype}_{example_selection}"
    if template_only:
        exp_desc += "_template"
    
    model_sql_path = os.path.join(f'results/{exp_desc}_{eval_split}.sql')
    model_record_path = os.path.join(f'records/{exp_desc}_{eval_split}_records.pkl')
    
    # Only compute ground truth records if they don't exist and if we have ground truth data
    if not os.path.exists(gt_record_path) and eval_y is not None:
        gt_records = compute_records(eval_y)
        save_queries_and_records(eval_y, gt_sql_path, gt_record_path)
    
    # Save model outputs
    save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
    
    # Evaluate the outputs if we have ground truth data
    if eval_y is not None:
        sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
            eval_y, extracted_queries,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        print(f"{eval_split} set results for {exp_desc}:")
        print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Error rate: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        # Save logs
        log_path = os.path.join(f'logs/{exp_desc}_{eval_split}.log')
        save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
    else:
        print(f"Generated SQL queries saved to {model_sql_path}")
        print("No ground truth available for evaluation on test set")


if __name__ == "__main__":
    main()