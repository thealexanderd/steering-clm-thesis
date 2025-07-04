import torch
import pandas as pd
from steering_vectors import (
    train_steering_vector,
    train_steering_vector_field
)
import logging
from drugGen_generator import load_model_and_tokenizer, setup_logging
from check_smiles import check_smiles

from helpers import generate_pairs_size, get_num_atoms
from helpers import get_solubility_predictions, create_solubility_pairs, evaluate_solubility
from helpers import create_lipo_pairs, predict_logp

model_name = "alimotahharynia/DrugGen"
model, tokenizer = load_model_and_tokenizer(model_name)


config = {
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "temperature": 1.2,
                "num_return_sequences": 50
            },
            "max_retries": 30
        }

generation_kwargs = config["generation_kwargs"]
generation_kwargs["bos_token_id"] = tokenizer.bos_token_id
generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_smiles(sequence, num_generated, unique=True):
    """
    Generates SMILES strings based on a given input sequence using a pre-trained model.

    Args:
      sequence (str): The input sequence or prompt to guide SMILES generation.
      num_generated (int): The number of SMILES strings to generate.
      unique (bool, optional): If True, ensures all generated SMILES are unique. 
        If False, allows duplicates. Defaults to True.

    Returns:
      tuple: A tuple containing:
        - list: The list of generated SMILES strings (unique or not, based on `unique`).
        - int: The total number of SMILES strings generated (including invalid or duplicate ones).
        - int: The number of invalid SMILES strings generated.

    Notes:
      - The function uses a tokenizer and model assumed to be defined in the global scope.
      - Invalid SMILES are filtered out using the `check_smiles` function.
      - The function will retry generation up to a maximum number of retries specified in `config["max_retries"]`.
      - If the maximum number of retries is reached before generating the requested number of SMILES, 
        the function returns what has been generated so far.
    """
    total_generated = 0
    invalid_generated = 0
    generated_smiles_set = set()
    generated_smiles_list = []
    prompt = f"<|startoftext|><P>{sequence}<L>"
    encoded_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    retries = 0

    if unique:
      statement = len(generated_smiles_set) < num_generated
    else:
      statement = len(generated_smiles_list) < num_generated

    while statement:
      if not unique and total_generated >= num_generated:
        return generated_smiles_list, total_generated, invalid_generated

      if retries >= config["max_retries"]:
          logging.warning("Max retries reached. Returning what has been generated so far.")
          break

      print(retries)

      sample_outputs = model.generate(encoded_prompt, **generation_kwargs)
      total_generated += len(sample_outputs)

      for sample_output in sample_outputs:
          output_decode = tokenizer.decode(sample_output, skip_special_tokens=False)
          try:
              generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
              if len(check_smiles(generated_smiles)) > 0:
                    invalid_generated += 1
                    continue
              if unique and generated_smiles not in generated_smiles_set:
                  generated_smiles_set.add(generated_smiles)
              elif not unique:
                  generated_smiles_list.append(generated_smiles)
          except (IndexError, AttributeError) as e:
              logging.warning(f"Failed to parse SMILES due to error: {str(e)}. Skipping.")

      retries += 1
      if unique:
        statement = len(generated_smiles_set) < num_generated
      else:
        statement = len(generated_smiles_list) < num_generated

    if unique:
      return list(generated_smiles_set), total_generated, invalid_generated
    else:
      return generated_smiles_list, total_generated, invalid_generated
    
    
def evaluate_group(label, vector, multiplier, protein, num_to_generate, eval_fn, tau=None, field=False):
    """
    Evaluates a group of generated SMILES strings for a given protein using a specified evaluation function.

    Args:
      label (str): Label identifying the group being evaluated.
      vector: A vector object with an `apply` context manager for model manipulation.
      multiplier (float): Multiplier parameter passed to the model during vector application.
      protein (str): Identifier or sequence of the protein for which SMILES are generated.
      num_to_generate (int): Number of SMILES strings to generate.
      eval_fn (callable): Evaluation function applied to each generated SMILES string. Should accept a SMILES string and a scores list.
      tau (float, optional): Temperature parameter for generation, used if `field` is True. Defaults to None.
      field (bool, optional): Whether to use the temperature parameter during generation. Defaults to False.

    Returns:
      dict or None: A dictionary containing:
        - "label": The group label.
        - "scores": List of evaluation scores for valid SMILES.
        - "smiles": List of valid SMILES strings.
        - "set": Set of unique valid SMILES strings.
        - "total_generated": Total number of SMILES generated.
        - "invalid_generated": Number of invalid SMILES generated.
      Returns None if no SMILES are generated.
    """
    print(f"Evaluating group {label} of protein {protein}")

    if field:
      with vector.apply(model, multiplier=multiplier, temperature=tau):
        outputs, total, invalid = generate_smiles(protein, num_to_generate, unique=False)
    else:
      with vector.apply(model, multiplier=multiplier):
          outputs, total, invalid = generate_smiles(protein, num_to_generate, unique=False)

    if len(outputs) == 0:
        print("No smiles generated")
        return None
    scores, valid_smiles = [], []

    for s in outputs:
        try:
            eval_fn(s, scores)
            valid_smiles.append(s)
        except:
            print(f"Eval failed on: {s}")

    print(scores)
    return {
        "label": label,
        "scores": scores,
        "smiles": valid_smiles,
        "set": set(valid_smiles),
        "total_generated": total,
        "invalid_generated": invalid
    }


def get_generate_steering_vector_and_eval_function(type_of_experiment, field=False):
    functions = None
    if type_of_experiment == "size":
        if field:
            functions = (get_num_atoms, generate_steering_vector_size_field)
        else:
            functions = (get_num_atoms, generate_steering_vector_size)

    elif type_of_experiment == "logs":
        if field:
            functions = (evaluate_solubility, generate_steering_vector_solubility_field)
        else:
            functions = (evaluate_solubility, generate_steering_vector_solubility)

    elif type_of_experiment == "logp":
        if field:
            functions = (predict_logp, generate_steering_vector_lipophilicity_field)
        else:
            functions = (predict_logp, generate_steering_vector_lipophilicity)
    else:
        raise ValueError(f"Unknown type of experiment: {type_of_experiment}")
    
    return functions


def generate_steering_vector_size(protein_sequence, duplicates = True, num_to_generate=50, token_index=-1, layer=11):
    """
    Generates a steering vector for a given protein sequence by generating SMILES strings, 
    creating protein-SMILES pairs, and training a steering vector on these pairs.

    Args:
      protein_sequence (str): The input protein sequence for which to generate the steering vector.
      duplicates (bool, optional): Whether to allow duplicate protein-SMILES pairs. Defaults to True.
      num_to_generate (int, optional): Number of SMILES strings to generate. Defaults to 50.
      token_index (int, optional): Token index to be used during steering vector training. Defaults to -1.
      layer (int, optional): Model layer to extract the steering vector from. Defaults to 11.

    Returns:
      tuple: A tuple containing:
        - steering_vector (np.ndarray): The trained steering vector.
        - int: The number of generated protein-SMILES pairs.
        - list: The list of generated protein-SMILES pairs.
    """
    print(f"Generating steering vector:")
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    pairss = generate_pairs_size(protein_sequence, generated_smiles, duplicates)

    steering_vector = train_steering_vector(
        model,
        tokenizer,
        pairss,
        show_progress=True,
        read_token_index=token_index,
        layers=[layer],
    )

    return steering_vector, len(pairss), pairss


def generate_steering_vector_size_field(protein_sequence, duplicates = True, num_to_generate=50, token_index=-1, layer=11):
    """
    Generates a steering vector field for a given protein sequence using generated SMILES strings.

    This function generates a specified number of SMILES strings conditioned on the input protein sequence,
    creates pairs for training, and then trains a steering vector field using the provided model and tokenizer.
    It returns the trained steering vector field, the number of generated pairs, and the pairs themselves.

    Args:
      protein_sequence (str): The input protein sequence to condition the SMILES generation.
      duplicates (bool, optional): Whether to allow duplicate pairs in the generated data. Defaults to True.
      num_to_generate (int, optional): Number of SMILES strings to generate. Defaults to 50.
      token_index (int, optional): The token index to read from during steering vector field training. Defaults to -1.
      layer (int, optional): The model layer to use for steering vector field training. Defaults to 11.

    Returns:
      tuple: A tuple containing:
        - steering_vector_field: The trained steering vector field.
        - int: The number of generated pairs.
        - list: The list of generated pairs used for training.
    """
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    pairss = generate_pairs_size(protein_sequence, generated_smiles, duplicates)

    steering_vector_field = train_steering_vector_field(
        model,
        tokenizer,
        pairss,
        show_progress=True,
        layers=[layer],
        read_token_index=token_index,
    )

    return steering_vector_field, len(pairss), pairss

    
def generate_steering_vector_solubility(protein_sequence, num_to_generate=50, duplicates=True, token_index=-1, layer=11):
    """
    Generates a steering vector for solubility based on a given protein sequence.

    This function generates SMILES strings from the provided protein sequence, predicts their solubility,
    creates solubility-based pairs, and then trains a steering vector using these pairs. The steering vector
    can be used to guide the model towards generating molecules with desired solubility properties.

    Args:
      protein_sequence (str): The input protein sequence to condition the SMILES generation.
      num_to_generate (int, optional): Number of SMILES strings to generate. Defaults to 50.
      duplicates (bool, optional): Whether to allow duplicate pairs in the solubility set. Defaults to True.
      token_index (int, optional): The token index to read from during steering vector training. Defaults to -1.
      layer (int, optional): The model layer to use for steering vector extraction. Defaults to 11.

    Returns:
      tuple: A tuple containing:
        - steering_vector (np.ndarray): The trained steering vector for solubility.
        - int: The number of solubility pairs used for training.
        - list: The list of solubility pairs (input, label) used for training.
    """
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    predictions = get_solubility_predictions(generated_smiles)

    pairs_solubility = create_solubility_pairs(protein_sequence, predictions, duplicates)

    steering_vector = train_steering_vector(
        model,
        tokenizer,
        pairs_solubility,
        show_progress=True,
        layers=[layer],
        read_token_index=token_index,
    )

    return steering_vector, len(pairs_solubility), pairs_solubility


def generate_steering_vector_solubility_field(protein_sequence, num_to_generate=50, duplicates=True, token_index=-1, layer=11):
    """
    Generates a steering vector for solubility field based on a given protein sequence.

    This function generates SMILES strings conditioned on the input protein sequence, predicts their solubility,
    creates solubility-based pairs, and trains a steering vector field using these pairs.

    Args:
      protein_sequence (str): The protein sequence to condition the SMILES generation on.
      num_to_generate (int, optional): Number of SMILES strings to generate. Defaults to 50.
      duplicates (bool, optional): Whether to allow duplicate pairs in the solubility pairs. Defaults to True.
      token_index (int, optional): The token index to read from during steering vector training. Defaults to -1.
      layer (int, optional): The model layer to use for steering vector training. Defaults to 11.

    Returns:
      tuple: A tuple containing:
        - steering_vector: The trained steering vector.
        - int: The number of solubility pairs generated.
        - list: The list of solubility pairs used for training.
    """
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    predictions = get_solubility_predictions(generated_smiles)

    pairs_solubility = create_solubility_pairs(protein_sequence, predictions, duplicates)

    steering_vector = train_steering_vector_field(
        model,
        tokenizer,
        pairs_solubility,
        show_progress=True,
        layers=[layer],
        read_token_index=token_index,
    )

    return steering_vector, len(pairs_solubility), pairs_solubility


def generate_steering_vector_lipophilicity(protein_sequence, num_to_generate, duplicates=True, token_index=-1, layer=11):
  """
  Generates a steering vector for lipophilicity based on a given protein sequence.

  This function repeatedly generates SMILES strings for the provided protein sequence until at least one valid lipophilicity pair is created. It then trains a steering vector using these pairs.

  Args:
    protein_sequence (str): The protein sequence to condition the SMILES generation.
    num_to_generate (int): The number of SMILES strings to generate per attempt.
    duplicates (bool, optional): Whether to allow duplicate pairs. Defaults to True.
    token_index (int, optional): The token index to read from during steering vector training. Defaults to -1.
    layer (int, optional): The model layer to use for steering vector training. Defaults to 11.

  Returns:
    tuple: A tuple containing:
      - steering_vector (np.ndarray): The trained steering vector.
      - int: The number of valid lipophilicity pairs generated.
      - list: The list of generated lipophilicity pairs.
  """

  while True:
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    pairs_lipo = create_lipo_pairs(protein_sequence, generated_smiles, duplicates)

    print(len(pairs_lipo))

    if len(pairs_lipo) > 0:
      break

  steering_vector = train_steering_vector(
      model,
      tokenizer,
      pairs_lipo,
      show_progress=True,
      layers=[layer],
      read_token_index=token_index,
  )

  return steering_vector, len(pairs_lipo), pairs_lipo


def generate_steering_vector_lipophilicity_field(protein_sequence, num_to_generate, duplicates=True, token_index=-1, layer=11):
  """
  Generates a steering vector for the lipophilicity field based on a given protein sequence.

  This function repeatedly generates SMILES strings for the provided protein sequence until at least one valid lipophilicity pair is created. It then trains a steering vector field using these pairs and returns the resulting steering vector, the number of valid pairs, and the pairs themselves.

  Args:
    protein_sequence (str): The protein sequence to condition the SMILES generation on.
    num_to_generate (int): The number of SMILES strings to generate per attempt.
    duplicates (bool, optional): Whether to allow duplicate pairs. Defaults to True.
    token_index (int, optional): The token index to read from during steering vector training. Defaults to -1.
    layer (int, optional): The model layer to use for steering vector training. Defaults to 11.

  Returns:
    tuple: A tuple containing:
      - steering_vector (Any): The trained steering vector for the lipophilicity field.
      - num_pairs (int): The number of valid lipophilicity pairs generated.
      - pairs_lipo (list): The list of generated lipophilicity pairs.
  """
  while True:
    generated_smiles, _, _ = generate_smiles(protein_sequence, num_to_generate)

    pairs_lipo = create_lipo_pairs(protein_sequence, generated_smiles, duplicates)

    print(len(pairs_lipo))

    if len(pairs_lipo) > 0:
      break

  steering_vector = train_steering_vector_field(
      model,
      tokenizer,
      pairs_lipo,
      show_progress=True,
      layers=[layer],
      read_token_index=token_index,
  )

  return steering_vector, len(pairs_lipo), pairs_lipo