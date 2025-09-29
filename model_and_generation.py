import torch
import torch.serialization as _ts

_old_load = torch.load

def _load_legacy(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # restore legacy default
    return _old_load(*args, **kwargs)

# Patch both entry points
torch.load = _load_legacy
_ts.load = _load_legacy
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
import os
import sys
from contextlib import nullcontext
from typing import Sequence

# Steering internals used for Plixer-specific extraction
from steering_vectors.record_activations import record_activations
from steering_vectors.token_utils import adjust_read_indices_for_padding, fix_pad_token
from steering_vectors.train_steering_vector import aggregate_activations
from steering_vectors.steering_vector import SteeringVector
from steering_vectors.steering_vector_field import SteeringVectorField
from steering_vectors.utils import batchify

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


# ===================== Plixer (vox2smiles) support =====================

def _ensure_plixer_import_path():
  """Ensure `models/plixer` is on sys.path so `src.*` imports work."""
  plixer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "plixer"))
  if plixer_root not in sys.path:
    sys.path.insert(0, plixer_root)
  return plixer_root


def load_plixer_model(
    vox2smiles_ckpt_path: str | None = None,
    poc2mol_ckpt_path: str | None = None,
    dtype: torch.dtype = torch.float32,
):
  """Load the combined Plixer model and its config using default checkpoints.

  Returns (combined_model, config).
  """
  _ensure_plixer_import_path()
  # Defaults mirror models/plixer/inference/generate_smiles_from_pdb.py
  default_vox2smiles_ckpt = os.path.join("models", "plixer", "checkpoints", "combined_protein_to_smiles", "epoch_000.ckpt")
  default_poc2mol_ckpt = os.path.join("models", "plixer", "checkpoints", "poc_vox_to_mol_vox", "epoch_173.ckpt")
  vox2smiles_ckpt = vox2smiles_ckpt_path or default_vox2smiles_ckpt
  poc2mol_ckpt = poc2mol_ckpt_path or default_poc2mol_ckpt

  from src.utils.utils import load_model as plixer_load_model  # type: ignore

  device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_plixer, config = plixer_load_model(
      vox2smiles_ckpt,
      poc2mol_ckpt,
      device_t,
      dtype=dtype,
  )
  model_plixer.eval()
  return model_plixer, config


def _plixer_build_poc2mol_config(config, dtype: torch.dtype):
  """Create Poc2MolDataConfig mirroring the inference script defaults."""
  from src.data.common.voxelization.config import Poc2MolDataConfig  # type: ignore
  complex_dataset_config = config.data.train_dataset.poc2mol_output_dataset.complex_dataset.config
  complex_dataset_config = {k: v for k, v in complex_dataset_config.items() if k != "_target_"}
  poc2mol_config = Poc2MolDataConfig(**complex_dataset_config)
  poc2mol_config.random_rotation = False
  poc2mol_config.random_translation = 0.0
  poc2mol_config.dtype = dtype
  return poc2mol_config


def _plixer_voxelize_protein(pdb_file: str, ligand_file: str | None, center: list[float] | None, poc2mol_config):
  from src.utils.utils import voxelize_protein, get_center_from_ligand  # type: ignore
  import numpy as np  # local import to avoid global dep if unused
  if center is not None:
    if ligand_file:
      print("--ligand_file provided with --center, ignoring ligand file and using --center")
    center_arr = np.array(center)
  else:
    if not ligand_file:
      raise ValueError("Either ligand_file or center must be provided for Plixer voxelization.")
    if not os.path.exists(ligand_file):
      raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    center_arr = get_center_from_ligand(ligand_file)
  protein_voxel = voxelize_protein(pdb_file, center_arr, poc2mol_config)
  return protein_voxel


@torch.inference_mode()
def plixer_generate_smiles(
    pdb_file: str,
    *,
    ligand_file: str | None = None,
    center: list[float] | None = None,
    num_samples: int = 10,
    temperature: float = 1.0,
    vox2smiles_ckpt_path: str | None = None,
    poc2mol_ckpt_path: str | None = None,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    steering_vector=None,
    multiplier: float = 1.0,
    field: bool = False,
    tau: float | None = None,
    batch_size: int = 50,
):
  """Generate SMILES with Plixer, optionally applying a steering vector to the decoder.

  Returns (generated_smiles_list, total_generated, invalid_generated).
  """
  torch.manual_seed(seed)
  model_plixer, config = load_plixer_model(vox2smiles_ckpt_path, poc2mol_ckpt_path, dtype=dtype)
  poc2mol_config = _plixer_build_poc2mol_config(config, dtype)
  protein_voxel = _plixer_voxelize_protein(pdb_file, ligand_file, center, poc2mol_config)

  # Forward to get predicted ligand voxels once
  output = model_plixer(protein_voxel, sample_smiles=False)

  # Prepare steering context on the SMILES decoder only
  ctx = nullcontext()
  ctx2 = nullcontext()
  if steering_vector is not None:
    target_model = model_plixer.vox2smiles_model.model  # VisionEncoderDecoderModel
    if field and (tau is not None):
      ctx = steering_vector.apply(target_model, multiplier=multiplier, temperature=tau)
    else:
      ctx = steering_vector.apply(target_model, multiplier=multiplier)
      ctx2 = steering_vector.apply(target_model, multiplier=3)

  generated_smiles_list = []
  invalid_generated = 0
  total_generated = 0
  # repeat predicted_ligand voxel batch_size times
  predicted_ligand_voxels = output['predicted_ligand_voxels']
  predicted_ligand_voxels = predicted_ligand_voxels.repeat(batch_size, 1, 1, 1, 1)
  # no_context_sample = model_plixer.vox2smiles_model.generate_smiles(
  #   predicted_ligand_voxels[:1],
  #   do_sample=False,
  #   max_attempts=10,
  # )
  # with ctx2:
  #   with_context_sample = model_plixer.vox2smiles_model.generate_smiles(
  #     predicted_ligand_voxels[:1],
  #     do_sample=False,
  #     max_attempts=10,
  #   )
  # print(no_context_sample)
  # print(with_context_sample)
  with ctx:
    # Generate num_samples SMILES by repeated sampling from the decoder conditioned on predicted voxels
    needed = num_samples
    while needed > 0:
      smiles_batch = model_plixer.vox2smiles_model.generate_smiles(
          predicted_ligand_voxels,
          do_sample=True,
          temperature=temperature,
          max_attempts=10,
      )
      total_generated += len(smiles_batch)
      print(total_generated)
      for s in smiles_batch:
        if s is None or len(s) == 0:
          invalid_generated += 1
          continue
        generated_smiles_list.append(s)
        needed -= 1
        if needed == 0:
          break

  return generated_smiles_list, total_generated, invalid_generated


def evaluate_group_plixer(
    label,
    vector,
    multiplier,
    pdb_file,
    num_to_generate,
    eval_fn,
    ligand_file=None,
    center=None,
    tau=None,
    field=False,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    temperature: float = 1.0,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
):
  """Evaluate a steering vector on Plixer by steering only the SMILES decoder.

  Mirrors the signature/behavior of `evaluate_group`, but takes a PDB input.
  """
  print(f"Evaluating group {label} for PDB {pdb_file}")

  outputs, total, invalid = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
      steering_vector=vector,
      multiplier=multiplier,
      field=field,
      tau=tau,
  )

  if len(outputs) == 0:
    print("No smiles generated")
    return None

  scores, valid_smiles = [], []
  for s in outputs:
    try:
      eval_fn(s, scores)
      valid_smiles.append(s)
    except Exception:
      print(f"Eval failed on: {s}")

  print(scores)
  return {
      "label": label,
      "scores": scores,
      "smiles": valid_smiles,
      "set": set(valid_smiles),
      "total_generated": total,
      "invalid_generated": invalid,
  }


# ===================== Plixer steering vector training =====================

def _build_plixer_decoder_and_tokenizer(dtype: torch.dtype):
  """Load Plixer and return (decoder_lm, tokenizer) for steering extraction."""
  # Load once to access decoder and tokenizer setup
  model_plixer, _ = load_plixer_model(dtype=dtype)
  # Decoder LM to train vectors on
  decoder_lm = model_plixer.vox2smiles_model.model.decoder
  # Tokenizer consistent with decoder
  from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer  # type: ignore
  tok = build_smiles_tokenizer()
  return decoder_lm, tok


def _pair_by_size_plixer(smiles: list[str], duplicates: bool = True):
  from helpers import get_num_atoms
  with_atom_num = []
  for s in smiles:
    if s:
      try:
        with_atom_num.append((s, get_num_atoms(s)))
      except Exception:
        print(f"Error in getting num atoms for {s}")
  with_atom_num.sort(key=lambda x: x[1])
  pairs = []
  ranges = [0.8, 0.6]
  length_i = len(with_atom_num)
  flag = False
  for i in range(length_i):
    for j in range(i + 1, length_i):
      first = with_atom_num[i]
      second = with_atom_num[j]
      if first[1] <= ranges[0] * second[1] and first[1] >= ranges[1] * second[1]:
        pairs.append((first[0], second[0]))
        flag = True
      elif second[1] <= ranges[0] * first[1] and second[1] >= ranges[1] * first[1]:
        pairs.append((second[0], first[0]))
        flag = True
      if not duplicates and flag:
        with_atom_num.remove(first)
        with_atom_num.remove(second)
        length_i -= 2
        flag = False
        break
  return pairs


def _pair_by_solubility_plixer(smiles: list[str], duplicates: bool = True):
  from helpers import get_solubility_predictions
  preds = get_solubility_predictions(smiles)
  pairs = []
  flag = False
  length_i = len(preds)
  for i in range(length_i):
    for j in range(i + 1, length_i):
      if preds[i][1] < preds[j][1]:
        smaller = preds[i][0]
        bigger = preds[j][0]
        pairs.append((bigger, smaller))
        flag = True
      elif preds[j][1] < preds[i][1]:
        smaller = preds[j][0]
        bigger = preds[i][0]
        pairs.append((bigger, smaller))
        flag = True
      if not duplicates and flag:
        flag = False
        break
  return pairs


def _pair_by_lipophilicity_plixer(smiles: list[str], duplicates: bool = True):
  from helpers import predict_logp
  pairs = []
  length_i = len(smiles)
  flag = False
  for i in range(length_i):
    for j in range(i + 1, length_i):
      logpi = predict_logp(smiles[i])
      if logpi is None:
        continue
      logpj = predict_logp(smiles[j])
      if logpj is None:
        continue
      if 1 <= logpi <= 3 and (logpj < 1 or logpj > 3):
        pairs.append((smiles[i], smiles[j]))
        flag = True
      elif 1 <= logpj <= 3 and (logpi < 1 or logpi > 3):
        pairs.append((smiles[j], smiles[i]))
        flag = True
      if not duplicates and flag:
        flag = False
        break
  return pairs


def _pair_by_lipophilicity_plixer_alt(smiles: list[str], duplicates: bool = True):
  from helpers import predict_logp
  pairs = []
  length_i = len(smiles)
  flag = False
  for i in range(length_i):
    for j in range(i + 1, length_i):
      logpi = predict_logp(smiles[i])
      if logpi is None:
        continue
      logpj = predict_logp(smiles[j])
      if logpj is None:
        continue
      if logpi >= 1 and logpj < 1 or (logpi<=3 and logpj>3):
        pairs.append((smiles[i], smiles[j]))
        flag = True
      elif logpj >= 1 and logpi < 1 or (logpj<=3 and logpi>3):
        pairs.append((smiles[j], smiles[i]))
        flag = True
      if not duplicates and flag:
        flag = False
        break
  return pairs


def get_generate_steering_vector_and_eval_function_plixer(type_of_experiment, field=False):
  if type_of_experiment == "size":
    return get_num_atoms, (plixer_generate_steering_vector_size_field if field else plixer_generate_steering_vector_size)
  elif type_of_experiment == "logs":
    return evaluate_solubility, (plixer_generate_steering_vector_solubility_field if field else plixer_generate_steering_vector_solubility)
  elif type_of_experiment == "logp":
    return predict_logp, (plixer_generate_steering_vector_lipophilicity_field if field else plixer_generate_steering_vector_lipophilicity)
  else:
    raise ValueError(f"Unknown type of experiment: {type_of_experiment}")


def _train_vec_from_pairs_plixer(pairs, layers, token_index, field=False, dtype: torch.dtype = torch.float32):
  # Build full VisionEncoderDecoderModel and context pixel_values
  vox2smiles_module, ved_model, tok, pixel_values = _prepare_plixer_context_for_training(dtype=dtype)
  pos_acts, neg_acts = _plixer_extract_activations(
      vox2smiles_module,
      ved_model,
      tok,
      pairs,
      pixel_values=pixel_values,
      layers=[layers],
      read_token_index=token_index,
      show_progress=True,
  )
  if not field:
    layer_vecs = aggregate_activations(pos_acts, neg_acts)
    return SteeringVector(layer_vecs, "decoder_block")
  else:
    field_activations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for layer_num in pos_acts.keys():
      pos_cat = torch.cat(pos_acts[layer_num], dim=0)
      neg_cat = torch.cat(neg_acts[layer_num], dim=0)
      assert pos_cat.shape == neg_cat.shape
      field_activations[layer_num] = [(n, p - n) for p, n in zip(pos_cat, neg_cat)]
    return SteeringVectorField(layer_activations=field_activations, layer_type="decoder_block")


def plixer_generate_steering_vector_size(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  smiles, _, _ = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
  )
  pairs = _pair_by_size_plixer(smiles, duplicates=duplicates)
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=False, dtype=dtype)
  return vec, len(pairs), pairs


def plixer_generate_steering_vector_size_field(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  smiles, _, _ = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
  )
  pairs = _pair_by_size_plixer(smiles, duplicates=duplicates)
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=True, dtype=dtype)
  return vec, len(pairs), pairs


def plixer_generate_steering_vector_solubility(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  smiles, _, _ = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
  )
  pairs = _pair_by_solubility_plixer(smiles, duplicates=duplicates)
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=False, dtype=dtype)
  return vec, len(pairs), pairs


def plixer_generate_steering_vector_solubility_field(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  smiles, _, _ = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
  )
  pairs = _pair_by_solubility_plixer(smiles, duplicates=duplicates)
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=True, dtype=dtype)
  return vec, len(pairs), pairs


def plixer_generate_steering_vector_lipophilicity(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  smiles, _, _ = plixer_generate_smiles(
      pdb_file,
      ligand_file=ligand_file,
      center=center,
      num_samples=num_to_generate,
      temperature=temperature,
      vox2smiles_ckpt_path=vox2smiles_ckpt_path,
      poc2mol_ckpt_path=poc2mol_ckpt_path,
      seed=seed,
      dtype=dtype,
  )

  pairs = _pair_by_lipophilicity_plixer(smiles, duplicates=duplicates)
  if len(pairs) == 0:
    pairs = _pair_by_lipophilicity_plixer_alt(smiles, duplicates=duplicates)
  if len(pairs) == 0:
    return None, None, smiles
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=False, dtype=dtype)
  return vec, len(pairs), pairs


def plixer_generate_steering_vector_lipophilicity_field(
    pdb_file,
    *,
    ligand_file=None,
    center=None,
    num_to_generate=50,
    token_index=-1,
    layer=11,
    temperature: float = 1.0,
    seed: int = 42,
    vox2smiles_ckpt_path=None,
    poc2mol_ckpt_path=None,
    dtype: torch.dtype = torch.float32,
    duplicates=True,
):
  while True:
    smiles, _, _ = plixer_generate_smiles(
        pdb_file,
        ligand_file=ligand_file,
        center=center,
        num_samples=num_to_generate,
        temperature=temperature,
        vox2smiles_ckpt_path=vox2smiles_ckpt_path,
        poc2mol_ckpt_path=poc2mol_ckpt_path,
        seed=seed,
        dtype=dtype,
    )
    pairs = _pair_by_lipophilicity_plixer(smiles, duplicates=duplicates)
    if len(pairs) > 0:
      break
  vec = _train_vec_from_pairs_plixer(pairs, layer, token_index, field=True, dtype=dtype)
  return vec, len(pairs), pairs


# --------- Plixer training-time helpers (condition decoder on voxels) ---------

def _prepare_plixer_context_for_training(dtype: torch.dtype):
  """Load Plixer and compute a representative pixel_values for decoder conditioning.

  We reuse the default PDB and ligand center from the inference script to derive
  the encoder context once for steering vector extraction.
  """
  # Load model and config
  model_plixer, config = load_plixer_model(dtype=dtype)
  poc2mol_config = _plixer_build_poc2mol_config(config, dtype)

  # Use the default paths from generate_smiles_from_pdb.py to build context
  default_pdb = os.path.join("models", "plixer", "data", "5sry.pdb")
  default_ligand = os.path.join("models", "plixer", "data", "5sry_C_RIW.mol2")
  protein_voxel = _plixer_voxelize_protein(default_pdb, default_ligand, None, poc2mol_config)
  with torch.no_grad():
    output = model_plixer(protein_voxel, sample_smiles=False)
  pixel_values = output['predicted_ligand_voxels']  # shape (B,C,D,H,W)

  # Decoder model and tokenizer
  vox2smiles_module = model_plixer.vox2smiles_model
  ved_model = vox2smiles_module.model
  from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer  # type: ignore
  tok = build_smiles_tokenizer()
  return vox2smiles_module, ved_model, tok, pixel_values


def _plixer_extract_activations(
    vox2smiles_module: torch.nn.Module,
    ved_model: torch.nn.Module,
    tokenizer,
    training_samples: Sequence[tuple[str, str]],
    *,
    pixel_values: torch.Tensor,
    layers: list[int] | None = None,
    read_token_index: int = -1,
    show_progress: bool = False,
    batch_size: int = 1,
):
  """Extract decoder activations for Plixer, conditioning on encoder pixel_values.

  Returns (pos_acts_by_layer, neg_acts_by_layer) as dict[layer_num] -> list[tensor].
  Each tensor is (B, hidden_dim) at the specified token index.
  """
  fix_pad_token(tokenizer)
  pos_by_layer: dict[int, list[torch.Tensor]] = {}
  neg_by_layer: dict[int, list[torch.Tensor]] = {}
  device = next(ved_model.parameters()).device

  for raw_batch in batchify(training_samples, batch_size=batch_size, show_progress=show_progress, tqdm_desc="Extracting activations (Plixer)"):
    batch: list[tuple[str, str]] = list(raw_batch)
    pos_prompts = [p for p, _ in batch]
    neg_prompts = [n for _, n in batch]

    # Tokenize
    def add_bos_eos(smiles_list):
      processed = []
      for s in smiles_list:
        sm = s
        if not sm.startswith(tokenizer.bos_token):
          sm = tokenizer.bos_token + sm
        if not sm.endswith(tokenizer.eos_token):
          sm = sm + tokenizer.eos_token
        processed.append(sm)
      return processed

    pos_tok = tokenizer(add_bos_eos(pos_prompts), return_tensors="pt", padding=True)
    neg_tok = tokenizer(add_bos_eos(neg_prompts), return_tensors="pt", padding=True)

    # Build masked labels for teacher forcing
    # IMPORTANT: pass unmasked labels to VoxToSmilesModel, it handles masking internally
    pos_labels = pos_tok["input_ids"].to(device)
    neg_labels = neg_tok["input_ids"].to(device)

    # Tile pixel_values to batch size
    B = pos_labels.size(0)
    pv = pixel_values.to(device)
    if pv.size(0) != B:
      pv = pv.repeat(B, 1, 1, 1, 1)

    with record_activations(ved_model, layer_type="decoder_block", layer_nums=layers) as record:
      # Call through VoxToSmilesModel to mirror training/inference code path
      outputs = vox2smiles_module(pv, labels=pos_labels)
    pos_act = _select_token_activations(record, pos_tok["attention_mask"], read_token_index)
    for layer_num, act in pos_act.items():
      pos_by_layer.setdefault(layer_num, []).append(act)

    # Negative
    Bn = neg_labels.size(0)
    pv_n = pixel_values.to(device)
    if pv_n.size(0) != Bn:
      pv_n = pv_n.repeat(Bn, 1, 1, 1, 1)
    with record_activations(ved_model, layer_type="decoder_block", layer_nums=layers) as record:
      vox2smiles_module(pv_n, labels=neg_labels)
    neg_act = _select_token_activations(record, neg_tok["attention_mask"], read_token_index)
    for layer_num, act in neg_act.items():
      neg_by_layer.setdefault(layer_num, []).append(act)

  return pos_by_layer, neg_by_layer


def _select_token_activations(record: dict[int, list[torch.Tensor]], attention_mask: torch.Tensor, read_token_index: int) -> dict[int, torch.Tensor]:
  adjusted_indices = adjust_read_indices_for_padding(torch.tensor([read_token_index] * attention_mask.size(0)), attention_mask)
  batch_indices = torch.arange(attention_mask.size(0))
  results: dict[int, torch.Tensor] = {}
  for layer_num, acts in record.items():
    if len(acts) == 0:
      continue
    activation = acts[-1]  # last forward
    results[layer_num] = activation[batch_indices.to(activation.device), adjusted_indices.to(activation.device)].detach()
  return results



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