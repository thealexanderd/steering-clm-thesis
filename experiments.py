from math import exp
import os
import gc
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from helpers import get_training_pairs_in_generated_smiles_and_rate
from model_and_generation import evaluate_group
from model_and_generation import get_generate_steering_vector_and_eval_function
import torch 


def perform_experiments(
    protein_sequences,
    new_list_of_uniprot_ids,
    filename_prefix="data/figures/",
    duplicates=None,
    num_to_generate_pairs=50,
    num_to_generate=250,
    token_index=-1,
    layer_index=11,
    tau=1.0,
    multiplier=1.0,
    experiment="size",
    field=False
):
    
    """
    Runs a series of molecular generation experiments for a list of protein sequences, evaluating the effect of steering vectors on generated molecules.
    For each protein, the function:
      - Generates steering vectors and training pairs.
      - Evaluates molecule generation under three conditions: no steering, positive steering, and negative steering.
      - Computes statistics comparing the generated molecules across conditions.
      - Saves generated SMILES and statistics to disk.
    Args:
        protein_sequences (list): List of protein sequences to process.
        new_list_of_uniprot_ids (list): List of UniProt IDs corresponding to the protein sequences.
        filename_prefix (str, optional): Directory prefix for saving results and figures. Defaults to "data/figures/".
        duplicates (optional): Optional parameter to control duplicate handling in pair generation.
        num_to_generate_pairs (int, optional): Number of training pairs to generate for steering vector calculation. Defaults to 50.
        num_to_generate (int, optional): Number of molecules to generate per condition. Defaults to 250.
        token_index (int, optional): Token index for steering vector extraction. Defaults to -1.
        layer_index (int, optional): Layer index for steering vector extraction. Defaults to 11.
        tau (float, optional): Softmax temperature for the distance weights in the field-based steering. Defaults to 1.0.
        multiplier (float, optional): Multiplier for the steering vector. Defaults to 1.0.
        experiment (str, optional): Type of experiment to run (e.g., "size"). Defaults to "size".
        field (bool, optional): Whether to use field-based steering. Defaults to False.
    Returns:
        pandas.DataFrame: DataFrame containing statistics and results for each protein sequence.
    """
    
    os.makedirs(filename_prefix, exist_ok=True)
    os.makedirs(f"{filename_prefix}smiles/", exist_ok=True)

    columns = [
        "uniprot_id", "no_steering", "steering", "neg_steering",
        "no_steering_total_generated", "no_steering_total_returned", "no_steering_invalid_generated", "no_steering_uniqiue",
        "steering_total_generated", "steering_total_returned", "steering_invalid_generated", "steering_unique",
        "negative_steering_total_generated", "negative_steering_total_returned", "negative_steering_invalid_generated", "negative_steering_unique",
        "p_value_no_pos", "p_value_no_neg", "ks_no_pos", "ks_no_neg", "num_training_pairs", "rate_of_training_positive", "rate_of_training_negative"
    ]
    data = pd.DataFrame(columns=columns)

    for i, protein in enumerate(protein_sequences):
        print(f"\n--- Processing protein {i} ---")
        uid = new_list_of_uniprot_ids[i]

        eval_function, generate_steering_vector_func = get_generate_steering_vector_and_eval_function(experiment, field=field)
        vec, num_pairs, pairs = generate_steering_vector_func(protein, num_to_generate=num_to_generate_pairs, duplicates=duplicates, token_index=token_index, layer=layer_index)

        with open(f"{filename_prefix}smiles/{uid}_pairs.pkl", "wb") as f:
            pickle.dump(pairs, f)

        # Evaluate each condition
        groups = [
            evaluate_group("no_steering", vec, multiplier=0, protein=protein, num_to_generate=num_to_generate, eval_fn=eval_function, tau=tau),
            evaluate_group("steering", vec, multiplier=multiplier, protein=protein, num_to_generate=num_to_generate, eval_fn=eval_function, tau=tau),
            evaluate_group("negative_steering", vec, multiplier=-multiplier, protein=protein, num_to_generate=num_to_generate, eval_fn=eval_function, tau=tau),
        ]

        if None in groups:
          print(f"Protein {uid} cannot be steered")
          continue
        group_map = {g["label"]: g for g in groups}

        # Save smiles
        for g in groups:
            with open(f"{filename_prefix}smiles/{uid}_{g['label']}_smiles.pkl", "wb") as f:
                pickle.dump(g["smiles"], f)

        # Compute statistics
        pval_no_pos = stats.ttest_ind(group_map["no_steering"]["scores"], group_map["steering"]["scores"], equal_var=False).pvalue
        pval_no_neg = stats.ttest_ind(group_map["no_steering"]["scores"], group_map["negative_steering"]["scores"], equal_var=False).pvalue

        ks_no_pos = stats.ks_2samp(group_map["no_steering"]["scores"], group_map["steering"]["scores"]).pvalue
        ks_no_neg = stats.ks_2samp(group_map["no_steering"]["scores"], group_map["negative_steering"]["scores"]).pvalue

        rate_pos = get_training_pairs_in_generated_smiles_and_rate(pairs, group_map["steering"]["smiles"], positive=True)
        rate_neg = get_training_pairs_in_generated_smiles_and_rate(pairs, group_map["negative_steering"]["smiles"], positive=False)

        new_row = {
            "uniprot_id": uid,
            "no_steering": np.mean(group_map["no_steering"]["scores"]),
            "steering": np.mean(group_map["steering"]["scores"]),
            "neg_steering": np.mean(group_map["negative_steering"]["scores"]),
            "no_steering_total_generated": group_map["no_steering"]["total_generated"],
            "no_steering_total_returned": len(group_map["no_steering"]["smiles"]),
            "no_steering_invalid_generated": group_map["no_steering"]["invalid_generated"],
            "no_steering_uniqiue": len(group_map["no_steering"]["set"]),
            "steering_total_generated": group_map["steering"]["total_generated"],
            "steering_total_returned": len(group_map["steering"]["smiles"]),
            "steering_invalid_generated": group_map["steering"]["invalid_generated"],
            "steering_unique": len(group_map["steering"]["set"]),
            "negative_steering_total_generated": group_map["negative_steering"]["total_generated"],
            "negative_steering_total_returned": len(group_map["negative_steering"]["smiles"]),
            "negative_steering_invalid_generated": group_map["negative_steering"]["invalid_generated"],
            "negative_steering_unique": len(group_map["negative_steering"]["set"]),
            "p_value_no_pos": pval_no_pos,
            "p_value_no_neg": pval_no_neg,
            "ks_no_pos": ks_no_pos,
            "ks_no_neg": ks_no_neg,
            "rate_of_training_positive": rate_pos,
            "rate_of_training_negative": rate_neg,
            "num_training_pairs": num_pairs
        }

        data.loc[len(data)] = new_row
        data.to_csv(f"{filename_prefix}{uid}.csv", index=False)

        # IPython.display.clear_output()

    return data


def main():
    df = pd.read_json("hf://datasets/alimotahharynia/approved_drug_target/uniprotId_sequence_2024_11_01.json")

    uniprot_to_sequence = dict(zip(df["UniProt_id"], df["Sequence"]))

    new_list_of_uniprot_ids = ['P07900', 'P00734', 'Q14524', 'P19823', 'P78334', 'P14416', 'P08913', 'P03372', 'P35348', 'P08172', "P09622", "P36956", 'P20309', 'P04818', 'P25100', 'P18825', 'P23634', 'Q8N1C3', 'P78334', 'P03372', 'P48169',  'P10275']

    protein_sequences = []
    for uid in new_list_of_uniprot_ids:
        sequence = uniprot_to_sequence.get(uid)
        if sequence:
            protein_sequences.append(sequence)
        else:
            print(f"UniProt ID {uid} not found in the dataset.")
    
    results = perform_experiments(
        protein_sequences,
        new_list_of_uniprot_ids,
        filename_prefix="data/figures/",
        duplicates=True,
        num_to_generate_pairs=50,
        num_to_generate=250,
        experiment="logp",
        field=False,
    )
    
    print(results)

if __name__ == "__main__":
    main()