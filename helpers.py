from rdkit import Chem
import soltrannet as stn
import numpy as np
from rdkit.Chem import Crippen
      

def get_training_pairs_in_generated_smiles_and_rate(training_pairs, generated_smiles, positive=True):
    """
    Calculates the rate of generated SMILES strings that are present in the training pairs.

    Args:
        training_pairs (list of tuple): List of training pairs, where each pair contains two strings.
        generated_smiles (list of str): List of generated SMILES strings to be checked.
        positive (bool, optional): If True, use the first element of each pair in training_pairs; 
            otherwise, use the second element. Defaults to True.

    Returns:
        float: The fraction of generated SMILES that are found in the selected elements of training_pairs.
    """
    positive_set = set()

    for pair in training_pairs:
        if positive:
          positive_string = pair[0].split("<L>")[1].split("<|endoftext|>")[0]
        else:
          positive_string = pair[1].split("<L>")[1].split("<|endoftext|>")[0]

        positive_set.add(positive_string)

    dupliacates = 0

    for smile in generated_smiles:
        if smile in positive_set:
            dupliacates += 1

    rate = dupliacates / len(generated_smiles)
    return rate


def get_num_atoms(smiles, array = None):
    """
    Returns the number of atoms in a molecule represented by a SMILES string.

    If an array is provided, appends the number of atoms to the array.
    Otherwise, returns the number of atoms as an integer.

    Args:
        smiles (str): The SMILES string representing the molecule.
        array (list, optional): A list to which the number of atoms will be appended. Defaults to None.

    Returns:
        int or None: The number of atoms in the molecule if array is None, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if array is not None:
      # print(array)
      array.append(mol.GetNumAtoms())
    else:
      return mol.GetNumAtoms()
    

def generate_pairs_size(uniprot_id, smiles, duplicates=True):
    """
    Generates pairs of SMILES strings for a given UniProt ID based on the number of atoms in each molecule.
    For each pair of SMILES strings, the function checks if the number of atoms in one string is within a specified range (defined by `ranges`) of the other. If so, it creates a tuple containing two prompts (shorter and longer), each consisting of a formatted string with the UniProt ID and the respective SMILES string.
    Args:
        uniprot_id (str): The UniProt identifier to be included in the prompt.
        smiles (list of str): A list of SMILES strings representing molecules.
        duplicates (bool, optional): If False, ensures that each SMILES string is used in at most one pair. Defaults to True.
    Returns:
        list of tuple: A list of tuples, each containing two formatted prompt strings (shorter, longer) based on the atom count comparison.
    Notes:
        - The function relies on an external function `get_num_atoms(smiles)` to determine the number of atoms in each SMILES string.
        - If a SMILES string is invalid or `get_num_atoms` fails, it is skipped with an error message.
        - The `ranges` variable defines the lower and upper bounds (as fractions) for comparing atom counts between pairs.
    """
  
    pairss = []
    prompt = f"<|startoftext|><P>{uniprot_id}<L>"
    ranges = [0.8, 0.6]

    with_atom_num = []

    for string in smiles:
        if string:
            try:
                with_atom_num.append((string, get_num_atoms(string)))
            except:
                print(f"Error in getting num atoms for {string}")

    with_atom_num.sort(key=lambda x: x[1])

    length_i = len(with_atom_num)
    flag = False

    for i in range(length_i):
        for j in range(i+1, length_i):
            first = with_atom_num[i]
            second = with_atom_num[j]
            if first[1] <= ranges[0] * second[1] and first[1] >= ranges[1] * second[1]:
                shorter = prompt + first[0]
                longer = prompt + second[0]
                pairss.append((shorter, longer))
                flag = True
            elif second[1] <= ranges[0] * first[1] and second[1] >= ranges[1] * first[1]:
                shorter = prompt + second[0]
                longer = prompt + first[0]
                pairss.append((shorter, longer))
                flag = True
            if not duplicates and flag:
                with_atom_num.remove(first)
                with_atom_num.remove(second)
                length_i -= 2
                flag=False
                break

    return pairss


def get_solubility_predictions(smiles):
    """
    Predicts solubility values for a list of SMILES strings.
    This function takes a list of SMILES strings, removes any invalid SMILES,
    predicts their solubility using a pre-trained model (`stn`), and returns
    the predictions sorted by the predicted solubility value.
    Args:
        smiles (list of str): List of SMILES strings representing molecules.
    Returns:
        numpy.ndarray: A 2D array where each row contains a SMILES string and its
        predicted solubility value, sorted in ascending order of solubility.
    Notes:
        - Invalid SMILES strings are removed from the input list.
        - Assumes `Chem`, `stn`, and `np` are already imported and available in the scope.
    """

    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        if mol is None:
            smiles.remove(i)

    predictions = list(stn.predict(smiles))

    predictions = np.array(predictions)
    predictions = predictions[:,0:2][:,::-1]

    predictions = predictions.astype(object)
    predictions[:,1] = predictions[:,1].astype(float)
    predictions = predictions[predictions[:,1].argsort()]

    return predictions


def create_solubility_pairs(uniprot_id, predictions, duplicates=True):
    """
    Generates ordered pairs of sequence prompts based on solubility predictions for a given UniProt ID.
    Each pair consists of two prompts: one for the sequence with higher predicted solubility ("bigger") and one for the sequence with lower predicted solubility ("smaller").
    The function compares all unique pairs of predictions and constructs prompt strings in the format "<|startoftext|><P>{uniprot_id}<L>{sequence}".
    Args:
        uniprot_id (str): The UniProt identifier to include in each prompt.
        predictions (list of tuples): A list where each element is a tuple (sequence, solubility_score).
        duplicates (bool, optional): If False, removes compared sequences from further pairing to avoid duplicate pairs. Defaults to True.
    Returns:
        list of tuples: A list of (bigger, smaller) prompt pairs, where "bigger" has a higher solubility score than "smaller".
    """
    pairs_solubility = []
    prompt = f"<|startoftext|><P>{uniprot_id}<L>"
    
    flag = False

    length_i = len(predictions)

    for i in range(length_i):
        for j in range(i+1, length_i):
            if predictions[i][1] < predictions[j][1]:
                smaller = prompt + predictions[i][0]
                bigger = prompt + predictions[j][0]
                pairs_solubility.append((bigger, smaller))
                flag = True
            elif predictions[j][1] < predictions[i][1]:
                smaller = prompt + predictions[j][0]
                bigger = prompt + predictions[i][0]
                pairs_solubility.append((bigger, smaller))
                flag = True
            if not duplicates and flag:
                predictions = np.delete(predictions, i, axis=0)
                predictions = np.delete(predictions, j-1, axis=0)
                length_i -= 2
                flag=False
                break
    return pairs_solubility


def evaluate_solubility(smile, array):
    """
    Evaluates the solubility of a molecule represented by a SMILES string and appends the prediction to the provided array.

    Args:
        smile (str): The SMILES string representing the molecule to be evaluated.
        array (list): The list to which the solubility prediction will be appended.

    Returns:
        None: The function modifies the input array in place by extending it with the solubility prediction(s).
    """
    array.extend(get_solubility_predictions([smile])[:,1])


def predict_logp(smiles, array=None):
    """
    Predicts the logP (octanol-water partition coefficient) of a molecule from its SMILES representation.

    Parameters:
        smiles (str): The SMILES string representing the molecule.
        array (list, optional): If provided, the predicted logP value will be appended to this list.
                                If not provided, the function returns the logP value.

    Returns:
        float or None: The predicted logP value if `array` is not provided; otherwise, None.

    Raises:
        Exception: If the SMILES string cannot be converted to a molecule and `array` is provided.

    Notes:
        Requires RDKit's Chem and Crippen modules to be imported.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        logp = Crippen.MolLogP(mol)
        if array is not None:
            array.append(logp)
        else:
            return logp
    elif mol is None and array is not None:
        raise Exception("Mol is None")
    

def create_lipo_pairs(uniprot_id, smiles, duplicates=True):
    """
    Generates pairs of SMILES strings based on their predicted logP values for a given UniProt ID.

    For each unique pair of SMILES strings, predicts their logP values using `predict_logp`. 
    Pairs are created such that one molecule has a logP in the range [1, 3] and the other outside this range.
    Each pair is formatted with a prompt containing the UniProt ID and the SMILES string.

    Args:
        uniprot_id (str): The UniProt identifier to include in the prompt.
        smiles (list of str): List of SMILES strings representing molecules.
        duplicates (bool, optional): If False, removes paired SMILES to avoid duplicate pairs. Defaults to True.

    Returns:
        list of tuple: List of tuples, each containing two formatted prompt strings for the paired SMILES.
    """
    prompt = f"<|startoftext|><P>{uniprot_id}<L>"
    flag = False

    pairs = []
    length_i = len(smiles)
    for i in range(length_i):
        for j in range(i+1, length_i):

            logpi = predict_logp(smiles[i])
            if logpi is None:
                continue

            logpj = predict_logp(smiles[j])
            if logpj is None:
                continue

            if logpi >= 1 and logpi <= 3 and (logpj < 1 or logpj > 3):
                pairs.append((prompt + smiles[i], prompt + smiles[j]))
                flag = True

            elif logpj >= 1 and logpj <= 3 and (logpi < 1 or logpi > 3):
                pairs.append((prompt + smiles[j], prompt + smiles[i]))
                flag = True

            if not duplicates and flag:
                smiles.remove(smiles[j])
                length_i -= 1
                flag=False
                break

    return pairs

