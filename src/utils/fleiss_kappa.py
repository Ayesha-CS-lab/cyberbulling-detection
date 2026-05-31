"""
Fleiss' Kappa calculation for inter-annotator agreement.
Uses the annotator columns from the Roman annotated data.
"""
import pandas as pd
import numpy as np


def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for inter-annotator agreement.

    Args:
        ratings_matrix: numpy array of shape (N, k) where
            N = number of items, k = number of categories.
            Each cell contains the count of annotators who assigned
            that category to that item.

    Returns:
        Fleiss' Kappa value
    """
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]  # number of raters per item

    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (N * n)

    # Extent of agreement for each item
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n) / (n * (n - 1))

    # Mean of P_i
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    # Fleiss' Kappa
    if P_e == 1:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def interpret_kappa(kappa):
    """Interpret Fleiss' Kappa value."""
    if kappa < 0:
        return "Poor agreement (less than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def calculate_from_dataset(excel_path='data/Roman annotated data.xlsx'):
    """
    Calculate Fleiss' Kappa from the annotated dataset.
    Expects columns: 'Annotator 1', 'Annotator 2', 'Annotator 3'
    with binary labels (0 or 1).
    """
    print(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)

    # Extract annotator columns
    a1 = df['Annotator 1'].fillna(0).astype(int)
    a2 = df['Annotator 2'].fillna(0).astype(int)
    a3 = df['Annotator 3'].fillna(0).astype(int)

    # Build ratings matrix: shape (N, 2) for binary classification
    # Column 0 = count of "not bullying" votes, Column 1 = count of "bullying" votes
    n_items = len(df)
    ratings_matrix = np.zeros((n_items, 2), dtype=int)

    for i in range(n_items):
        votes = [a1.iloc[i], a2.iloc[i], a3.iloc[i]]
        bullying_count = sum(votes)
        non_bullying_count = 3 - bullying_count
        ratings_matrix[i, 0] = non_bullying_count
        ratings_matrix[i, 1] = bullying_count

    kappa = fleiss_kappa(ratings_matrix)

    print("=" * 50)
    print("INTER-ANNOTATOR AGREEMENT (Fleiss' Kappa)")
    print("=" * 50)
    print(f"Number of items:      {n_items}")
    print(f"Number of annotators: 3")
    print(f"Number of categories: 2 (Bullying / Non-Bullying)")
    print(f"Fleiss' Kappa:        {kappa:.4f}")
    print(f"Interpretation:       {interpret_kappa(kappa)}")
    print("=" * 50)

    # Distribution
    total_votes = a1 + a2 + a3
    print(f"\nVote distribution:")
    print(f"  Unanimous agreement (all 3 agree): {((total_votes == 0) | (total_votes == 3)).sum()}")
    print(f"  Majority agreement (2 out of 3):   {((total_votes == 1) | (total_votes == 2)).sum()}")

    return kappa


if __name__ == '__main__':
    calculate_from_dataset()
