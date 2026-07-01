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


def _build_ratings_matrix(votes_df, categories):
    """Build an (N, k) Fleiss matrix: cell = how many raters chose that category."""
    cat_index = {c: j for j, c in enumerate(categories)}
    n_items = len(votes_df)
    matrix = np.zeros((n_items, len(categories)), dtype=int)
    for i, (_, row) in enumerate(votes_df.iterrows()):
        for v in row:
            matrix[i, cat_index[v]] += 1
    return matrix


def calculate_from_dataset(excel_path='data/Roman annotated data.xlsx',
                           binary_map=None):
    """
    Calculate Fleiss' Kappa from the annotated dataset.

    The annotators used THREE labels (0, 1, 2). By default this computes the
    assumption-free 3-category Kappa. If `binary_map` is given (a dict mapping
    each raw label to 0/1), a collapsed binary Kappa is computed instead so the
    result can be reported on a 'bullying / non-bullying' scale once the meaning
    of the labels is confirmed.
    """
    print(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)
    cols = ['Annotator 1', 'Annotator 2', 'Annotator 3']

    votes = df[cols].copy()
    before = len(votes)
    votes = votes.dropna()                      # drop rows with a missing rating
    votes = votes.astype(int)
    dropped = before - len(votes)

    if binary_map is not None:
        votes = votes.applymap(lambda v: binary_map.get(v, v))

    categories = sorted(set(votes.values.ravel().tolist()))
    matrix = _build_ratings_matrix(votes, categories)
    kappa = fleiss_kappa(matrix)

    # Agreement breakdown (per item, how many raters chose the majority label)
    def n_agree(row):
        return max(np.bincount(row.values))
    agree_counts = votes.apply(n_agree, axis=1)

    print("=" * 50)
    print("INTER-ANNOTATOR AGREEMENT (Fleiss' Kappa)")
    print("=" * 50)
    print(f"Number of items:      {len(votes)} (dropped {dropped} with missing ratings)")
    print(f"Number of annotators: 3")
    print(f"Categories:           {categories}"
          f"{'  [binary-mapped]' if binary_map else ''}")
    print(f"Fleiss' Kappa:        {kappa:.4f}")
    print(f"Interpretation:       {interpret_kappa(kappa)}")
    print(f"\nVote distribution per category (raters x items):")
    for c, count in zip(categories, matrix.sum(axis=0)):
        print(f"  label {c}: {count}")
    print(f"\nAgreement breakdown:")
    print(f"  Unanimous (all 3 agree): {(agree_counts == 3).sum()}")
    print(f"  Majority  (2 of 3 agree): {(agree_counts == 2).sum()}")
    print(f"  No agreement (all differ): {(agree_counts == 1).sum()}")
    print("=" * 50)
    return kappa


if __name__ == '__main__':
    calculate_from_dataset()
