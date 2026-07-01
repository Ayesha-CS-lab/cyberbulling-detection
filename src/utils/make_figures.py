"""
Generate all thesis / presentation figures from the confirmed results.

Outputs high-resolution PNGs to docs/figures/ :
  - confusion matrices (Stage 1 m-BERT, MuRIL; Stage 2; before/after collapse)
  - Stage 1 model comparison bar chart (SVM, BiLSTM, m-BERT, MuRIL)
  - class-distribution charts (messages, pairs)
  - inter-annotator agreement breakdown
  - Stage 2 cross-validation stability

Numbers are taken from the saved evaluation reports (reproducible, seed=42).
Run:  python -m src.utils.make_figures
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DPI = 200


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.relpath(path))


def confusion(cm, classes, title, name, cmap='Blues'):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title)
    _save(fig, name)


def main():
    # ---- Stage 1 confusion matrices (TN, FP, FN, TP) ----
    confusion([[8094, 926], [727, 4101]],
              ['Non-Aggressive', 'Aggressive'],
              'Stage 1 — m-BERT Aggression (test)', 'cm_stage1_mbert.png')
    confusion([[7915, 1105], [536, 4292]],
              ['Non-Aggressive', 'Aggressive'],
              'Stage 1 — MuRIL Aggression (test)', 'cm_stage1_muril.png')

    # ---- Stage 2 confusion matrix ----
    confusion([[1556, 149], [2, 196]],
              ['Not CB', 'Cyberbullying'],
              'Stage 2 — Cyberbullying (test)', 'cm_stage2.png', cmap='Greens')

    # ---- Stage 3 image/multimodal confusion ----
    confusion([[201, 342], [297, 559]],
              ['Not Bullying', 'Bullying'],
              'Stage 3 — Image+Text Multimodal (test)', 'cm_stage3_image.png', cmap='Purples')

    # ---- Before / after collapse ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title, cmap in [
        (axes[0], [[0, 84], [0, 216]], 'BEFORE: collapsed model\n(predicts all positive)', 'Reds'),
        (axes[1], [[7915, 1105], [536, 4292]], 'AFTER: corrected model\n(MuRIL Stage 1)', 'Greens')]:
        sns.heatmap(np.array(cm), annot=True, fmt='d', cmap=cmap, cbar=False,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=ax,
                    annot_kws={'size': 13})
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    fig.suptitle('Majority-class collapse vs corrected model', fontweight='bold')
    _save(fig, 'cm_before_after.png')

    # ---- Stage 1 model comparison bar chart ----
    models = ['SVM+TFIDF', 'BiLSTM', 'm-BERT', 'MuRIL']
    metrics = {
        'Accuracy':  [0.847, 0.858, 0.881, 0.882],
        'Precision': [0.764, 0.780, 0.816, 0.795],
        'Recall':    [0.814, 0.824, 0.849, 0.889],
        'F1':        [0.788, 0.802, 0.832, 0.840],
    }
    x = np.arange(len(models)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (mname, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=mname)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0.6, 0.95); ax.set_ylabel('Score')
    ax.set_title('Stage 1 — Aggression detection: model comparison')
    ax.legend(loc='lower right', ncol=4)
    _save(fig, 'stage1_comparison.png')

    # ---- Class distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['Non-Aggressive', 'Aggressive'], [60919, 31389],
                color=['#4c72b0', '#c44e52'])
    axes[0].set_title('Messages (Stage 1): 92,308 total')
    for i, v in enumerate([60919, 31389]):
        axes[0].text(i, v + 800, f'{v:,}', ha='center')
    axes[1].bar(['Not CB', 'Cyberbullying'], [8519, 992],
                color=['#4c72b0', '#c44e52'])
    axes[1].set_title('User-pairs (Stage 2): 9,511 total')
    for i, v in enumerate([8519, 992]):
        axes[1].text(i, v + 100, f'{v:,}', ha='center')
    fig.suptitle('Class distribution (imbalance)', fontweight='bold')
    _save(fig, 'class_distribution.png')

    # ---- Stage 2 cross-validation stability ----
    folds = [1, 2, 3, 4, 5]
    acc = [0.9149, 0.9211, 0.9238, 0.9190, 0.9274]
    f1 = [0.7022, 0.7243, 0.7259, 0.7127, 0.7366]
    rec = [0.9598, 0.9899, 0.9697, 0.9646, 0.9747]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(folds, acc, 'o-', label='Accuracy')
    ax.plot(folds, rec, 's-', label='Recall')
    ax.plot(folds, f1, '^-', label='F1')
    ax.set_xticks(folds); ax.set_xlabel('Fold'); ax.set_ylabel('Score')
    ax.set_ylim(0.65, 1.0)
    ax.set_title('Stage 2 — 5-fold cross-validation stability')
    ax.legend(loc='center right')
    _save(fig, 'stage2_cv.png')

    # ---- Inter-annotator agreement ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['Unanimous\n(3/3)', 'Majority\n(2/3)', 'Split'], [1337, 662, 0],
                color=['#55a868', '#dd8452', '#c44e52'])
    axes[0].set_title('Annotator agreement (1,999 items)')
    for i, v in enumerate([1337, 662, 0]):
        axes[0].text(i, v + 15, str(v), ha='center')
    axes[1].bar(['3-category', '0 vs {1,2}', '{0,1} vs 2'], [0.666, 0.601, 0.723],
                color='#4c72b0')
    axes[1].axhline(0.6, ls='--', c='grey', lw=1)
    axes[1].set_ylim(0, 0.85); axes[1].set_title("Fleiss' κ (>0.6 = substantial)")
    for i, v in enumerate([0.666, 0.601, 0.723]):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    _save(fig, 'fleiss_agreement.png')

    print(f'\nAll figures written to {FIG_DIR}')


if __name__ == '__main__':
    main()
