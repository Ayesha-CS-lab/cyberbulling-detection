"""
Generate architecture / methodology block diagrams as PNGs for the thesis.

Outputs to docs/figures/:
  - arch_overall.png  : two-stage methodology (matches proposal Figure 1)
  - arch_stage1.png   : Stage 1 aggression network (m-BERT/MuRIL)
  - arch_stage2.png   : Stage 2 cyberbullying dense network
  - arch_image.png    : multimodal image+text fusion branch

Run:  python -m src.utils.make_diagrams
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DPI = 200


def box(ax, cx, cy, w, h, text, fc='#dce6f5', ec='#33558b', fs=10, bold=False):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.08',
                       fc=fc, ec=ec, lw=1.5, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color='#444'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=14, lw=1.4, color=color, zorder=1))
    if text:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=8, color=color)


def _canvas(w=10, h=13):
    fig, ax = plt.subplots(figsize=(w * 0.8, h * 0.8))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    return fig, ax


def _save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def overall():
    fig, ax = _canvas(11, 14)
    cx = 5.5
    box(ax, cx, 13.2, 5.4, 0.9, 'Annotated dataset\n(users, messages, CB labels)', fc='#e8e8e8', bold=True)
    box(ax, cx, 11.7, 5.4, 0.9, 'For each user pair  (Sender → Receiver)', fc='#fdf0d5')
    box(ax, cx, 10.0, 6.2, 1.1,
        'STAGE 1 — Text Classification of all messages\nm-BERT / MuRIL  →  Aggressive / Non-Aggressive', fc='#dce6f5', bold=True)
    # branch row
    box(ax, 2.3, 8.0, 3.0, 0.95, 'Aggression %\n(aggressive / total)', fc='#d9ead3')
    box(ax, 5.5, 8.0, 2.6, 0.95, 'Repetition\n(frequency over time)', fc='#d9ead3')
    box(ax, 8.7, 8.0, 2.6, 0.95, 'Intent to harm\n+ Peerness + context', fc='#d9ead3')
    box(ax, cx, 6.1, 6.2, 1.1,
        'STAGE 2 — Cyberbullying decision\nDense neural network (MLP)', fc='#dce6f5', bold=True)
    box(ax, cx, 4.4, 4.6, 0.9, 'Cyberbullying label\n(Bullying / Not)', fc='#f4cccc', bold=True)

    arrow(ax, cx, 12.75, cx, 12.15)
    arrow(ax, cx, 11.25, cx, 10.55)
    for bx in (2.3, 5.5, 8.7):
        arrow(ax, cx, 9.45, bx, 8.5)
        arrow(ax, bx, 7.5, cx, 6.65)
    arrow(ax, cx, 5.55, cx, 4.85)
    # loop back
    ax.add_patch(FancyArrowPatch((cx - 2.3, 4.4), (1.0, 4.4), arrowstyle='-', lw=1.2, color='#888'))
    ax.add_patch(FancyArrowPatch((1.0, 4.4), (1.0, 11.7), arrowstyle='-', lw=1.2, color='#888'))
    ax.add_patch(FancyArrowPatch((1.0, 11.7), (cx - 2.7, 11.7), arrowstyle='-|>', mutation_scale=14, lw=1.2, color='#888'))
    ax.text(0.8, 8.0, 'next\nuser pair', fontsize=8, color='#888', rotation=90, va='center')
    ax.set_title('Two-stage cyberbullying detection methodology', fontsize=13, fontweight='bold')
    _save(fig, 'arch_overall.png')


def stage1():
    fig, ax = _canvas(5, 13)
    cx = 2.5
    steps = [
        ('Message text\n(English / Roman Urdu)', '#e8e8e8'),
        ('Preprocess + Tokenize\n(WordPiece, max len 128)', '#fdf0d5'),
        ('m-BERT / MuRIL encoder\n(12 layers, 768-dim)', '#dce6f5'),
        ('[CLS] pooled embedding\n(768-dim)', '#dce6f5'),
        ('Dropout (0.3)', '#dce6f5'),
        ('Linear  768 → 1', '#dce6f5'),
        ('Sigmoid → aggression probability', '#d9ead3'),
    ]
    ys = [12.2, 10.6, 9.0, 7.4, 5.9, 4.4, 2.8]
    for (t, c), y in zip(steps, ys):
        box(ax, cx, y, 4.2, 1.0, t, fc=c, bold=(c == '#d9ead3'))
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, cx, y1 - 0.5, cx, y2 + 0.5)
    ax.set_title('Stage 1 — Aggression classifier', fontsize=12, fontweight='bold')
    _save(fig, 'arch_stage1.png')


def stage2():
    fig, ax = _canvas(5, 13)
    cx = 2.5
    steps = [
        ('Pair feature vector (19)\naggression%, repetition, intent,\npeerness, age, grade, gender', '#e8e8e8'),
        ('Standardize\n(train mean / std)', '#fdf0d5'),
        ('Linear 19 → 32  +  ReLU  +  Dropout', '#dce6f5'),
        ('Linear 32 → 16  +  ReLU  +  Dropout', '#dce6f5'),
        ('Linear 16 → 1', '#dce6f5'),
        ('Sigmoid → cyberbullying probability', '#d9ead3'),
    ]
    ys = [12.0, 10.2, 8.5, 6.9, 5.3, 3.6]
    hs = [1.3, 1.0, 1.0, 1.0, 1.0, 1.0]
    for (t, c), y, h in zip(steps, ys, hs):
        box(ax, cx, y, 4.4, h, t, fc=c, bold=(c == '#d9ead3'))
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, cx, y1 - 0.55, cx, y2 + 0.55)
    ax.set_title('Stage 2 — Cyberbullying classifier', fontsize=12, fontweight='bold')
    _save(fig, 'arch_stage2.png')


def image_branch():
    fig, ax = _canvas(10, 11)
    box(ax, 2.5, 9.5, 3.6, 1.0, 'Meme image', fc='#e8e8e8', bold=True)
    box(ax, 7.5, 9.5, 3.6, 1.0, 'Meme text (OCR)', fc='#e8e8e8', bold=True)
    box(ax, 2.5, 7.8, 3.6, 1.0, 'ResNet50\n(image encoder, 2048-d)', fc='#dce6f5')
    box(ax, 7.5, 7.8, 3.6, 1.0, 'm-BERT / MuRIL\n(text encoder, 768-d)', fc='#dce6f5')
    box(ax, 2.5, 6.1, 3.6, 0.9, 'Project → 512', fc='#dce6f5')
    box(ax, 7.5, 6.1, 3.6, 0.9, 'Project → 512', fc='#dce6f5')
    box(ax, 5.0, 4.4, 5.2, 1.0, 'Attention-based fusion layer', fc='#fdf0d5', bold=True)
    box(ax, 5.0, 2.8, 5.2, 1.0, 'Classifier → Sigmoid', fc='#dce6f5')
    box(ax, 5.0, 1.3, 4.0, 0.9, 'Bullying / Not', fc='#f4cccc', bold=True)
    arrow(ax, 2.5, 9.0, 2.5, 8.3); arrow(ax, 7.5, 9.0, 7.5, 8.3)
    arrow(ax, 2.5, 7.3, 2.5, 6.55); arrow(ax, 7.5, 7.3, 7.5, 6.55)
    arrow(ax, 2.5, 5.65, 4.2, 4.9); arrow(ax, 7.5, 5.65, 5.8, 4.9)
    arrow(ax, 5.0, 3.9, 5.0, 3.3); arrow(ax, 5.0, 2.3, 5.0, 1.75)
    ax.set_title('Multimodal image + text fusion branch', fontsize=12, fontweight='bold')
    _save(fig, 'arch_image.png')


def data_views():
    fig, ax = _canvas(13, 11)
    # raw input tables (left)
    box(ax, 2.4, 9.2, 4.0, 1.1, 'Communication_Data\n92,520 messages (timestamped,\nper sender-receiver pair)', fc='#e8e8e8')
    box(ax, 2.4, 7.4, 4.0, 0.9, 'Roman Urdu corpus\n1,952 annotated messages', fc='#e8e8e8')
    box(ax, 2.4, 5.6, 4.0, 0.9, 'CB_Labels\n9,511 user pairs', fc='#e8e8e8')
    box(ax, 2.4, 3.8, 4.0, 1.0, 'users_data (100 users)\n+ peerness_values', fc='#e8e8e8')

    # converter (center)
    box(ax, 6.7, 6.4, 3.0, 1.4, 'prepare_dataset.py\n\nclean · de-duplicate\naggregate per pair\nderive features', fc='#fdf0d5', bold=True)

    # output tables (right)
    box(ax, 10.7, 8.2, 4.2, 1.5,
        'messages.csv  (Stage 1 view)\n[ message, label, language ]\n92,308 rows\n(90,356 EN + 1,952 RU)', fc='#dce6f5', bold=True)
    box(ax, 10.7, 4.0, 4.2, 1.8,
        'pairs.csv  (Stage 2 view)\n18 columns: aggression%,\nrepetition, intent, peerness,\nage / grade / gender\n9,511 rows', fc='#d9ead3', bold=True)

    for y in (9.2, 7.4, 5.6, 3.8):
        arrow(ax, 4.4, y, 5.2, 6.6)
    arrow(ax, 8.2, 6.9, 8.6, 8.0, 'messages')
    arrow(ax, 8.2, 5.9, 8.6, 4.3, 'pairs')
    ax.set_title('Construction of message-level and pair-level data views', fontsize=12.5, fontweight='bold')
    _save(fig, 'data_views.png')


def preprocess_pipeline():
    fig, ax = _canvas(6.5, 15)
    cx = 3.25
    steps = [
        ('Raw message text', '#e8e8e8', True),
        ('Remove URLs  (http://… , www.…)', '#dce6f5', False),
        ('Remove HTML tags  (<…>)', '#dce6f5', False),
        ('Remove user mentions  (@username)', '#dce6f5', False),
        ('Remove emojis / special chars → space', '#dce6f5', False),
        ('Normalise whitespace', '#dce6f5', False),
        ('Roman Urdu normalisation\nlowercase · aa→a · ee→i · oo→u', '#fdf0d5', False),
        ('Tokenise  (WordPiece / MuRIL, max 128)', '#d9ead3', False),
        ('Token IDs → m-BERT / MuRIL encoder', '#d9ead3', True),
    ]
    ys = [14.0, 12.6, 11.4, 10.2, 9.0, 7.8, 6.4, 4.9, 3.6]
    for (t, c, b), y in zip(steps, ys):
        h = 1.05 if '\n' in t else 0.8
        box(ax, cx, y, 5.2, h, t, fc=c, bold=b)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, cx, y1 - 0.45, cx, y2 + 0.45)
    # annotate the conditional step
    ax.text(6.1, 6.4, 'Roman Urdu\nonly', fontsize=8, color='#b07a00', va='center', ha='left')
    ax.text(cx, 13.35, 'e.g.  "@ali tumm bohooot buray ho!!  http://x.co"',
            fontsize=7.5, color='#666', ha='center', style='italic')
    ax.set_title('Text preprocessing & Roman Urdu normalisation pipeline',
                 fontsize=12, fontweight='bold')
    _save(fig, 'preprocess_pipeline.png')


def timeline_scoring():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 5.3))
    # (day position, aggressive?, text)
    msgs = [
        (0.6, False, ''), (0.95, True, 'tum bura ho'), (1.8, True, 'tujhe dekh lunga'),
        (2.4, False, ''), (2.9, True, 'bakwaas band kar'), (3.6, True, 'nikamma insaan'),
        (4.3, True, 'jaan se maar dunga'), (4.7, False, ''), (5.2, True, 'khatam kar dunga'),
    ]
    for d in range(1, 6):
        ax.axvspan(d - 0.5, d + 0.5, color='#eef2f7' if d % 2 else '#f8f8f8', zorder=0)
        ax.text(d, 0.35, f'Day {d}', ha='center', fontsize=9, color='#666')
    ax.axhline(1.0, color='#cccccc', lw=2, zorder=1)
    heights = [1.5, 1.85, 1.5, 1.85, 1.55, 1.85]
    hi = 0
    for x, agg, lab in msgs:
        if agg:
            ax.scatter(x, 1.0, s=200, marker='X', color='#c44e52',
                       edgecolors='black', linewidths=0.6, zorder=3)
            y = heights[hi % len(heights)]; hi += 1
            ax.annotate(f'"{lab}"', xy=(x, 1.06), xytext=(x, y), ha='center',
                        fontsize=8, color='#7a1f22',
                        arrowprops=dict(arrowstyle='-', color='#c44e52', lw=0.8))
        else:
            ax.scatter(x, 1.0, s=90, marker='o', color='#9fb0c0', zorder=2)
    ax.text(0.5, 2.12, 'Conversation:  Sender → Target', fontsize=10, fontweight='bold')
    ax.text(3.05, 0.0,
            'aggressive_count = 6   ·   active_days = 5   ·   span = 5 days   ·   intent = 0.9 (death threat)\n'
            '→ repeated aggression persisting over time  ⇒  REPETITION signal HIGH  ⇒  Cyberbullying',
            ha='center', fontsize=8.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fdf0d5', ec='#b07a00'))
    leg = [Line2D([0], [0], marker='X', color='w', markerfacecolor='#c44e52',
                  markeredgecolor='k', markersize=11, label='Aggressive message'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#9fb0c0',
                  markersize=9, label='Non-aggressive message')]
    ax.legend(handles=leg, loc='upper right', ncol=1, frameon=True, fontsize=9)
    ax.set_xlim(0.4, 5.75); ax.set_ylim(-0.2, 2.3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Repetition & intent scoring over a conversation timeline',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'repetition_timeline.png')


def pillars():
    fig, ax = _canvas(11, 8)
    box(ax, 2.3, 6.4, 3.0, 1.2, 'Aggression\n(is the message\nhostile / abusive?)', fc='#d9ead3', bold=True)
    box(ax, 5.5, 6.4, 3.0, 1.2, 'Repetition\n(does it recur\nover time?)', fc='#d9ead3', bold=True)
    box(ax, 8.7, 6.4, 3.0, 1.2, 'Intent to harm\n(deliberate\npurpose?)', fc='#d9ead3', bold=True)
    box(ax, 8.9, 4.1, 3.0, 0.9, 'Peerness\n(social context)', fc='#fdf0d5')
    box(ax, 4.3, 2.3, 4.2, 1.1, 'CYBERBULLYING', fc='#f4cccc', fs=13, bold=True)
    for bx in (2.3, 5.5, 8.7):
        arrow(ax, bx, 5.8, 4.3, 2.9)
    arrow(ax, 7.4, 4.1, 5.6, 2.65, 'context')
    ax.text(5.5, 0.9,
            'A single aggressive message is not cyberbullying — it becomes cyberbullying only when it is\n'
            'repeated over time, carries intent to harm, and is read in the context of the peer relationship.',
            ha='center', fontsize=9, color='#444', style='italic')
    ax.set_title('The behavioural pillars of cyberbullying', fontsize=13, fontweight='bold')
    _save(fig, 'pillars.png')


def main():
    pillars(); overall(); stage1(); stage2(); image_branch(); data_views()
    preprocess_pipeline(); timeline_scoring()
    print(f'\nDiagrams written to {FIG_DIR}')


if __name__ == '__main__':
    main()
