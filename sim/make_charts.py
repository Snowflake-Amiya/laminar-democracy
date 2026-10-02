"""
make_charts.py -- Publication figures for the TDD white paper.

Follows typesetting/charts.md:
  - top/right spines deleted; grid dashed 0.5pt at <=20% opacity (or omitted
    when values are labeled directly)
  - frameless legends placed OUTSIDE the data area
  - line charts label only first/last (and extremes where meaningful)
  - horizontal bars for long category labels
  - no internal chart titles (external PDF captions serve as identifiers)
  - one color family from the locked cascade palette (serenity/seed 9)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 10.5, 'axes.labelsize': 11, 'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#aab4bd',
    'xtick.color': '#72777c', 'ytick.color': '#72777c',
    'text.color': '#17191a', 'axes.labelcolor': '#17191a',
})

# Locked cascade palette (serenity / seed 9)
ACCENT = '#1f68b1'      # series 1 (T3)
ACCENT_2 = '#6e4fca'    # series 2 (SORT / tier-3)
HEADER = '#526375'      # series 3 (T2)
ICON = '#587a9c'
BORDER = '#aab4bd'      # pale (DD baseline)
MUTED = '#72777c'
TEXT = '#17191a'
SEM_ERR = '#b05148'     # capture risk (semantic exception, low-sat)
SEM_ERR_LT = '#d2908a'  # same-hue lightness variant (with-veto)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIGS = os.path.join(HERE, 'figs')
os.makedirs(FIGS, exist_ok=True)

SYS_COLOR = {'DD': BORDER, 'T2': HEADER, 'T3': ACCENT, 'SORT': ACCENT_2}
SYS_LABEL = {'DD': 'Direct democracy', 'T2': 'Two-tier (T2)',
             'T3': 'Three-tier (T3)', 'SORT': 'Citizens\' assembly'}


def style_ax(ax, grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.2,
                color=HEADER)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, facecolor='white')
    plt.close(fig)
    print(f"  wrote figs/{name}")


# ----------------------------------------------------------------------
# Fig 2 -- E1: decision quality by system x domain
# ----------------------------------------------------------------------
def fig_e1():
    rows = read_csv('e1_accuracy.csv')
    doms = ['A', 'B', 'C']
    systems = ['DD', 'T2', 'T3', 'SORT']
    acc = {(r['system'], r['domain']): float(r['accuracy']) for r in rows}

    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    x = np.arange(len(doms))
    w = 0.19
    for i, s in enumerate(systems):
        vals = [acc[(s, d)] for d in doms]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92,
                      color=SYS_COLOR[s], label=SYS_LABEL[s], edgecolor='none')
        for b, v in zip(bars, vals):
            inside = v > 0.9
            ax.text(b.get_x() + b.get_width() / 2,
                    v - 0.045 if inside else v + 0.012,
                    f'{v:.2f}', ha='center', va='top' if inside else 'bottom',
                    fontsize=8.2,
                    color='white' if inside else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(['Domain A\n(basic civic)',
                        'Domain B\n(intermediate)',
                        'Domain C\n(highly technical)'])
    ax.set_ylabel('Decision accuracy')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_ax(ax, grid=False)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.16), ncol=4,
              frameon=False, handlelength=1.2, columnspacing=1.4)
    save(fig, 'fig2_e1_accuracy.png')


# ----------------------------------------------------------------------
# Fig 3 -- E2: welfare and C-accuracy vs test validity
# ----------------------------------------------------------------------
def fig_e2():
    rows = read_csv('e2_welfare.csv')
    vs = sorted({float(r['validity']) for r in rows})
    wel = {s: [0] * len(vs) for s in SYS_COLOR}
    accC = {s: [0] * len(vs) for s in SYS_COLOR}
    for r in rows:
        i = vs.index(float(r['validity']))
        wel[r['system']][i] = float(r['welfare'])
        accC[r['system']][i] = float(r['acc_C'])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5),
                             constrained_layout=True)
    for ax, data, ylab in [(axes[0], wel, 'Aggregate welfare'),
                           (axes[1], accC, 'Accuracy on domain C')]:
        for s in ['DD', 'SORT', 'T2', 'T3']:
            ax.plot(vs, data[s], '-o', color=SYS_COLOR[s],
                    label=SYS_LABEL[s], linewidth=2.4, markersize=4.5,
                    markerfacecolor='white', markeredgewidth=1.6)
        # first/last labels for the protagonist and baseline only
        for s, dy in [('T3', 0.012), ('DD', -0.024)]:
            ax.annotate(f'{data[s][0]:.2f}', (vs[0], data[s][0]),
                        textcoords='offset points', xytext=(-2, dy * 400),
                        fontsize=8.2, color=SYS_COLOR[s], ha='right')
            ax.annotate(f'{data[s][-1]:.2f}', (vs[-1], data[s][-1]),
                        textcoords='offset points', xytext=(4, dy * 300),
                        fontsize=8.2, color=SYS_COLOR[s], ha='left')
        ax.set_xlabel('Certification-test validity v')
        ax.set_ylabel(ylab)
        ax.set_xticks(vs)
        ax.set_ylim(0.38 if ylab.startswith('Accuracy') else 0.70, 1.03)
        style_ax(ax)
    axes[0].annotate('(a)', (-0.14, 1.06), xycoords='axes fraction',
                     fontsize=10, color=MUTED)
    axes[1].annotate('(b)', (-0.16, 1.06), xycoords='axes fraction',
                     fontsize=10, color=MUTED)
    axes[0].legend(loc='lower right', frameon=False, handlelength=1.4)
    save(fig, 'fig3_e2_validity.png')


# ----------------------------------------------------------------------
# Fig 4 -- E3: tier-3 pool size and groupthink (Hong-Page)
# ----------------------------------------------------------------------
def fig_e3():
    rows = read_csv('e3_poolsize.csv')
    t3 = [r for r in rows if r['system'] == 'T3' and r['validity'] == '0.7']
    q3s, curves = [], {b: [] for b in ['0.0', '0.3', '0.6', '1.0']}
    for r in t3:
        q3 = float(r['cutoff'])
        if q3 not in q3s:
            q3s.append(q3)
    q3s.sort()
    for b in curves:
        for q in q3s:
            m = [r for r in t3 if r['beta'] == b and float(r['cutoff']) == q]
            curves[b].append(float(m[0]['acc_C']))
    t2s = [r for r in rows if r['system'] == 'T2' and r['validity'] == '0.7']
    t2_row = [r for r in t2s if r['beta'] == '0.3' and r['cutoff'] == '0.1'][0]
    t2_best_c = float(t2_row['acc_C'])
    t2_best_b = float(t2_row['acc_B'])

    # same-hue lightness ramp for the groupthink coefficient
    ramp = ['#1f68b1', '#5a8fc7', '#8ab0d9', '#b7cfe8']
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for b, c in zip(['0.0', '0.3', '0.6', '1.0'], ramp):
        ax.plot(q3s, curves[b], '-o', color=c, linewidth=2.3, markersize=4.2,
                markerfacecolor='white', markeredgewidth=1.4,
                label=f'groupthink beta = {b}')
    ax.axhline(t2_best_c, color=HEADER, linestyle='--', linewidth=1.6,
               label=f'best two-tier (cutoff 10%), C = {t2_best_c:.2f}')
    ax.annotate(f'two-tier on B: {t2_best_b:.2f}',
                xy=(q3s[-1], t2_best_c), xytext=(q3s[-1] - 0.028,
                t2_best_c - 0.052), fontsize=8.2, color=HEADER)
    ax.annotate(f'{curves["0.0"][0]:.2f}', (q3s[0], curves['0.0'][0]),
                textcoords='offset points', xytext=(2, 6), fontsize=8.2,
                color=ramp[0])
    ax.annotate(f'{curves["0.0"][-1]:.2f}', (q3s[-1], curves['0.0'][-1]),
                textcoords='offset points', xytext=(-4, 6), fontsize=8.2,
                color=ramp[0], ha='right')
    ax.set_xlabel('Tier-3 pool share q (share of population certified)')
    ax.set_ylabel('Accuracy on domain C')
    ax.set_xticks(q3s)
    ax.set_xticklabels([f'{q:.0%}' for q in q3s])
    ax.set_ylim(0.82, 1.015)
    style_ax(ax)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.14), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.2)
    save(fig, 'fig4_e3_poolsize.png')


# ----------------------------------------------------------------------
# Fig 5 -- E4: referendum signature thresholds
# ----------------------------------------------------------------------
def fig_e4():
    rows = read_csv('e4_referendum.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3),
                             constrained_layout=True, sharey=True)
    order = ['basic (all citizens)', 'tier-2 members', 'tier-3 members']
    colors = {'basic (all citizens)': ACCENT, 'tier-2 members': HEADER,
              'tier-3 members': ACCENT_2}
    for ax, tag, title in [(axes[0], 'raw y=2 z=2', '(a) raw rule  y=2, z=2'),
                           (axes[1], 'adjusted y=2 z=5', '(b) adjusted rule  y=2, z=5')]:
        sub = {r['path']: r for r in rows if r['parameterization'] == tag}
        days = [float(sub[p]['median_days']) for p in order]
        succ = [float(sub[p]['success_rate']) for p in order]
        y = np.arange(len(order))[::-1]
        bars = ax.barh(y, days, height=0.52, color=[colors[p] for p in order],
                       edgecolor='none')
        ax.set_yticks(y)
        ax.set_yticklabels(['Basic\n(all citizens)', 'Tier-2\nmembers',
                            'Tier-3\nmembers'] if tag.startswith('raw')
                           else ['', '', ''])
        for b, d, s, p in zip(bars, days, succ, order):
            txt = f'{d:.0f} d' + ('' if s >= 0.999 else f'  ({s:.0%} pass)')
            ax.text(min(d, 168) + 3, b.get_y() + b.get_height() / 2, txt,
                    va='center', fontsize=8.6, color=MUTED)
        ax.axvline(180, color=SEM_ERR, linestyle=':', linewidth=1.4)
        ax.text(179, 2.42, 'window = 180 d', fontsize=8, color=SEM_ERR,
                ha='right')
        ax.set_xlabel('Median days to threshold')
        ax.set_xlim(0, 235)
        ax.set_title(title, fontsize=9.5, color=MUTED, loc='left', pad=8)
        style_ax(ax)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.2,
                color=HEADER)
        ax.grid(False, axis='y')
    save(fig, 'fig5_e4_referendum.png')


# ----------------------------------------------------------------------
# Fig 6 -- E5: capture resistance
# ----------------------------------------------------------------------
def fig_e5():
    rows = read_csv('e5_capture.csv')
    labels, no_veto, veto = [], [], []
    pretty = {
        'basic referendum - weak misinformation':
            'Basic path - weak misinformation',
        'basic referendum - strong misinformation':
            'Basic path - strong misinformation',
        'tier-2 path - weak misinformation':
            'Tier-2 path - weak misinformation',
        'tier-2 path - strong misinformation':
            'Tier-2 path - strong misinformation',
        'tier-3 path - weak misinformation':
            'Tier-3 path - weak misinformation',
        'tier-3 path - strong misinformation':
            'Tier-3 path - strong misinformation',
        'tier-3 path - bribe 5% of pool':
            'Tier-3 path - bribe 5% of pool',
        'tier-3 path - bribe 15% of pool':
            'Tier-3 path - bribe 15% of pool',
        'tier-2 path - bribe 15% of pool':
            'Tier-2 path - bribe 15% of pool',
    }
    for r in rows:
        labels.append(pretty[r['scenario']])
        no_veto.append(float(r['p_capture']))
        veto.append(float(r['p_capture_with_veto']))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    y = np.arange(len(labels))[::-1]
    h = 0.36
    b1 = ax.barh(y + h / 2 + 0.02, no_veto, height=h, color=SEM_ERR,
                 label='no veto overlay', edgecolor='none')
    b2 = ax.barh(y - h / 2 - 0.02, veto, height=h, color=SEM_ERR_LT,
                 label='with democratic veto', edgecolor='none')
    for b, v in list(zip(b1, no_veto)) + list(zip(b2, veto)):
        if v >= 0.005:
            ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f'{v:.2f}',
                    va='center', fontsize=7.8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlabel('Probability that a welfare-reducing measure passes')
    ax.set_xlim(0, 0.84)
    style_ax(ax, grid=False)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.2,
            color=HEADER)
    ax.grid(False, axis='y')
    ax.legend(loc='lower right', frameon=False, handlelength=1.3)
    save(fig, 'fig6_e5_capture.png')


# ----------------------------------------------------------------------
# Fig 7 -- E6: incentive dynamics
# ----------------------------------------------------------------------
def fig_e6():
    rows = read_csv('e6_dynamics.csv')
    tdd = [r for r in rows if r['system'] == 'TDD']
    per = [int(r['period']) for r in tdd]
    compC = [float(r['mean_competence_C']) for r in tdd]
    compB = [float(r['mean_competence_B']) for r in tdd]
    s2 = [float(r['share_t2']) for r in tdd]
    s3 = [float(r['share_t3']) for r in tdd]
    dd_c = float([r for r in rows if r['system'] != 'TDD'][0]
                 ['mean_competence_C'])
    dd_b = float([r for r in rows if r['system'] != 'TDD'][0]
                 ['mean_competence_B'])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot(per, compC, '-o', color=ACCENT, linewidth=2.4, markersize=4,
            markerfacecolor='white', markeredgewidth=1.5,
            label='tiered (with status incentive)')
    ax.axhline(dd_c, color=MUTED, linestyle='--', linewidth=1.6,
               label='direct democracy (static)')
    ax.annotate(f'{compC[0]:.2f}', (per[0], compC[0]),
                textcoords='offset points', xytext=(-4, -2), fontsize=8.2,
                color=ACCENT, ha='right')
    ax.annotate(f'{compC[-1]:.2f}', (per[-1], compC[-1]),
                textcoords='offset points', xytext=(4, 0), fontsize=8.2,
                color=ACCENT)
    ax.set_xlabel('Period')
    ax.set_ylabel('Mean citizen competence (domain C)')
    ax.set_ylim(0.44, 0.66)
    style_ax(ax)
    ax.legend(loc='upper left', frameon=False, handlelength=1.4)

    ax = axes[1]
    ax.plot(per, s2, '-o', color=HEADER, linewidth=2.4, markersize=4,
            markerfacecolor='white', markeredgewidth=1.5,
            label='tier-2 status share')
    ax.plot(per, s3, '-o', color=ACCENT_2, linewidth=2.4, markersize=4,
            markerfacecolor='white', markeredgewidth=1.5,
            label='tier-3 status share')
    ax.annotate(f'{s2[0]:.0%} -> {s2[-1]:.0%}', (per[-1], s2[-1]),
                textcoords='offset points', xytext=(-6, 8), fontsize=8.4,
                color=HEADER, ha='right')
    ax.annotate(f'{s3[0]:.0%} -> {s3[-1]:.0%}', (per[-1], s3[-1]),
                textcoords='offset points', xytext=(-6, -13), fontsize=8.4,
                color=ACCENT_2, ha='right')
    ax.set_xlabel('Period')
    ax.set_ylabel('Share of population holding status')
    ax.set_ylim(0, 0.56)
    style_ax(ax)
    ax.legend(loc='upper left', frameon=False, handlelength=1.4)
    axes[0].annotate('(a)', (-0.13, 1.06), xycoords='axes fraction',
                     fontsize=10, color=MUTED)
    axes[1].annotate('(b)', (-0.16, 1.06), xycoords='axes fraction',
                     fontsize=10, color=MUTED)
    save(fig, 'fig7_e6_dynamics.png')


if __name__ == '__main__':
    print("Generating white-paper figures ...")
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
    fig_e5()
    fig_e6()
    print("Done.")
