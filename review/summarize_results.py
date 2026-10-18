"""Regenerate standardized-feature source/transfer comparisons from public aggregates."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parents[1];O=R/'review_outputs';O.mkdir(exist_ok=True)
p=pd.read_csv(R/'results/standardization/paired_comparison.csv')
p['domain']=p.source_standardized.eq(p.target).map({True:'In domain',False:'Cross domain'})
p.groupby(['encoder_standardized','domain','rule']).agg(
    evaluations=('rule','size'),raw_collapse=('collapsed_raw_features','sum'),standardized_collapse=('collapsed_standardized','sum'),
    raw_macro_f1=('macro_f1_raw_features','mean'),standardized_macro_f1=('macro_f1_standardized','mean'),
    raw_auroc=('auroc_raw_features','mean'),standardized_auroc=('auroc_standardized','mean')).reset_index().to_csv(O/'standardization_summary.csv',index=False)
q=p[p.rule.eq('raw')&p.domain.eq('In domain')&p.ratio_standardized.eq(.8)&p.loss_standardized.eq('unweighted_ce')]
q.groupby(['encoder_standardized','source_standardized']).agg(n=('rule','size'),raw_collapse=('collapsed_raw_features','sum'),standardized_collapse=('collapsed_standardized','sum'),raw_f1=('macro_f1_raw_features','mean'),standardized_f1=('macro_f1_standardized','mean'),standardized_f1_sd=('macro_f1_standardized','std'),raw_auroc=('auroc_raw_features','mean'),standardized_auroc=('auroc_standardized','mean')).reset_index().to_csv(O/'standardization_unweighted_share080.csv',index=False)
colors={'coaid':'#0072B2','fakenewsnet':'#D55E00','liar':'#009E73'}
fig,axs=plt.subplots(2,2,figsize=(7.2,6.1),sharex=True,sharey=True)
for row,encoder in enumerate(['RoBERTa','BERT']):
    for col,domain in enumerate(['In domain','Cross domain']):
        ax=axs[row,col];a=p[p.rule.eq('raw')&p.encoder_standardized.eq(encoder)&p.domain.eq(domain)]
        ax.plot([0,1],[0,1],color='.65',lw=1,zorder=0)
        for src,color in colors.items():
            for share,marker in [(.5,'o'),(.8,'^')]:
                b=a[a.source_standardized.eq(src)&a.ratio_standardized.eq(share)]
                ax.scatter(b.macro_f1_raw_features,b.macro_f1_standardized,c=color,marker=marker,s=23,alpha=.7,linewidths=.35,edgecolors='white')
        ax.set_title(encoder+' '+domain.lower(),fontsize=10);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.15)
        if row==1:ax.set_xlabel('Raw-feature macro-F1',fontsize=10)
        if col==0:ax.set_ylabel('Standardized-feature macro-F1',fontsize=10)
handles=[Line2D([],[],color=c,marker='o',ls='',label={'coaid':'CoAID','fakenewsnet':'FakeNewsNet','liar':'LIAR'}[k]) for k,c in colors.items()]
handles += [Line2D([],[],color='.3',marker=m,ls='',label='Share '+str(r)) for r,m in [(.5,'o'),(.8,'^')]]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=8)
fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(O/'standardization_paired.png',dpi=300);fig.savefig(O/'standardization_paired.pdf');plt.close(fig)
print('Saved aggregate summaries and comparison figure to',O)
print('Points reuse classifiers across targets/rules; three seeds on one split are descriptive.')
