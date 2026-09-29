from pathlib import Path
from pptx import Presentation
import pandas as pd, numpy as np, json, zipfile, hashlib
from lxml import etree
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'defense_presentation_2026-09-24'
p=Presentation(OUT/'Ece_CANGA_Final_Thesis_Defense.pptx')
report={};assert len(p.slides)==40
report['slides']=40;report['editable_charts']=sum(sh.has_chart for s in p.slides for sh in s.shapes)
report['editable_tables']=sum(sh.has_table for s in p.slides for sh in s.shapes)
def tab(i):return next(sh.table for sh in p.slides[i-1].shapes if sh.has_table)
base=ROOT/'outputs/thesis_consolidated_results_v1'
for no,file in [(31,'youtube_five_model_pooled.csv'),(32,'synthetic_five_model_augmented_pooled.csv')]:
 d=pd.read_csv(base/file);t=tab(no)
 for i,r in d.iterrows():
  for j,col in enumerate(['macro_f1','macro_average_precision','macro_roc_auc'],1):assert f'{float(t.cell(i+1,j).text):.4f}'==f'{r[col]:.4f}',(no,i,col)
report['aggregate_table_values_verified']=30
preds=ROOT/'outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions'
ids=['logistic_regression','xgboost','random_forest','bert','bilstm_attention']
labels=['joy','trust','anticipation','surprise','disgust','sadness','fear']
natural=pd.read_csv(base/'youtube_five_model_pooled.csv')
for j,model in enumerate(ids):
 d=pd.read_csv(preds/f'{model}_oof.csv');y=d[[f'true_{l}' for l in labels]].to_numpy();prob=d[[f'prob_{l}' for l in labels]].to_numpy();pred=(prob>=.3).astype(int)
 r=natural[natural.model_id==model].iloc[0]
 for name,value in [('macro_f1',f1_score(y,pred,average='macro',zero_division=0)),('macro_average_precision',average_precision_score(y,prob,average='macro')),('macro_roc_auc',roc_auc_score(y,prob,average='macro'))]:assert round(r[name],4)==round(value,4)
 per=f1_score(y,pred,average=None,zero_division=0)
 for k,v in enumerate(per):assert f'{v:.4f}'==f'{float(tab(37).cell(k+1,j+2).text):.4f}'
 for k,cutoff in enumerate([.2,.3,.5,.7]):assert f'{f1_score(y,(prob>=cutoff).astype(int),average="macro",zero_division=0):.4f}'==f'{float(tab(39).cell(j+1,k+1).text):.4f}'
 for k,l in enumerate(labels):assert int(tab(37).cell(k+1,1).text)==int(y[:,k].sum())
 if model=='bert':
  for i,l in enumerate(['joy','trust','anticipation']):assert int(tab(20).cell(i+1,2).text)==int(d[f'pred_{l}'].sum())
report['recomputed_natural_aggregate_metrics']=15
report['recomputed_per_emotion_f1']=35
report['recomputed_threshold_f1']=20
report['emotion_support_counts_verified']=7
fold=pd.read_csv(preds.parent/'tables/fold_assignments.csv');ft=fold.groupby('fold_id').agg(perfumes=('perfume_id','size'),groups=('source_group_id','nunique'))
for i,r in ft.iterrows():
 assert int(tab(33).cell(i,1).text)==r.perfumes
 assert int(tab(33).cell(i,2).text)==r.groups
assert fold.source_group_id.nunique()==359
assert fold.groupby('source_group_id').fold_id.nunique().max()==1
report['source_groups']=359;report['source_group_fold_leakage']=0
# Exact underlying editable chart values are checked against their authoritative artifacts.
for no,files in [(17,['youtube_five_model_pooled.csv']),(24,['youtube_five_model_pooled.csv','synthetic_five_model_augmented_pooled.csv'])]:
 c=next(sh.chart for sh in p.slides[no-1].shapes if sh.has_chart)
 for ser,file in zip(c.series,files):assert np.allclose(ser.values,pd.read_csv(base/file).macro_average_precision,atol=.00005)
c=next(sh.chart for sh in p.slides[18].shapes if sh.has_chart);assert list(c.series[0].values)==[622,476,366,44,19,11,3]
report['chart_values_verified']=22
links=[]
with zipfile.ZipFile(OUT/'Ece_CANGA_Final_Thesis_Defense.pptx') as z:
 assert z.testzip() is None
 for name in z.namelist():
  if name.endswith(('.xml','.rels')):etree.fromstring(z.read(name))
  if name.endswith('.rels'):
   for rel in etree.fromstring(z.read(name)):
    if rel.get('TargetMode')=='External':links.append(rel.get('Target'))
 assert not any('vbaProject' in n for n in z.namelist())
assert len(set(links))==8
report['reference_hyperlinks']=sorted(set(links));report['zip_xml_integrity']='passed'
report['thesis_sha256']=hashlib.sha256((ROOT/'docs/thesis/Ece_CANGA_Thesis.docx').read_bytes()).hexdigest()
assert report['thesis_sha256']=='6a9e17e44beec682462e7c782364374bc2918e7267b4baeb5af7d5845685f974'
(OUT/'build'/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
