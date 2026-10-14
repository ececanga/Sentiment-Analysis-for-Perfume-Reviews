import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
root=Path('outputs/thesis_three_seed_v1');inv=json.loads((root/'inventory.json').read_text());summ=json.loads((root/'summary.json').read_text())
labels=['joy','sadness','fear','trust','surprise','anticipation','disgust']; grouped={}; baseline=None; records=[]
assert len(inv)==195 and all(x['status']=='complete' for x in inv)
for job in inv:
 p=root/job['run_id'];df=pd.read_csv(p/'oof.csv').sort_values('perfume_id');assert len(df)==784 and df.perfume_id.is_unique
 y=df[['true_'+l for l in labels]].to_numpy();prob=df[['prob_'+l for l in labels]].to_numpy();assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
 identity=df[['perfume_id','fold_id']+['true_'+l for l in labels]].reset_index(drop=True)
 if baseline is None:baseline=identity
 else:pd.testing.assert_frame_equal(baseline,identity,check_dtype=False)
 pred=prob>=.3
 assert np.array_equal(pred,df[['pred_'+l for l in labels]].to_numpy())
 vals={'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'macro_roc_auc':roc_auc_score(y,prob,average='macro'),'macro_average_precision':average_precision_score(y,prob,average='macro')}
 stored=json.loads((p/'metrics.json').read_text())
 for k,v in vals.items():assert np.isclose(v,stored[k],atol=1e-12,rtol=0),(job['run_id'],k,v,stored[k])
 key=tuple(job[k] for k in ['experiment','model','condition','variant']);grouped.setdefault(key,{})[job['model_seed']]=vals
 records.append({'run_id':job['run_id'],'oof_sha256':hashlib.sha256((p/'oof.csv').read_bytes()).hexdigest(),**vals})
for r in summ['summaries']:
 assert not r['incomplete'] and r['n_completed_seeds']==5
 vals=[x[r['metric']] for x in grouped[tuple(r[k] for k in ['experiment','model','condition','variant'])].values()]
 assert np.isclose(np.mean(vals),r['mean'],atol=1e-12,rtol=0)
 assert np.isclose(np.std(vals,ddof=1),r['sample_std_ddof1'],atol=1e-12,rtol=0)
for r in summ['paired_comparisons']:
 a,b=grouped[tuple(r['comparison'])],grouped[tuple(r['minus'])]
 dif=[a[s][r['metric']]-b[s][r['metric']] for s in summ['expected_seeds']]
 assert not r['incomplete']
 assert np.isclose(np.mean(dif),r['mean'],atol=1e-12,rtol=0)
 assert np.isclose(np.std(dif,ddof=1),r['sample_std_ddof1'],atol=1e-12,rtol=0)
out=Path('outputs/thesis_five_seed_revision_20261012')
pd.DataFrame(records).to_csv(out/'audited_per_seed_metrics.csv',index=False)
pd.DataFrame(summ['summaries']).to_csv(out/'five_seed_summary.csv',index=False)
pd.DataFrame(summ['paired_comparisons']).to_csv(out/'paired_seed_differences.csv',index=False)
report={'status':'passed','jobs':len(inv),'summaries':len(summ['summaries']),'paired_comparisons':len(summ['paired_comparisons']),'checks':['784 unique perfumes per job','identical IDs, labels and outer folds','finite probabilities in [0,1]','predictions match threshold 0.3','metrics recomputed from OOF predictions','means and sample SD recomputed','paired differences recomputed'],'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()}
(out/'results_audit.json').write_text(json.dumps(report,indent=2));print(report)
