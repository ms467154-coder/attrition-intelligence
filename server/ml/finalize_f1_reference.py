from __future__ import annotations
import json, os, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix
from xgboost import XGBClassifier
from round2_optimize import make_pipe

SEED=42
ROOT=Path(os.environ.get('ATTRITION_PROJECT_ROOT','/home/ubuntu/audit'))
DATA=ROOT/'WA_Fn-UseC_-HR-Employee-Attrition.csv'; OUT=ROOT/'ml_f1_optimization'; OUT.mkdir(exist_ok=True)

def metric(y,p,prob):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'f1':float(f1_score(y,p,zero_division=0)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'pr_auc':float(average_precision_score(y,prob)),'roc_auc':float(roc_auc_score(y,prob)),'brier':float(brier_score_loss(y,prob)),'log_loss':float(log_loss(y,np.c_[1-prob,prob],labels=[0,1])),'threshold':None,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
class Ensemble:
    def __init__(self,models,weights): self.models=models; self.weights=weights
    def fit(self,X,y):
        for m in self.models.values(): m.fit(X,y)
        return self
    def predict_proba(self,X):
        p=sum(self.weights[k]*m.predict_proba(X)[:,1] for k,m in self.models.items()); return np.c_[1-p,p]

def main():
    df=pd.read_csv(DATA); y=(df.pop('Attrition')=='Yes').astype(int); excluded=[c for c in df if df[c].nunique(dropna=False)<=1]+(['EmployeeNumber'] if 'EmployeeNumber' in df else []); X=df.drop(columns=excluded); num=X.select_dtypes(include=['number']).columns.tolist(); cat=X.select_dtypes(exclude=['number']).columns.tolist(); Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    models={'logistic_C_0.127':make_pipe(LogisticRegression(C=.1274275,max_iter=3000,solver='liblinear',random_state=SEED),num,cat,False),'xgboost':make_pipe(XGBClassifier(n_estimators=500,max_depth=2,learning_rate=.03,min_child_weight=3,subsample=.85,colsample_bytree=.85,reg_lambda=5,eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=2),num,cat,False)}; weights={'logistic_C_0.127':.6,'xgboost':.4}; probs={k:cross_val_predict(m,Xdev,ydev,cv=cv,method='predict_proba',n_jobs=-1)[:,1] for k,m in models.items()}; dev_prob=sum(weights[k]*probs[k] for k in models)
    rows=[]
    for t in np.arange(.01,.991,.01):
        p=(dev_prob>=t).astype(int); m=metric(ydev,p,dev_prob); m['threshold']=round(float(t),2); rows.append(m)
    tdf=pd.DataFrame(rows); best=tdf.sort_values(['f1','precision','recall'],ascending=False).iloc[0].to_dict(); tdf.to_csv(OUT/'chosen_ensemble_thresholds.csv',index=False)
    final=Ensemble(models,weights).fit(Xdev,ydev); test_prob=final.predict_proba(Xtest)[:,1]; test_pred=(test_prob>=best['threshold']).astype(int); test=metric(ytest,test_pred,test_prob); test['threshold']=best['threshold']; test['model']='soft_ensemble_logistic_xgboost';
    rng=np.random.default_rng(SEED); boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(ytest),len(ytest)); yy=ytest.iloc[ix]
        if yy.nunique()<2: continue
        boot.append(metric(yy,test_pred[ix],test_prob[ix]))
    ci=pd.DataFrame(boot).quantile([.025,.975]).T; ci.to_csv(OUT/'final_test_bootstrap_ci.csv')
    joblib.dump(final,OUT/'best_f1_model.joblib')
    rec={'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'seed':SEED,'cv':'StratifiedKFold(5,shuffle=True,random_state=42)','feature_set':list(X.columns),'excluded_features':excluded,'model':'soft_ensemble_logistic_xgboost','weights':weights,'hyperparameters':{'logistic_C':.1274275,'xgboost':{'n_estimators':500,'max_depth':2,'learning_rate':.03,'min_child_weight':3,'subsample':.85,'colsample_bytree':.85,'reg_lambda':5}},'cv_metrics':best,'test_metrics':test,'test_ci':ci.to_dict('index'),'target_f1':.85,'target_achieved':bool(test['f1']>=.85)}; (OUT/'f1_final_records.json').write_text(json.dumps(rec,indent=2,default=float),encoding='utf-8')
    summary=f'''# F1 Optimization Result\n\n- **Best model:** soft ensemble of Logistic Regression and XGBoost\n- **Best features:** all leakage-safe features after excluding constant columns and `EmployeeNumber`; no target-derived or test-derived features\n- **Best hyperparameters:** Logistic Regression `C=0.1274275`; XGBoost `n_estimators=500`, `max_depth=2`, `learning_rate=0.03`, `min_child_weight=3`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=5`; OOF weights Logistic `0.6`, XGBoost `0.4`\n- **Best threshold:** `{best['threshold']:.2f}`\n- **CV F1:** `{best['f1']:.4f}`\n- **Test F1:** `{test['f1']:.4f}`\n- **Precision:** `{test['precision']:.4f}`\n- **Recall:** `{test['recall']:.4f}`\n- **What changed:** compared seven model families, engineered interactions, positive-class weights, dense F1 threshold sweeps, and OOF soft ensembles; the selected two-model ensemble was chosen on development-only F1 and the final test was accessed only after freezing it.\n- **Whether F1 >= 0.85 was achieved:** **{test['f1']>=.85}**.\n\nThe development F1 of `{best['f1']:.4f}` and final test F1 of `{test['f1']:.4f}` indicate that the current data does not support F1 0.85 without leakage or artificial inflation.\n'''; (OUT/'f1_result_summary.md').write_text(summary,encoding='utf-8'); print(json.dumps({'cv':best,'test':test,'target_achieved':test['f1']>=.85},indent=2,default=float))
if __name__=='__main__': main()
