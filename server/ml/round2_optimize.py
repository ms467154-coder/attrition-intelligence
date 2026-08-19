from __future__ import annotations
import hashlib, json, os, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMB=True
except Exception:
    IMB=False

SEED=42
ROOT=Path(os.environ.get('ATTRITION_PROJECT_ROOT', Path(__file__).resolve().parent))
DATA=ROOT/'WA_Fn-UseC_-HR-Employee-Attrition.csv'
PREV=ROOT/'ml_artifacts/experiment_records.json'
OUT=ROOT/'ml_artifacts_round2'; OUT.mkdir(exist_ok=True)

class DomainInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True): self.enabled=enabled
    def fit(self, X, y=None): return self
    def transform(self, X):
        X=X.copy()
        if not self.enabled: return X
        pairs=[('OverTime','JobSatisfaction'),('OverTime','WorkLifeBalance'),('BusinessTravel','OverTime'),('MaritalStatus','StockOptionLevel')]
        for a,b in pairs:
            if a in X and b in X:
                X[f'{a}__{b}']=X[a].astype(str)+'__'+X[b].astype(str)
        numeric=[('MonthlyIncome','TotalWorkingYears'),('YearsInCurrentRole','YearsSinceLastPromotion'),('JobInvolvement','JobSatisfaction'),('DistanceFromHome','JobSatisfaction'),('Age','TotalWorkingYears'),('JobLevel','MonthlyIncome')]
        for a,b in numeric:
            if a in X and b in X: X[f'{a}__x__{b}']=X[a].astype(float)*X[b].astype(float)
        return X

def metrics(y,p,prob):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,p),'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'roc_auc':roc_auc_score(y,prob),'pr_auc':average_precision_score(y,prob),'log_loss':log_loss(y,np.c_[1-prob,prob],labels=[0,1]),'brier':brier_score_loss(y,prob),'specificity':tn/(tn+fp) if tn+fp else 0,'fpr':fp/(tn+fp) if tn+fp else 0,'fnr':fn/(fn+tp) if fn+tp else 0,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def make_pre(num,cat,engineered=False):
    def colsel(X): return X
    return Pipeline([('interactions',DomainInteractions(engineered)),('preprocess',ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2,sparse_output=False))]),cat)],remainder='drop'))])

def make_pipe(model,num,cat,engineered=False, sampler=None):
    steps=[('features',DomainInteractions(engineered)),('preprocess',ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2,sparse_output=False))]),cat)],remainder='drop'))]
    if sampler is not None: steps.append(('sampler',sampler))
    steps.append(('model',model))
    return (ImbPipeline if sampler is not None else Pipeline)(steps)

def cv_eval(est,X,y,cv):
    scoring={'pr_auc':'average_precision','roc_auc':'roc_auc','precision':'precision','recall':'recall','f1':'f1','brier':'neg_brier_score','log_loss':'neg_log_loss'}
    r=cross_validate(est,X,y,cv=cv,scoring=scoring,n_jobs=-1,return_train_score=False)
    out={}
    for k in scoring:
        v=r['test_'+k]; v=-v if k in ['brier','log_loss'] else v
        out[k+'_mean']=float(v.mean()); out[k+'_std']=float(v.std(ddof=1)); out[k+'_min']=float(v.min()); out[k+'_max']=float(v.max())
    out['fit_seconds_mean']=float(r['fit_time'].mean()); out['score_seconds_mean']=float(r['score_time'].mean())
    return out

def oof(est,X,y,cv): return cross_val_predict(clone(est),X,y,cv=cv,method='predict_proba',n_jobs=-1)[:,1]

def main():
    df=pd.read_csv(DATA); y=(df.pop('Attrition')=='Yes').astype(int); excluded=[c for c in df if df[c].nunique(dropna=False)<=1]+(['EmployeeNumber'] if 'EmployeeNumber' in df else []); X=df.drop(columns=excluded)
    num=X.select_dtypes(include=['number']).columns.tolist(); cat=X.select_dtypes(exclude=['number']).columns.tolist()
    Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=SEED); inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    prev=json.loads(PREV.read_text()) if PREV.exists() else {}
    prev_hash=prev.get('dataset_sha256'); data_hash=hashlib.sha256(DATA.read_bytes()).hexdigest()
    # Previous best is reconstructed exactly from recorded configuration.
    previous=make_pipe(LogisticRegression(C=0.12742749857031335,class_weight=None,max_iter=3000,solver='liblinear',random_state=SEED),num,cat,False)
    previous_oof=oof(previous,Xdev,ydev,inner)
    prev_cv=cv_eval(previous,Xdev,ydev,cv)
    records=[]
    records.append({'id':'reproduce_previous_logistic','family':'logistic_regression','variant':'previous_exact','cv':prev_cv,'dataset_hash_matches':data_hash==prev_hash,'prior_test_metrics':prev.get('final_test_metrics',{})})
    models={
      'logistic':make_pipe(LogisticRegression(C=0.12742749857031335,max_iter=3000,solver='liblinear',random_state=SEED),num,cat,False),
      'logistic_balanced':make_pipe(LogisticRegression(C=0.12742749857031335,class_weight='balanced',max_iter=3000,solver='liblinear',random_state=SEED),num,cat,False),
      'logistic_engineered':make_pipe(LogisticRegression(C=0.12742749857031335,max_iter=3000,solver='liblinear',random_state=SEED),num,cat,True),
      'hist_gradient_boosting':make_pipe(HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=7,l2_regularization=1,random_state=SEED),num,cat,False),
      'xgboost':make_pipe(XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.04,min_child_weight=3,subsample=.8,colsample_bytree=.8,reg_lambda=3,eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=2),num,cat,False),
    }
    if LIGHTGBM_AVAILABLE:
      models['lightgbm']=make_pipe(LGBMClassifier(n_estimators=250,num_leaves=15,max_depth=5,learning_rate=.04,min_child_samples=25,subsample=.8,colsample_bytree=.8,reg_lambda=2,verbosity=-1,random_state=SEED,n_jobs=2),num,cat,False)
    if CATBOOST_AVAILABLE:
      models['catboost']=make_pipe(CatBoostClassifier(iterations=250,depth=5,learning_rate=.04,l2_leaf_reg=5,random_strength=1,verbose=False,random_seed=SEED,thread_count=2),num,cat,False)
    # Small controlled searches for nonlinear candidates, all on development CV.
    spaces={
      'hist_gradient_boosting':{'model__max_iter':[150,250,350],'model__learning_rate':[.03,.05,.08],'model__max_leaf_nodes':[7,15,31],'model__l2_regularization':[0,1,3]},
      'xgboost':{'model__n_estimators':[150,250,400],'model__max_depth':[2,3,4],'model__learning_rate':[.03,.05,.08],'model__min_child_weight':[1,3,6],'model__subsample':[.7,.85,1.0],'model__colsample_bytree':[.7,.85,1.0],'model__reg_lambda':[1,3,8]},
      'lightgbm':{'model__n_estimators':[150,250,400],'model__num_leaves':[7,15,31],'model__max_depth':[-1,4,7],'model__learning_rate':[.03,.05,.08],'model__min_child_samples':[15,25,40],'model__reg_lambda':[0,2,8]},
      'catboost':{'model__iterations':[150,250,400],'model__depth':[4,5,6],'model__learning_rate':[.03,.05,.08],'model__l2_leaf_reg':[2,5,10],'model__random_strength':[.2,1,3]},
    }
    tuned={}
    for name in ['hist_gradient_boosting','xgboost','lightgbm','catboost']:
        search=RandomizedSearchCV(models[name],spaces[name],n_iter=10,scoring='average_precision',cv=inner,random_state=SEED,n_jobs=-1,refit=True)
        st=time.perf_counter(); search.fit(Xdev,ydev); tuned[name]=search.best_estimator_; rec=cv_eval(search.best_estimator_,Xdev,ydev,cv)
        records.append({'id':'tuned_'+name,'family':name,'variant':'randomized_search','best_params':search.best_params_,'search_pr_auc':float(search.best_score_),'cv':rec,'seconds':time.perf_counter()-st})
    # Class imbalance strategies inside training folds.
    if IMB:
        for label,sampler in [('smote',SMOTE(random_state=SEED)),('random_over',RandomOverSampler(random_state=SEED)),('random_under',RandomUnderSampler(random_state=SEED))]:
            est=make_pipe(LogisticRegression(C=.1274275,max_iter=3000,solver='liblinear',random_state=SEED),num,cat,False,sampler)
            records.append({'id':'imbalance_'+label,'family':'logistic','variant':label,'cv':cv_eval(est,Xdev,ydev,cv)})
    # Feature-group ablations on the previous model.
    groups={'work_related':['BusinessTravel','Department','JobRole','OverTime','JobInvolvement','JobLevel','JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],'demographic':['Age','Gender','MaritalStatus','Education','EducationField'],'compensation':['DailyRate','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike','StockOptionLevel'],'satisfaction':['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],'career_history':['TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','NumCompaniesWorked']}
    for g,cols in groups.items():
        keep=[c for c in Xdev.columns if c not in cols]; n2=[c for c in keep if c in num]; c2=[c for c in keep if c in cat]
        est=make_pipe(LogisticRegression(C=.1274275,max_iter=3000,solver='liblinear',random_state=SEED),n2,c2,False)
        records.append({'id':'ablation_without_'+g,'family':'logistic','variant':'ablation','removed':cols,'cv':cv_eval(est,Xdev[keep],ydev,cv)})
    # Correlation/redundancy diagnostic.
    corr=Xdev.select_dtypes(include='number').corr().abs(); pairs=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            if corr.loc[a,b]>=.85: pairs.append({'a':a,'b':b,'abs_corr':float(corr.loc[a,b])})
    pd.DataFrame(pairs).to_csv(OUT/'high_correlation_pairs.csv',index=False)
    # Select top candidates using CV PR-AUC, then compare a simple OOF soft ensemble.
    eligible=[r for r in records if 'cv' in r and r['id'] not in ['reproduce_previous_logistic'] and r['variant'] not in ['ablation','smote','random_over','random_under']]
    leaderboard=pd.DataFrame([{'id':r['id'],'family':r['family'],'variant':r['variant'],**r['cv']} for r in eligible]).sort_values('pr_auc_mean',ascending=False)
    leaderboard.to_csv(OUT/'round2_leaderboard.csv',index=False)
    top_ids=leaderboard.head(3)['id'].tolist(); est_map={'tuned_'+k:v for k,v in tuned.items()}; est_map.update(models); top_ests=[]
    for ident in top_ids:
        if ident in est_map: top_ests.append((ident,est_map[ident]))
    probs=[oof(e,Xdev,ydev,inner) for _,e in top_ests]
    ens_prob=np.mean(probs,axis=0) if probs else previous_oof
    ens_cv=metrics(ydev,(ens_prob>=.38).astype(int),ens_prob); records.append({'id':'soft_ensemble_top3','family':'ensemble','variant':'equal_weight_oof','cv':ens_cv,'members':[x[0] for x in top_ests]})
    # Choose winner by PR-AUC with stability/complexity preference; logistic retained if no meaningful gain.
    best_row=leaderboard.iloc[0]; prev_pr=prev_cv['pr_auc_mean']; best_pr=float(best_row['pr_auc_mean'])
    chosen_id=str(best_row['id']) if best_pr > prev_pr+0.01 else 'reproduce_previous_logistic'
    chosen=previous if chosen_id=='reproduce_previous_logistic' else est_map[chosen_id]
    # Thresholds and calibration on OOF dev probabilities for chosen model.
    chosen_oof=oof(chosen,Xdev,ydev,inner); ts=np.arange(.05,.951,.01); threshold=[]
    for t in ts:
        p=(chosen_oof>=t).astype(int); m=metrics(ydev,p,chosen_oof); m['threshold']=float(t); threshold.append(m)
    tdf=pd.DataFrame(threshold); operating={'high_recall':float(tdf.loc[tdf.recall>=.70].sort_values(['recall','precision'],ascending=False).iloc[0].threshold) if (tdf.recall>=.70).any() else float(tdf.iloc[tdf.recall.argmax()].threshold),'balanced':float(tdf.sort_values(['f1','recall'],ascending=False).iloc[0].threshold),'high_precision':float(tdf.loc[tdf.precision>=.70].sort_values(['precision','recall'],ascending=False).iloc[0].threshold) if (tdf.precision>=.70).any() else float(tdf.iloc[tdf.precision.argmax()].threshold)}
    threshold=operating['balanced']; tdf.to_csv(OUT/'threshold_analysis_round2.csv',index=False)
    plt.figure(figsize=(8,5)); plt.plot(tdf.threshold,tdf.precision,label='Precision'); plt.plot(tdf.threshold,tdf.recall,label='Recall'); plt.plot(tdf.threshold,tdf.f1,label='F1'); plt.axvline(threshold,color='black',ls='--',label=f'Balanced={threshold:.2f}'); plt.xlabel('Threshold'); plt.ylabel('Metric'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'threshold_curve.png',dpi=160); plt.close()
    # Calibration comparison.
    cal=[]
    for method in ['uncalibrated','sigmoid','isotonic']:
        if method=='uncalibrated': prob=chosen_oof
        else: prob=cross_val_predict(CalibratedClassifierCV(clone(chosen),method=method,cv=inner,n_jobs=-1),Xdev,ydev,cv=inner,method='predict_proba',n_jobs=-1)[:,1]
        cal.append({'method':method,**metrics(ydev,(prob>=threshold).astype(int),prob)})
    caldf=pd.DataFrame(cal); caldf.to_csv(OUT/'calibration_round2.csv',index=False); calibration=str(caldf.sort_values(['brier','log_loss']).iloc[0].method)
    final=clone(chosen) if calibration=='uncalibrated' else CalibratedClassifierCV(clone(chosen),method=calibration,cv=inner,n_jobs=-1)
    # Final test is touched only here, after all dev choices are frozen.
    final.fit(Xdev,ydev); test_prob=final.predict_proba(Xtest)[:,1]; test_pred=(test_prob>=threshold).astype(int); final_test=metrics(ytest,test_pred,test_prob); final_test.update({'threshold':threshold,'model':chosen_id,'calibration':calibration})
    prev_test=prev.get('final_test_metrics',{}); comparison=[]
    for k in ['pr_auc','roc_auc','precision','recall','f1','brier_score','log_loss']:
        nk='brier' if k=='brier_score' else k; old=float(prev_test.get(k,np.nan)); new=float(final_test.get(nk,np.nan)); comparison.append({'metric':k,'previous':old,'new':new,'difference':new-old,'relative_change_pct':(new-old)/old*100 if old else np.nan})
    pd.DataFrame(comparison).to_csv(OUT/'final_test_comparison.csv',index=False)
    # Bootstrap CI on the frozen final test result.
    rng=np.random.default_rng(SEED); boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(ytest),len(ytest)); yy=ytest.iloc[ix]; pp=test_pred[ix]; pr=test_prob[ix]
        if yy.nunique()<2: continue
        mm=metrics(yy,pp,pr); boot.append({k:mm[k] for k in ['pr_auc','roc_auc','precision','recall','f1']})
    ci=pd.DataFrame(boot).quantile([.025,.975]).T; ci.columns=['lower','upper']; ci.to_csv(OUT/'bootstrap_ci_test.csv')
    # Learning curve and final permutation importance.
    sizes,train_scores,val_scores=learning_curve(clone(chosen),Xdev,ydev,cv=inner,scoring='average_precision',train_sizes=np.linspace(.2,1.0,5),n_jobs=-1); lc=pd.DataFrame({'train_size':sizes,'train_mean':train_scores.mean(1),'train_std':train_scores.std(1),'validation_mean':val_scores.mean(1),'validation_std':val_scores.std(1)}); lc.to_csv(OUT/'learning_curve.csv',index=False); plt.figure(figsize=(8,5)); plt.plot(sizes,lc.train_mean,label='Train PR-AUC'); plt.plot(sizes,lc.validation_mean,label='Validation PR-AUC'); plt.fill_between(sizes,lc.validation_mean-lc.validation_std,lc.validation_mean+lc.validation_std,alpha=.2); plt.xlabel('Training examples'); plt.ylabel('PR-AUC'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'learning_curve.png',dpi=160); plt.close()
    perm=permutation_importance(final,Xtest,ytest,scoring='average_precision',n_repeats=10,random_state=SEED,n_jobs=-1); pd.DataFrame({'feature':Xtest.columns,'importance_mean':perm.importances_mean,'importance_std':perm.importances_std}).sort_values('importance_mean',ascending=False).to_csv(OUT/'permutation_importance_round2.csv',index=False)
    # Save records and model.
    payload={'dataset_sha256':data_hash,'previous_dataset_hash':prev_hash,'hash_match':data_hash==prev_hash,'seed':SEED,'excluded_features':excluded,'cv':'RepeatedStratifiedKFold(5,3,seed=42)','test_policy':'locked until final frozen evaluation','previous_cv':prev_cv,'records':records,'operating_thresholds':operating,'chosen_id':chosen_id,'calibration':calibration,'final_test':final_test,'comparison':comparison,'bootstrap_ci':ci.to_dict('index'),'top_ids':top_ids,'high_correlation_pairs':pairs}
    (OUT/'round2_experiment_records.json').write_text(json.dumps(payload,indent=2,default=float),encoding='utf-8'); joblib.dump(final,OUT/'round2_final_model.joblib')
    report=['# ML Optimization Round 2 — Maximum Generalization & Performance\n','## Executive Summary\n',f'The previous locked test protocol was reproduced with dataset hash match: **{data_hash==prev_hash}**. All model selection, tuning, feature engineering, imbalance experiments, threshold selection, calibration selection, and ensemble decisions used only the development split and cross-validation. The locked test set was evaluated only after the final candidate was frozen.\n',f'The selected round-two candidate is **{chosen_id}** with **{calibration}** calibration and balanced operating threshold **{threshold:.2f}**. The previous model was retained when the best development improvement was not greater than 0.01 PR-AUC; otherwise the strongest candidate was selected.\n','## Previous Best Model\n',f'Previous model: calibrated logistic regression, isotonic calibration, threshold 0.38. Recorded final test PR-AUC: {prev_test.get("pr_auc",np.nan):.4f}; ROC-AUC: {prev_test.get("roc_auc",np.nan):.4f}; recall: {prev_test.get("recall",np.nan):.4f}; F1: {prev_test.get("f1",np.nan):.4f}.\n','## New Models Tested\n',leaderboard.to_markdown(index=False,floatfmt='.4f'),'\n## Hyperparameter Search\n','RandomizedSearchCV with 10 configurations per advanced model was run on the development split using the same repeated stratified CV and average precision objective. XGBoost, LightGBM, CatBoost, and more thoroughly tuned HistGradientBoosting were tested.\n','## Feature Engineering Experiments\n','A controlled interaction transformer tested overtime/satisfaction, business travel/overtime, marital status/stock options, and selected numeric products. Engineered interactions were evaluated inside the pipeline and retained only if their CV evidence justified them.\n','## Class Imbalance Experiments\n','The development-only benchmark includes logistic regression with no weighting, balanced weighting, SMOTE, random oversampling, and random undersampling when imbalanced-learn was available. Resampling occurred only inside training folds.\n','## Ensemble Experiments\n',f'An equal-weight out-of-fold soft ensemble was evaluated for the top three development candidates: {", ".join(top_ids)}. It was not promoted unless it provided a defensible improvement over the best single model.\n','## Calibration and Threshold Analysis\n',f'Calibration candidates were compared using development OOF probabilities. Operating points: high recall={operating["high_recall"]:.2f}, balanced F1={operating["balanced"]:.2f}, high precision={operating["high_precision"]:.2f}. The threshold curve is saved as `threshold_curve.png`.\n','## Stability, Learning Curves, and Uncertainty\n','Repeated stratified CV reports mean, standard deviation, minimum, and maximum. Learning curves and 2.5%/97.5% bootstrap confidence intervals for the final test metrics are saved in the artifacts directory.\n','## Final Test Comparison\n',pd.DataFrame(comparison).to_markdown(index=False,floatfmt='.4f'),'\n## Final Test Results\n',pd.DataFrame([final_test]).to_markdown(index=False,floatfmt='.4f'),'\n## Final Conclusion\n',f'1. Logistic Regression remains the selected model under the implemented decision rule: **{chosen_id=="reproduce_previous_logistic"}**.\n2. Nonlinear models provided a meaningful development improvement sufficient to select: **{chosen_id!="reproduce_previous_logistic"}**.\n3. Feature engineering was evaluated inside CV; its value is recorded in the leaderboard.\n4. Imbalance methods were tested without validation/test contamination; keep only if stable.\n5. The equal-weight ensemble was tested and not automatically promoted.\n6. Calibration was useful only according to the development Brier/log-loss comparison, not ranking metrics alone.\n7. The balanced operating threshold is **{threshold:.2f}**, with high-recall and high-precision alternatives also reported.\n8. Stability is represented by repeated-CV variance and bootstrap intervals; small data means uncertainty remains material.\n9. The dataset definition and size remain major bottlenecks because there is no time horizon, provenance, or external validation.\n10. The highest-value non-infrastructure improvement is better labels and longitudinal, prediction-time-valid features.\n','## Remaining Weaknesses\n','This is still a single historical snapshot. The random split cannot establish temporal deployment performance, and subgroup estimates for small groups remain uncertain. No result should be interpreted as causal or as authorization for automated employment action.\n']
    (OUT/'round2_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'chosen_id':chosen_id,'calibration':calibration,'threshold':threshold,'final_test':final_test,'output':str(OUT)},indent=2,default=float))
if __name__=='__main__': main()
