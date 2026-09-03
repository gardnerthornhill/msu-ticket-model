import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
df=pd.read_csv('joined.csv')
df.loc[df.tv=='SECN','tv_tier']=1
df['tv_tier']=df.tv_tier.astype(float)
df['opp_ap_num']=df.opp_ap.fillna(30)   # unranked = 30
df['opp_ranked']=df.opp_ap.notna().astype(int)
df['sellout']=(df.attendance>=60000).astype(int)
df['late_season']=(df.week>=10).astype(int)
df['msu_games_played']=df.msu_w+df.msu_l
both=df.dropna(subset=['getin','attendance'])
print(f"n with attendance={df.attendance.notna().sum()}, with price={df.getin.notna().sum()}, both={len(both)}")

def corr(x,y):
    d=pd.concat([x,y],axis=1).dropna(); 
    if len(d)<5: return (np.nan,np.nan,np.nan,np.nan,len(d))
    r,p=stats.pearsonr(d.iloc[:,0],d.iloc[:,1]); rs,ps=stats.spearmanr(d.iloc[:,0],d.iloc[:,1]); return (r,p,rs,ps,len(d))

print("\n=== 1. GET-IN PRICE vs ATTENDANCE (n=16) ===")
for lab,x in [('price',both.getin),('log price',both.log_getin)]:
    r,p,rs,ps,n=corr(x,both.attendance); print(f"{lab:10s} Pearson r={r:.3f} (p={p:.3f})  Spearman rho={rs:.3f} (p={ps:.3f})")
print("\nWithin-season Spearman (price vs attendance):")
for s,g in both.groupby('season'):
    if len(g)>=5: rs,ps=stats.spearmanr(g.getin,g.attendance); print(f"  {s}: rho={rs:.3f} p={ps:.3f} n={len(g)}")
print("\nExcluding sellouts (attendance>=60k):")
ns=both[both.sellout==0]; r,p,rs,ps,n=corr(ns.getin,ns.attendance); print(f"  Pearson r={r:.3f} p={p:.3f}; Spearman rho={rs:.3f} p={ps:.3f}; n={n}")

vars_=['getin','log_getin','change3d','opp_elo','elo_diff','msu_elo','opp_sp','opp_talent','opp_ap_num','opp_ranked','opp_winpct','msu_winpct','msu_w','msu_l','conf_game','opp_p4','opp_fbs','spread','ou','kickoff_hr','night','week','late_season','tv_tier','ei','margin']
print("\n=== 2. UNIVARIATE CORRELATIONS WITH ATTENDANCE (all games w/ attendance, n<=21) ===")
print(f"{'variable':14s} {'pearson':>8s} {'p':>6s} {'spearman':>9s} {'p':>6s} {'n':>3s}")
for v in vars_:
    if v=='attendance': continue
    r,p,rs,ps,n=corr(df[v],df.attendance); print(f"{v:14s} {r:8.3f} {p:6.3f} {rs:9.3f} {ps:6.3f} {n:3d}")
print("\n=== 3. UNIVARIATE CORRELATIONS WITH GET-IN PRICE (n<=17) ===")
print(f"{'variable':14s} {'pearson':>8s} {'p':>6s} {'spearman':>9s} {'p':>6s} {'n':>3s}")
for v in vars_:
    if v in('getin','log_getin'): continue
    r,p,rs,ps,n=corr(df[v],df.getin); print(f"{v:14s} {r:8.3f} {p:6.3f} {rs:9.3f} {ps:6.3f} {n:3d}")

print("\n=== 4. DOES PRICE ADD PREDICTIVE VALUE FOR ATTENDANCE? Leave-one-out CV (n=16) ===")
def loo(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float); preds=[]
    for i in range(len(y)):
        msk=np.arange(len(y))!=i; Xi=sm.add_constant(X[msk]) if X.ndim>1 and X.shape[1]>0 else np.ones((msk.sum(),1))
        b=np.linalg.lstsq(Xi,y[msk],rcond=None)[0]
        xt=np.r_[1,X[i]] if X.ndim>1 and X.shape[1]>0 else np.array([1.0])
        preds.append(xt@b)
    preds=np.array(preds); rmse=np.sqrt(np.mean((preds-y)**2)); r2=1-np.sum((preds-y)**2)/np.sum((y-y.mean())**2); return rmse,r2
y=both.attendance.values
specs={'mean only':[], 'price':['getin'], 'log price':['log_getin'], 'opp Elo':['opp_elo'], 'spread':['spread'], 'opp AP rank':['opp_ap_num'], 'conf game':['conf_game'],
       'opp Elo + log price':['opp_elo','log_getin'], 'spread + log price':['spread','log_getin'], 'conf + log price':['conf_game','log_getin'], 'conf + opp Elo':['conf_game','opp_elo']}
for k,cols in specs.items():
    X=both[cols].values if cols else np.zeros((len(both),0)); rmse,r2=loo(X,y); print(f"{k:22s} LOO-RMSE={rmse:7.0f}  LOO-R2={r2:6.3f}")

print("\n=== 5. OLS: attendance ~ log price + opp Elo (in-sample) ===")
X=sm.add_constant(both[['log_getin','opp_elo']]); m=sm.OLS(both.attendance,X).fit(); print(m.summary().tables[1]); print(f"R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")
print("\n=== 5b. OLS: attendance ~ log price + conf_game ===")
X=sm.add_constant(both[['log_getin','conf_game']]); m=sm.OLS(both.attendance,X).fit(); print(m.summary().tables[1]); print(f"R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")
print("\n=== 5c. Partial corr of log price & attendance controlling opp Elo ===")
def partial(x,y,z):
    rx=sm.OLS(x,sm.add_constant(z)).fit().resid; ry=sm.OLS(y,sm.add_constant(z)).fit().resid; return stats.pearsonr(rx,ry)
print(partial(both.log_getin,both.attendance,both[['opp_elo']]))
print("controlling conf_game:",partial(both.log_getin,both.attendance,both[['conf_game']]))
print("controlling opp_elo+season:",partial(both.log_getin,both.attendance,pd.get_dummies(both[['opp_elo','season']],columns=['season'],drop_first=True).astype(float)))

print("\n=== 6. WHAT DRIVES PRICE? OLS log price ~ opp Elo + week + season(2025 dummy) (n=17) ===")
p=df.dropna(subset=['getin']).copy(); p['s2025']=(p.season==2025).astype(int)
X=sm.add_constant(p[['opp_elo','week','s2025']]); m=sm.OLS(p.log_getin,X).fit(); print(m.summary().tables[1]); print(f"R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")
X=sm.add_constant(p[['opp_ap_num','late_season','s2025']]); m=sm.OLS(p.log_getin,X).fit(); print(m.summary().tables[1]); print(f"R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")

print("\n=== 7. Price residual vs attendance: is a game 'over/under-priced' relative to opponent quality informative? ===")
X=sm.add_constant(both[['opp_elo']]); resid=sm.OLS(both.log_getin,X).fit().resid
print("Spearman(price residual, attendance):",stats.spearmanr(resid,both.attendance))
X=sm.add_constant(both[['opp_elo']]); aresid=sm.OLS(both.attendance,X).fit().resid
print("Spearman(price residual, attendance residual):",stats.spearmanr(resid,aresid))
print(pd.DataFrame({'opp':both.opponent.values,'season':both.season.values,'price':both.getin.values,'att':both.attendance.values,'price_resid':resid.round(2).values,'att_resid':aresid.round(0).values}).sort_values('price_resid').to_string(index=False))

print("\n=== 8. Group means ===")
print(df.groupby('conf_game')[['getin','attendance']].agg(['mean','median','count']).round(0))
print(df.groupby('opp_ranked')[['getin','attendance']].agg(['mean','median','count']).round(0))
print(df.groupby('season')[['getin','attendance']].agg(['mean','median','count']).round(0))
print(df.groupby('night')[['getin','attendance']].agg(['mean','median','count']).round(0))
print("\nMSU record entering vs attendance:", stats.spearmanr(df.dropna(subset=['attendance']).msu_winpct.fillna(0.5), df.dropna(subset=['attendance']).attendance))
