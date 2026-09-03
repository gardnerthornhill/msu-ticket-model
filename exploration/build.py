import csv, json, statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd, numpy as np

m={'UMass':'Massachusetts','Southeastern Louisiana':'SE Louisiana'}
CT=ZoneInfo('America/Chicago')
rows=list(csv.DictReader(open('tickets.csv')))
games={y:json.load(open(f'games_{y}.json')) for y in [2023,2024,2025]}
lines={y:{g['id']:g for g in json.load(open(f'lines_{y}.json'))} for y in [2023,2024,2025]}
media={y:{g['id']:g for g in json.load(open(f'media_{y}.json'))} for y in [2023,2024,2025]}
sp={y:{t['team']:t for t in json.load(open(f'sp_{y}.json'))} for y in [2023,2024,2025]}
talent={y:{t['team']:t['talent'] for t in json.load(open(f'talent_{y}.json'))} for y in [2023,2024,2025]}
rank={}
for y in [2023,2024,2025]:
    for wk in json.load(open(f'rankings_{y}.json')):
        if wk['seasonType']!='regular': continue
        for p in wk['polls']:
            if p['poll']=='AP Top 25':
                rank[(y,wk['week'])]={r['school']:r['rank'] for r in p['ranks']}
def record_before(glist, team, date):
    w=l=0
    for g in glist:
        if not g['completed'] or g['startDate']>=date: continue
        if g['homeTeam']==team: w+= g['homePoints']>g['awayPoints']; l+= g['homePoints']<g['awayPoints']
        elif g['awayTeam']==team: w+= g['awayPoints']>g['homePoints']; l+= g['awayPoints']<g['homePoints']
    return w,l
tv_tier={'ABC':3,'CBS':3,'ESPN':3,'FOX':3,'NBC':3,'ESPN2':2,'ESPNU':1,'SEC Network':1,'ESPN+':0,'SECN+':0,'SEC Network+':0}
out=[]
for r in rows:
    y=int(r['date'][:4]); opp=m.get(r['opponent'],r['opponent'])
    g=next(g for g in games[y] if g['homeTeam']=='Mississippi State' and g['awayTeam']==opp)
    dt=datetime.fromisoformat(g['startDate'].replace('Z','+00:00')).astimezone(CT)
    oppg=json.load(open(f"opp/{y}_{opp.replace(' ','_')}.json"))
    ow,ol=record_before(oppg,opp,g['startDate']); mw,ml=record_before(games[y],'Mississippi State',g['startDate'])
    ln=lines[y].get(g['id'],{}).get('lines',[])
    spreads=[x['spread'] for x in ln if x.get('spread') is not None]; ous=[x['overUnder'] for x in ln if x.get('overUnder') is not None]
    rk=rank.get((y,g['week']),{})
    outlet=media[y].get(g['id'],{}).get('outlet')
    out.append(dict(season=y,week=g['week'],date=dt.strftime('%Y-%m-%d'),opponent=r['opponent'],
        getin=float(r['getin']) if r['getin'] else np.nan, change3d=float(r['change3d']) if r['change3d'] else np.nan,
        attendance=g['attendance'], kickoff_hr=dt.hour+dt.minute/60, night=int(dt.hour>=18), weekday=dt.strftime('%a'),
        conf_game=int(g['conferenceGame']), opp_conf=g['awayConference'], opp_fbs=int(g['awayClassification']=='fbs'),
        opp_p4=int(g['awayConference'] in ('SEC','Big Ten','Big 12','ACC','Pac-12')),
        msu_elo=g['homePregameElo'], opp_elo=g['awayPregameElo'], 
        opp_sp=sp[y].get(opp,{}).get('rating'), msu_sp=sp[y]['Mississippi State']['rating'],
        opp_talent=talent[y].get(opp), opp_ap=rk.get(opp), msu_ap=rk.get('Mississippi State'),
        opp_ranked=int(opp in rk), opp_w=ow, opp_l=ol, msu_w=mw, msu_l=ml,
        spread=statistics.median(spreads) if spreads else np.nan, ou=statistics.median(ous) if ous else np.nan,
        tv=outlet, tv_tier=tv_tier.get(outlet), ei=g['excitementIndex'], msu_pts=g['homePoints'], opp_pts=g['awayPoints'],
        margin=g['homePoints']-g['awayPoints']))
df=pd.DataFrame(out)
df['opp_elo']=df['opp_elo'].fillna(df['opp_elo'].min()-100)  # FCS opps have no Elo; treat as weakest
df['elo_diff']=df['opp_elo']-df['msu_elo']
df['opp_winpct']=df.opp_w/(df.opp_w+df.opp_l).replace(0,np.nan)
df['msu_winpct']=df.msu_w/(df.msu_w+df.msu_l).replace(0,np.nan)
df['log_getin']=np.log(df.getin)
df.to_csv('joined.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(df[['season','date','opponent','getin','change3d','attendance','kickoff_hr','weekday','conf_game','opp_elo','msu_elo','opp_sp','opp_ap','msu_ap','opp_w','opp_l','msu_w','msu_l','spread','ou','tv','opp_talent','ei','margin']].to_string())
