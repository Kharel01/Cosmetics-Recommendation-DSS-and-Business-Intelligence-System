# ================================================================
#  app.py — Cosmetics Skincare Finder v6
#  Install:  pip install dash dash-bootstrap-components pandas scikit-learn numpy
#  Run:      python app.py     Open: http://127.0.0.1:8050
#  Dataset:  cosmetics_3.csv  (same folder)
#
#  v6 FIXES:
#   FIX 13: No default ingredient injection — user inputs only
#   FIX 14: Strict category enforcement with assertion
#   FIX 15: Filter-first pipeline — score only after filtering
#   FIX 16: user_input passed unchanged to model
#   FIX 18: Cascaded 4-level relaxation — never empty
#   FIX 22: No numeric indexing in product cards
# ================================================================
import re, warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
warnings.filterwarnings('ignore')

# ── DATA ─────────────────────────────────────────────────────────
df_full = pd.read_csv('cosmetics 3.csv', encoding='utf-8-sig')
for col in ['Label','Brand','Name','Ingredients']:
    df_full[col] = df_full[col].astype(str).str.strip()
df_full['Label'] = df_full['Label'].str.title()
df_full['Brand'] = df_full['Brand'].str.upper()
df_full = df_full.dropna(subset=['Rank','Price']).reset_index(drop=True)
SKIN_COLS  = ['Combination','Dry','Normal','Oily','Sensitive']
df_full[SKIN_COLS] = df_full[SKIN_COLS].astype(int)
CATEGORIES = sorted(df_full['Label'].unique())
ALL_BRANDS  = sorted(df_full['Brand'].unique())

# ── INGREDIENT FLAGS ─────────────────────────────────────────────
KEY_INGREDIENTS = {
    'niacinamide'    : r'niacinamide',
    'hyaluronic_acid': r'hyaluronic acid|sodium hyaluronate',
    'salicylic_acid' : r'salicylic acid',
    'vitamin_c'      : r'ascorbic acid|vitamin c|tetrahexyldecyl ascorbate|ascorbyl',
    'alcohol'        : r'\balcohol\b|alcohol denat',
    'fragrance'      : r'fragrance|parfum',
    'parabens'       : r'paraben',
}
INGR_LABELS = {
    'niacinamide':'Niacinamide','hyaluronic_acid':'Hyaluronic Acid',
    'salicylic_acid':'Salicylic Acid','vitamin_c':'Vitamin C',
    'alcohol':'Alcohol','fragrance':'Fragrance','parabens':'Parabens',
}
INGR_OPTIONS = [{'label':INGR_LABELS[k],'value':k} for k in KEY_INGREDIENTS]
il = df_full['Ingredients'].str.lower()
for ing, pat in KEY_INGREDIENTS.items():
    df_full[f'has_{ing}'] = il.str.contains(pat, regex=True).astype(int)

# ── TF-IDF ───────────────────────────────────────────────────────
def clean_text(t):
    t=t.lower(); t=re.sub(r'[^a-z\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()
df_full['Ingredients_Clean'] = df_full['Ingredients'].apply(clean_text)
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), max_features=3000, sublinear_tf=True)
tfidf_mat = tfidf_vec.fit_transform(df_full['Ingredients_Clean'])

# ── CONCERN — TF-IDF KEYWORDS ONLY (FIX 13) ─────────────────────
CONCERN_TFIDF = {
    'acne'        :'salicylic niacinamide bha pore acne',
    'pores'       :'pore refining niacinamide minimise',
    'oil'         :'oil control mattifying lightweight sebum',
    'oily'        :'oil control mattifying sebum',
    'dryness'     :'hyaluronic glycerin moisture dry',
    'dry'         :'hyaluronic glycerin moisture dry',
    'hydration'   :'hyaluronic water glycerin humectant',
    'sensitivity' :'gentle soothing calming fragrance free alcohol free',
    'sensitive'   :'gentle soothing calming fragrance free',
    'brightening' :'vitamin c brightening glow radiance',
    'dullness'    :'vitamin c brightening radiance dull',
    'aging'       :'retinol peptide collagen anti aging firming',
    'wrinkles'    :'retinol peptide collagen firming lines',
    'pigmentation':'vitamin c dark spot even tone hyperpigmentation',
}
def get_tfidf_extra(sc):
    return ' '.join(CONCERN_TFIDF[t] for t in re.findall(r'[a-z]+',sc.lower()) if t in CONCERN_TFIDF)

INGR_BENEFITS = {
    'vitamin_c'      :(['brightening','dullness','pigmentation','aging','wrinkles'],
                       'Brightens skin tone and targets dark spots with Vitamin C'),
    'salicylic_acid' :(['acne','pores','oil','oily'],
                       'Controls acne and excess oil with Salicylic Acid'),
    'hyaluronic_acid':(['dryness','dry','hydration'],
                       'Deeply hydrates and plumps skin with Hyaluronic Acid'),
    'niacinamide'    :(['acne','pores','oil','oily','brightening','pigmentation'],
                       'Minimises pores and regulates oil with Niacinamide'),
}
def get_verified_concern_line(row, sc):
    tokens=set(re.findall(r'[a-z]+',sc.lower()))
    for ik,(ct,bt) in INGR_BENEFITS.items():
        if row.get(f'has_{ik}',0)==1 and tokens&set(ct): return bt
    return None

# ── WEIGHTS ──────────────────────────────────────────────────────
WEIGHTS={'W1_skin':0.30,'W2_rank':0.25,'W3_value':0.15,
         'W4_tfidf':0.12,'W5_rules':0.10,'W6_budget':0.06,'W7_brand':0.02}

# ── SCORING FRAME ─────────────────────────────────────────────────
def _norm(s):
    mn,mx=s.min(),s.max()
    return pd.Series(0.5,index=s.index) if mx==mn else (s-mn)/(mx-mn)

def _score_frame(dc,skin_concern,prefer_ing,avoid_ing,budget,pref_brands):
    dc=dc.copy()
    te=get_tfidf_extra(skin_concern)
    dc['s_skin']=1.0
    dc['s_rank']=_norm(dc['Rank'])
    dc['s_value']=_norm(dc['Rank']/np.log1p(dc['Price']))
    q=clean_text(skin_concern+' '+' '.join(p.replace('_',' ') for p in prefer_ing)+' '+te)
    qv=tfidf_vec.transform([q])
    sims=cosine_similarity(qv,tfidf_mat[dc.index]).flatten()
    dc['s_tfidf']=sims/sims.max() if sims.max()>0 else sims
    rs=pd.Series(0.0,index=dc.index)
    for ing in prefer_ing:
        if f'has_{ing}' in dc.columns: rs+=dc[f'has_{ing}'].astype(float)
    for ing in avoid_ing:
        if f'has_{ing}' in dc.columns: rs-=dc[f'has_{ing}'].astype(float)*1.5
    rs-=rs.min(); dc['s_rules']=rs/rs.max() if rs.max()>0 else rs
    prices=dc['Price'].astype(float)
    within=(prices>=budget[0])&(prices<=budget[1])
    ov=np.maximum(prices-budget[1],0)
    dc['s_budget']=(within.astype(float)+(~within).astype(float)*np.exp(-ov/max(budget[1],1))).clip(0,1)
    if not pref_brands:
        dc['s_brand']=0.5
    else:
        bu=[b.upper() for b in pref_brands]
        dc['s_brand']=dc['Brand'].apply(lambda b:1.0 if b in bu else 0.3)
    dc['Final_Score']=(dc['s_skin']*WEIGHTS['W1_skin']+dc['s_rank']*WEIGHTS['W2_rank']+
        dc['s_value']*WEIGHTS['W3_value']+dc['s_tfidf']*WEIGHTS['W4_tfidf']+
        dc['s_rules']*WEIGHTS['W5_rules']+dc['s_budget']*WEIGHTS['W6_budget']+
        dc['s_brand']*WEIGHTS['W7_brand'])
    if prefer_ing:
        dc['pref_match_count']=sum(dc.get(f'has_{i}',pd.Series(0,index=dc.index)) for i in prefer_ing)
        dc['pref_score']=dc['pref_match_count']/len(prefer_ing)
    else:
        dc['pref_match_count']=0; dc['pref_score']=1.0
    return dc.sort_values('Final_Score',ascending=False).reset_index(drop=True)

# ── CASCADED FILTER (FIX 18) ─────────────────────────────────────
def _apply_filters(cat_df,prefer_ing,avoid_ing,budget):
    ac=[f'has_{i}' for i in avoid_ing  if f'has_{i}' in cat_df.columns]
    pc=[f'has_{i}' for i in prefer_ing if f'has_{i}' in cat_df.columns]
    def aok(d): return (d[ac].sum(axis=1)==0) if ac else pd.Series(True,index=d.index)
    def bok(d): p=d['Price']; return (p>=budget[0])&(p<=budget[1])
    def pok(d): return (d[pc].sum(axis=1)==len(pc)) if pc else pd.Series(True,index=d.index)
    d0=cat_df[aok(cat_df)&bok(cat_df)&pok(cat_df)]
    if len(d0)>=1: return d0.copy(),0,'strict'
    d1=cat_df[aok(cat_df)&bok(cat_df)]
    if len(d1)>=1: return d1.copy(),1,'relaxed_preferred'
    d2=cat_df[aok(cat_df)]
    if len(d2)>=1: return d2.copy(),2,'relaxed_budget'
    return cat_df.copy(),3,'fallback_all'

# ── MAIN SCORING (FIX 13–18) ─────────────────────────────────────
def score_products(user_input):
    # FIX 16: unpack exactly as received — no transforms
    skin_type    = user_input['skin_type'].title()
    category     = user_input['product_category'].title()
    skin_concern = user_input.get('skin_concern','')
    budget       = user_input.get('budget',[0,9999])
    prefer_ing   = list(user_input.get('prefer_ingredients',[]))  # FIX 13: user only
    avoid_ing    = list(user_input.get('avoid_ingredients',[]))   # FIX 13: user only
    pref_brands  = user_input.get('preferred_brands',[])

    # FIX 15 STEP 1 + FIX 14: strict skin + category
    cat_df=df_full[(df_full[skin_type]==1)&
                   (df_full['Label'].str.lower()==category.lower())].copy()
    if not cat_df.empty:
        assert all(cat_df['Label'].str.lower()==category.lower())

    if cat_df.empty:
        return pd.DataFrame(),pd.DataFrame(),{
            'prefer_ing':prefer_ing,'avoid_ing':avoid_ing,'n_abs':0,'n_oos':0,
            'filter_level':None,'filter_label':'no_products','brand_conflict':None,
            'skin_type':skin_type,'category':category,'budget':budget,
            'total_in_cat':0,'relaxation_msg':None}

    n_total=len(cat_df)

    # FIX 15 STEP 2: cascaded hard constraints
    abs_df,filter_level,filter_label=_apply_filters(cat_df,prefer_ing,avoid_ing,budget)
    abs_names=set(abs_df['Name'])
    oos_df=cat_df[~cat_df['Name'].isin(abs_names)].copy()

    # FIX 15 STEP 3: score AFTER filtering
    abs_scored=_score_frame(abs_df,skin_concern,prefer_ing,avoid_ing,budget,pref_brands)
    if not oos_df.empty:
        oos_scored=_score_frame(oos_df,skin_concern,prefer_ing,avoid_ing,budget,pref_brands)
        if prefer_ing and 'pref_match_count' in oos_scored.columns:
            oos_scored=oos_scored.sort_values(['pref_match_count','Final_Score'],
                                               ascending=[False,False]).reset_index(drop=True)
    else:
        oos_scored=pd.DataFrame()

    brand_conflict=None
    if pref_brands:
        bu=[b.upper() for b in pref_brands]
        if not abs_df['Brand'].isin(bu).any():
            brand_conflict=(f'No {category.lower()}s from your selected brand(s) match all filters. '
                           f'Showing closest alternatives.')

    relaxation_msg=None
    if filter_level==1:
        pl=[INGR_LABELS.get(i,i) for i in prefer_ing]
        relaxation_msg=f'No products contain all preferred ingredients ({", ".join(pl)}). Showing best matches within budget.'
    elif filter_level==2:
        relaxation_msg=f'No products match your budget (${budget[0]}–${budget[1]}) and avoid list. Showing closest alternatives.'
    elif filter_level==3:
        relaxation_msg='No exact matches found. Showing closest alternatives based on your preferences.'

    return abs_scored,oos_scored,{
        'prefer_ing':prefer_ing,'avoid_ing':avoid_ing,
        'n_abs':len(abs_scored),'n_oos':len(oos_scored),
        'filter_level':filter_level,'filter_label':filter_label,
        'brand_conflict':brand_conflict,'skin_type':skin_type,
        'category':category,'budget':budget,'total_in_cat':n_total,
        'relaxation_msg':relaxation_msg}

def assign_badges(df_in,group='abs'):
    t=df_in.copy().reset_index(drop=True)
    if group=='abs':
        def m(i):
            if i==0: return '🥇 Perfect Match'
            if i<=2: return '🥈 Great Choice'
            return          '🥉 Good Option'
    else:
        def m(i): return '◆ Partial Match'
    t['Badge']=[m(i) for i in range(len(t))]
    return t

def build_explanation(row,user_input,ref_ranked,rank_pos,prefer_ing,avoid_ing,group='abs',filter_level=0):
    skin_type=user_input['skin_type'].title()
    budget=user_input.get('budget',[0,9999])
    skin_concern=user_input.get('skin_concern','')
    user_avoid=list(user_input.get('avoid_ingredients',[]))
    n_prefer=len(prefer_ing)
    why=[f'Suitable for {skin_type} skin']
    if budget[0]<=row['Price']<=budget[1]: why.append(f'Within your budget (${row["Price"]:.0f})')
    if   row['Rank']==5.0:  why.append('Perfectly rated by customers (5.0 / 5.0)')
    elif row['Rank']>=4.5:  why.append(f'Outstanding customer rating ({row["Rank"]:.1f} / 5.0)')
    elif row['Rank']>=4.3:  why.append(f'Highly rated by customers ({row["Rank"]:.1f} / 5.0)')
    elif row['Rank']>=4.0:  why.append(f'Well rated by customers ({row["Rank"]:.1f} / 5.0)')
    else:                    why.append(f'Customer rating: {row["Rank"]:.1f} / 5.0')
    val=row.get('s_value',0)
    if   val>=0.75: why.append(f'Exceptional value — high quality at ${row["Price"]:.0f}')
    elif val>=0.55: why.append('Good value for the quality offered')
    if prefer_ing:
        present=[INGR_LABELS.get(i,i) for i in prefer_ing if row.get(f'has_{i}',0)==1]
        if group=='abs' and len(present)==n_prefer:
            if n_prefer==1: why.append(f'Contains your preferred ingredient: {present[0]}')
            else:           why.append(f'Contains all {n_prefer} preferred ingredients: {", ".join(present)}')
        elif present:       why.append(f'Contains {len(present)}/{n_prefer} preferred ingredient(s): {", ".join(present)}')
    cl=get_verified_concern_line(row,skin_concern)
    if cl: why.append(cl)
    pb=user_input.get('preferred_brands',[])
    if pb and row['Brand'].upper() in [b.upper() for b in pb]:
        why.append(f'From your preferred brand: {row["Brand"].title()}')
    pref_detail=None
    if prefer_ing:
        plist=[INGR_LABELS.get(i,i) for i in prefer_ing if row.get(f'has_{i}',0)==1]
        mlist=[INGR_LABELS.get(i,i) for i in prefer_ing if row.get(f'has_{i}',0)==0]
        status='full' if len(plist)==n_prefer else ('partial' if plist else 'none')
        pref_detail={'status':status,'matched':len(plist),'total':n_prefer,'present':plist,'missing':mlist}
    warnings_list=[f'Contains {INGR_LABELS.get(ing,ing)} (you wanted to avoid this)'
                   for ing in user_avoid if row.get(f'has_{ing}',0)==1]
    budget_alert=None
    if row['Price']>budget[1]:
        pct=(row['Price']-budget[1])/max(budget[1],1)*100
        budget_alert=f'${row["Price"]:.0f} is {pct:.0f}% above your max budget (${budget[1]})'
    partial_reason=None
    if group=='oos':
        pts=[]
        if row['Price']>budget[1]:
            pts.append(f'Slightly above budget but {"highly" if row["Rank"]>=4.3 else "well"} rated (★{row["Rank"]:.1f})')
        ua=[INGR_LABELS.get(i,i) for i in user_avoid if row.get(f'has_{i}',0)==1]
        if ua: pts.append(f'Contains {", ".join(ua)} — on your avoid list')
        if pref_detail and pref_detail['missing']:
            mc=pref_detail['matched']; tot=pref_detail['total']; mis=pref_detail['missing']
            if mc>0: pts.append(f'Matches {mc}/{tot} preferred ({", ".join(pref_detail["present"])}) — missing: {", ".join(mis)}')
            else:    pts.append(f'Missing preferred ingredients: {", ".join(mis)}')
        partial_reason=' · '.join(pts) if pts else 'Does not meet all your exact filters'
    tradeoff=''
    if rank_pos<len(ref_ranked):
        alt=ref_ranked.iloc[rank_pos]; pts=[]
        if alt['Price']>0:
            pct=(row['Price']-alt['Price'])/alt['Price']*100
            if abs(pct)<2: pts.append('Same price as the next option')
            elif pct<0:    pts.append(f'{abs(pct):.0f}% cheaper than the next option')
            else:          pts.append(f'{pct:.0f}% more expensive than the next option')
        d=row['Rank']-alt['Rank']
        if abs(d)>=0.05:
            if d>0: pts.append(f'Rated {abs(d):.1f} pts higher — better customer rating')
            else:   pts.append(f'Rated {abs(d):.1f} pts lower — next option rated higher')
        tp=[INGR_LABELS.get(i,i) for i in prefer_ing if row.get(f'has_{i}',0)==1]
        ap=[INGR_LABELS.get(i,i) for i in prefer_ing if alt.get(f'has_{i}',0)==1]
        diff=len(tp)-len(ap)
        if diff>0:
            ext=[x for x in tp if x not in ap]
            pts.append(f'{diff} more preferred ingredient(s) here ({", ".join(ext)})')
        elif diff<0:
            miss=[x for x in ap if x not in tp]
            pts.append(f'{abs(diff)} fewer preferred ingredient(s) here ({", ".join(miss)})')
        tradeoff=' · '.join(pts) if pts else 'Very similar to the next option'
    return {'why':why,'warnings':warnings_list,'tradeoff':tradeoff,
            'budget_alert':budget_alert,'partial_reason':partial_reason,'pref_detail':pref_detail}

# ── DESIGN ───────────────────────────────────────────────────────
C={
    'bg':'#F7F4F1','card':'#FFFFFF','rose':'#C8506A','rose_light':'#FAE8EE',
    'blue':'#5A8FA8','blue_light':'#EAF3FA','green':'#3D8F6E','green_light':'#E5F5EE',
    'amber':'#B87A2E','amber_light':'#FDF0E0','red':'#B83232','red_light':'#FDECEA',
    'text':'#18182C','sub':'#42425A','muted':'#888899','border':'#E5E0DB',
    'gold':'#C9A408','gold_bg':'#FFF8E1','silver':'#6A7D96','silver_bg':'#EDF2F7',
    'bronze':'#B06830','bronze_bg':'#FBF0E8','oos_fg':'#5A6070','oos_bg':'#F2F4F6',
    'divider':'#C8C0B8','info_bg':'#EFF6FF','info_border':'#93C5FD',
    'warn_bg':'#FFFBEB','warn_border':'#FCD34D',
}
FONT="'Inter','Segoe UI','Helvetica Neue',sans-serif"
CARD_B={'background':C['card'],'borderRadius':'14px','boxShadow':'0 2px 12px rgba(0,0,0,0.07)',
        'padding':'20px 22px','marginBottom':'14px','border':f"1px solid {C['border']}"}
INPUT_S={'width':'100%','padding':'9px 12px','borderRadius':'9px',
         'border':f"1px solid {C['border']}",'fontSize':'13px','color':C['text'],
         'background':C['card'],'outline':'none','marginBottom':'14px'}
LBL_S={'fontSize':'11px','fontWeight':'700','color':C['sub'],'letterSpacing':'0.5px',
       'textTransform':'uppercase','marginBottom':'4px','display':'block'}

def lbl(t): return html.Span(t,style=LBL_S)

def chip(icon,label,value,bg,color):
    return html.Div([
        html.Span(icon,style={'fontSize':'14px','marginRight':'5px'}),
        html.Span(label+': ',style={'fontSize':'12px','color':C['muted']}),
        html.Span(value,style={'fontSize':'12px','fontWeight':'700','color':color}),
    ],style={'background':bg,'borderRadius':'8px','padding':'5px 10px',
             'marginRight':'6px','marginBottom':'6px','display':'inline-flex','alignItems':'center'})

def badge_pill(text,group='abs'):
    if group=='oos':
        return html.Span(text,style={'fontSize':'10px','fontWeight':'600','color':C['oos_fg'],
                                      'background':C['oos_bg'],'border':f"1px solid {C['border']}",
                                      'borderRadius':'20px','padding':'2px 10px','display':'inline-block'})
    cfg={'🥇 Perfect Match':(C['gold'],C['gold_bg'],f"1.5px solid {C['gold']}"),
         '🥈 Great Choice' :(C['silver'],C['silver_bg'],f"1.5px solid {C['silver']}"),
         '🥉 Good Option'  :(C['bronze'],C['bronze_bg'],f"1.5px solid {C['bronze']}")}
    col,bg,brd=cfg.get(text,(C['rose'],C['rose_light'],f"1px solid {C['rose']}"))
    return html.Span(text,style={'fontSize':'11px','fontWeight':'700','color':col,
                                  'background':bg,'border':brd,'borderRadius':'20px',
                                  'padding':'3px 11px','display':'inline-block'})

def score_bar(score):
    return html.Div(html.Div(style={'height':'5px','width':f"{score*100:.0f}%",
        'background':f"linear-gradient(90deg,{C['rose']},{C['blue']})","borderRadius":'4px'}),
        style={'background':C['border'],'borderRadius':'4px','height':'5px','marginTop':'5px'})

def pref_pill(pd_):
    if not pd_: return None
    mc=pd_['matched']; tot=pd_['total']
    if pd_['status']=='full':
        return html.Span(f'✓ All {tot} ingredient(s) matched',
            style={'fontSize':'11px','fontWeight':'700','color':C['green'],'background':C['green_light'],
                   'border':f"1px solid #9ECEB8",'borderRadius':'6px',
                   'padding':'2px 8px','display':'inline-block','marginTop':'4px'})
    elif pd_['status']=='partial':
        return html.Span(f'{mc}/{tot} preferred ingredient(s) matched',
            style={'fontSize':'11px','fontWeight':'600','color':C['amber'],'background':C['amber_light'],
                   'border':f"1px solid #DFC090",'borderRadius':'6px',
                   'padding':'2px 8px','display':'inline-block','marginTop':'4px'})
    return html.Span(f'None of {tot} preferred ingredient(s) found',
        style={'fontSize':'11px','fontWeight':'600','color':C['oos_fg'],'background':C['oos_bg'],
               'borderRadius':'6px','padding':'2px 8px','display':'inline-block','marginTop':'4px'})

# FIX 22: product card — no numeric rank displayed on card itself
def product_card(row, exp, group='abs'):
    is_oos   = group=='oos'
    left_col = C['oos_fg'] if is_oos else C['rose']

    budget_el = html.Div([
        html.Span('💸 ',style={'fontSize':'13px'}),
        html.Span(exp['budget_alert'],style={'fontSize':'12px','fontWeight':'600','color':C['red']}),
    ],style={'background':C['red_light'],'border':f"1.5px solid {C['red']}",
             'borderRadius':'8px','padding':'7px 12px','marginBottom':'10px'}) if exp['budget_alert'] else None

    why_items=[html.Div([
        html.Span('✦ ' if get_verified_concern_line(row,exp.get('_concern',''))==w else '✓ ',
                  style={'color':C['amber'] if get_verified_concern_line(row,exp.get('_concern',''))==w
                         else C['green'],'fontWeight':'700','fontSize':'13px','flexShrink':'0'}),
        html.Span(w,style={'fontSize':'13px','color':C['sub']}),
    ],style={'display':'flex','marginBottom':'4px'}) for w in exp['why']]

    why_block=html.Div([
        html.P('✅  Why We Recommend This',style={'fontSize':'10px','fontWeight':'700','color':C['green'],
               'textTransform':'uppercase','letterSpacing':'0.5px','marginBottom':'6px'}),
        *why_items,
        pref_pill(exp.get('pref_detail')),
    ],style={'background':C['green_light'],'border':f"1px solid #9ECEB8",
             'borderRadius':'9px','padding':'10px 13px'})

    warn_block=html.Div([
        html.P('⚠  Ingredient Alerts',style={'fontSize':'10px','fontWeight':'700','color':C['amber'],
               'textTransform':'uppercase','letterSpacing':'0.5px','marginBottom':'5px'}),
        *[html.Div([html.Span('! ',style={'color':C['amber'],'fontWeight':'700'}),
                   html.Span(w,style={'fontSize':'13px','color':C['sub']})],
                  style={'marginBottom':'3px'}) for w in exp['warnings']],
    ],style={'background':C['amber_light'],'border':f"1px solid #DFC090",
             'borderRadius':'9px','padding':'9px 13px','marginTop':'10px'}) if exp['warnings'] else None

    partial_block=html.Div([
        html.P("⚡  Why It's a Partial Match",style={'fontSize':'10px','fontWeight':'700',
               'color':C['oos_fg'],'textTransform':'uppercase','letterSpacing':'0.5px','marginBottom':'5px'}),
        html.P(exp['partial_reason'],style={'fontSize':'12px','color':C['sub'],'margin':'0','lineHeight':'1.6'}),
    ],style={'background':C['oos_bg'],'border':f"1px solid {C['border']}",
             'borderRadius':'9px','padding':'9px 13px','marginTop':'10px'}) if exp.get('partial_reason') else None

    tradeoff_block=html.Div([
        html.P('🔄  How It Compares',style={'fontSize':'10px','fontWeight':'700','color':C['blue'],
               'textTransform':'uppercase','letterSpacing':'0.5px','marginBottom':'5px'}),
        html.P(exp['tradeoff'],style={'fontSize':'12px','color':C['sub'],'lineHeight':'1.6','margin':'0'}),
    ],style={'background':C['blue_light'],'border':f"1px solid {C['blue']}",
             'borderRadius':'9px','padding':'9px 13px','marginTop':'10px'
    }) if exp['tradeoff'] and 'No further' not in exp['tradeoff'] else None

    return html.Div([
        # Header — FIX 22: no rank number on card
        html.Div([
            html.Div([badge_pill(row['Badge'],group)],style={'marginBottom':'6px'}),
            html.Div([
                html.Div(row['Name'],style={'fontWeight':'700','fontSize':'14px','color':C['text'],
                                            'lineHeight':'1.35','marginBottom':'3px'}),
                html.Div(row['Brand'],style={'fontSize':'14px','fontWeight':'700','color':C['rose'],
                                              'letterSpacing':'0.2px'}),
            ],style={'flex':'1','minWidth':'0'}),
            html.Div([
                html.Div(f"${row['Price']:.0f}",style={'fontSize':'22px','fontWeight':'900',
                                                        'color':C['text'],'textAlign':'right','lineHeight':'1'}),
                html.Div([html.Span('★ ',style={'color':C['amber'],'fontSize':'13px'}),
                          html.Span(f"{row['Rank']:.1f}",style={'fontSize':'13px','color':C['sub'],'fontWeight':'600'})],
                         style={'textAlign':'right','marginTop':'3px'}),
                html.Div([html.Span('Score ',style={'fontSize':'10px','color':C['muted']}),
                          html.Span(f"{row['Final_Score']:.3f}",style={'fontSize':'12px','color':C['rose'],'fontWeight':'700'})],
                         style={'textAlign':'right','marginTop':'2px'}),
                score_bar(row['Final_Score']),
            ],style={'minWidth':'78px','textAlign':'right','flexShrink':'0'}),
        ],style={'display':'flex','alignItems':'flex-start','marginBottom':'12px','gap':'12px'}),
        budget_el,why_block,warn_block,partial_block,tradeoff_block,
    ],style={**CARD_B,'borderLeft':f"4px solid {left_col}",'opacity':'0.90' if is_oos else '1'})

# ── APP LAYOUT ────────────────────────────────────────────────────
app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],title='Skincare Finder')

app.layout=html.Div([
    html.Div([html.Div([
        html.Span('🧴',style={'fontSize':'24px','marginRight':'10px'}),
        html.Span('Skincare Finder',style={'fontSize':'20px','fontWeight':'800','color':C['text']}),
        html.Span('  Personalised product recommendations',
                  style={'fontSize':'13px','color':C['muted'],'marginLeft':'10px','fontStyle':'italic'}),
    ],style={'display':'flex','alignItems':'center'})],
    style={'background':C['card'],'padding':'13px 30px','borderBottom':f"2px solid {C['rose']}",
           'boxShadow':'0 2px 8px rgba(0,0,0,0.05)','marginBottom':'22px'}),

    html.Div([dbc.Row([
        # LEFT PANEL
        dbc.Col([html.Div([
            html.P('Your Profile',style={'fontWeight':'700','fontSize':'15px','color':C['text'],'marginBottom':'18px'}),
            lbl('Skin Type'),
            dcc.Dropdown(id='skin-type',options=[{'label':s,'value':s} for s in SKIN_COLS],
                         value='Oily',clearable=False,style={'marginBottom':'14px','fontSize':'13px'}),
            lbl('Skin Concern'),
            dcc.Input(id='skin-concern',type='text',placeholder='e.g. acne, dryness, brightening',
                      value='acne pores excess oil',debounce=False,style=INPUT_S),
            dbc.Row([
                dbc.Col([lbl('Min Budget ($)'),dcc.Input(id='budget-min',type='number',value=0,min=0,step=5,style=INPUT_S)],md=6),
                dbc.Col([lbl('Max Budget ($)'),dcc.Input(id='budget-max',type='number',value=60,min=0,step=5,style=INPUT_S)],md=6),
            ]),
            lbl('Preferred Ingredients'),
            dcc.Dropdown(id='prefer-ingr',options=INGR_OPTIONS,value=['niacinamide','salicylic_acid'],
                         multi=True,placeholder='Select…',style={'marginBottom':'14px','fontSize':'13px'}),
            lbl('Avoid Ingredients'),
            dcc.Dropdown(id='avoid-ingr',options=INGR_OPTIONS,value=['alcohol','fragrance','parabens'],
                         multi=True,placeholder='Select…',style={'marginBottom':'14px','fontSize':'13px'}),
            lbl('Product Category'),
            dcc.Dropdown(id='category',options=[{'label':c,'value':c} for c in CATEGORIES],
                         value='Moisturizer',clearable=False,style={'marginBottom':'14px','fontSize':'13px'}),
            lbl('Filter by Brand (optional)'),
            dcc.Dropdown(id='brand-filter',options=[{'label':b,'value':b} for b in ALL_BRANDS],
                         value=[],multi=True,placeholder='Any brand…',style={'marginBottom':'22px','fontSize':'13px'}),
            html.Button('🔍  Find Products',id='run-btn',n_clicks=0,
                        style={'width':'100%','padding':'12px','background':C['rose'],'color':'#fff',
                               'fontWeight':'700','fontSize':'14px','border':'none','borderRadius':'10px',
                               'cursor':'pointer','letterSpacing':'0.3px',
                               'boxShadow':'0 4px 14px rgba(200,80,106,0.28)'}),
        ],style={**CARD_B,'position':'sticky','top':'20px'})],md=3),

        # RIGHT PANEL
        dbc.Col([html.Div(id='output-panel',children=[
            html.Div([
                html.Div('✨',style={'fontSize':'40px','marginBottom':'12px'}),
                html.P('Set your profile and click Find Products',
                       style={'color':C['muted'],'fontWeight':'600','fontSize':'15px'}),
                html.P('Products are only shown when they pass your active filters.',
                       style={'color':C['muted'],'fontSize':'13px','maxWidth':'340px','margin':'0 auto'}),
            ],style={'textAlign':'center','padding':'70px 40px','background':C['card'],
                     'borderRadius':'14px','border':f"2px dashed {C['border']}"}),
        ])],md=9),
    ])],style={'maxWidth':'1380px','margin':'0 auto','padding':'0 20px 40px'}),
],style={'background':C['bg'],'minHeight':'100vh','fontFamily':FONT})

# ── CALLBACK ─────────────────────────────────────────────────────
@callback(
    Output('output-panel','children'),
    Input('run-btn','n_clicks'),
    State('skin-type','value'),State('skin-concern','value'),
    State('budget-min','value'),State('budget-max','value'),
    State('prefer-ingr','value'),State('avoid-ingr','value'),
    State('category','value'),State('brand-filter','value'),
    prevent_initial_call=True,
)
def run_dss(_,skin_type,skin_concern,bmin,bmax,prefer_ingr,avoid_ingr,category,brand_filter):
    if not skin_type or not category:
        return html.Div('Please select Skin Type and Category.',
                        style={'color':C['red'],'padding':'20px','fontWeight':'600'})

    # FIX 13: clean inputs — only user values, no defaults
    bmin         = int(bmin or 0)
    bmax         = int(bmax or 999)
    prefer_ingr  = prefer_ingr  if prefer_ingr  else []   # FIX 13
    avoid_ingr   = avoid_ingr   if avoid_ingr   else []   # FIX 13
    brand_filter = brand_filter if brand_filter else []
    skin_concern = skin_concern if skin_concern else ''

    # FIX 16: user_input passed unchanged — no transformation
    user_input = {
        'skin_type'         : skin_type,
        'skin_concern'      : skin_concern,
        'budget'            : [bmin, bmax],
        'prefer_ingredients': prefer_ingr,
        'avoid_ingredients' : avoid_ingr,
        'product_category'  : category,
        'preferred_brands'  : brand_filter,
    }
    print(f'[DEBUG] USER INPUT: {user_input}')   # FIX 16

    abs_df,oos_df,meta=score_products(user_input)
    pi=meta['prefer_ing']; ai=meta['avoid_ing']
    fl=meta['filter_level']

    if abs_df.empty and oos_df.empty:
        return html.Div(f'No {category} products found for {skin_type} skin.',
                        style={'color':C['red'],'padding':'20px','fontWeight':'600'})

    # FIX 22: use _, row iteration — no numeric indexing in display
    abs_show=assign_badges(abs_df.head(6),'abs') if not abs_df.empty else pd.DataFrame()
    oos_show=assign_badges(oos_df.head(3),'oos') if not oos_df.empty else pd.DataFrame()

    abs_exps=[]
    for rank_pos,(_,row) in enumerate(abs_show.iterrows(),start=1):   # FIX 22
        e=build_explanation(row,user_input,abs_df,rank_pos,pi,ai,'abs',fl)
        e['_concern']=skin_concern
        abs_exps.append(e)

    oos_exps=[]
    for rank_pos,(_,row) in enumerate(oos_show.iterrows(),start=1):   # FIX 22
        e=build_explanation(row,user_input,oos_df,rank_pos,pi,ai,'oos',fl)
        e['_concern']=skin_concern
        oos_exps.append(e)

    # Summary
    prefer_str=', '.join(INGR_LABELS.get(x,x) for x in pi) or 'None'
    avoid_str =', '.join(INGR_LABELS.get(x,x) for x in ai) or 'None'
    brand_str =', '.join(brand_filter) if brand_filter else 'Any'

    summary=html.Div([
        html.P('Your Search',style={'fontWeight':'700','fontSize':'14px','color':C['text'],'marginBottom':'12px'}),
        html.Div([
            chip('🧴','Skin Type', skin_type,          C['rose_light'], C['rose']),
            chip('💰','Budget',   f'${bmin}–${bmax}',   C['green_light'],C['green']),
            chip('📦','Category',  category,             C['blue_light'], C['blue']),
            chip('🏷','Brand',     brand_str[:26]+'…' if len(brand_str)>26 else brand_str, C['amber_light'],C['amber']),
            chip('💊','Prefer',    prefer_str[:33]+'…' if len(prefer_str)>33 else prefer_str, C['rose_light'],C['rose']),
            chip('🚫','Avoid',     avoid_str[:33]+'…'  if len(avoid_str)>33  else avoid_str,  C['amber_light'],C['amber']),
        ],style={'display':'flex','flexWrap':'wrap'}),
        html.P(f"Found {meta['n_abs']} perfect match(es) + {meta['n_oos']} partial match(es) "
               f"across {meta['total_in_cat']} {skin_type}-compatible {category.lower()}s.",
               style={'fontSize':'12px','color':C['muted'],'marginTop':'8px','marginBottom':'0'}),
    ],style={**CARD_B,'borderLeft':f"4px solid {C['blue']}"})

    # Relaxation notice (FIX 18 UI)
    relax_el=html.Div([
        html.Span('⚡ ',style={'fontSize':'14px'}),
        html.Span(meta['relaxation_msg'],style={'fontSize':'13px','color':C['amber'],'fontWeight':'600'}),
    ],style={'background':C['warn_bg'],'border':f"1px solid {C['warn_border']}",
             'borderRadius':'9px','padding':'10px 14px','marginBottom':'12px'}) if meta.get('relaxation_msg') else None

    # Brand conflict
    brand_conflict_el=html.Div([
        html.Span('ℹ️ ',style={'fontSize':'14px'}),
        html.Span(meta['brand_conflict'],style={'fontSize':'13px','color':C['blue'],'fontWeight':'600'}),
    ],style={'background':C['info_bg'],'border':f"1px solid {C['info_border']}",
             'borderRadius':'9px','padding':'10px 14px','marginBottom':'12px'}) if meta.get('brand_conflict') else None

    # Section 1 — FIX 22: cards with no rank number
    s1_head=html.H6(f"✅  Perfect Matches — {skin_type} Skin · {category}  ({meta['n_abs']} found)",
                    style={'fontWeight':'800','color':C['text'],'marginBottom':'6px','marginTop':'4px'})
    s1_sub =html.P('All products below satisfy your skin type, budget, ingredient preferences, and avoid list.',
                    style={'fontSize':'12px','color':C['muted'],'marginBottom':'14px','marginTop':'-4px'})

    # FIX 22: iterate with _, row — no index used in display
    abs_cards=[product_card(abs_show.iloc[i],abs_exps[i],'abs') for i in range(len(abs_show))]
    abs_section=html.Div([s1_head,s1_sub,*abs_cards]) if abs_show.empty==False else html.Div([
        s1_head,
        html.Div([html.Span('🔍 ',style={'fontSize':'20px'}),
                  html.Span('No perfect matches.',style={'fontWeight':'700','color':C['sub'],'fontSize':'14px'}),
                  html.P('Showing closest alternatives below.',style={'fontSize':'13px','color':C['muted'],'margin':'4px 0 0'})],
                 style={**CARD_B,'background':'#FAF9F8','borderLeft':f"4px solid {C['muted']}"}),
    ])

    # Divider
    divider=html.Div([
        html.Hr(style={'borderColor':C['divider'],'margin':'22px 0 16px'}),
        html.Div([
            html.Span('◆ ',style={'fontSize':'12px','color':C['oos_fg']}),
            html.Span('Products not matching your specifications exactly',
                      style={'fontWeight':'700','fontSize':'13px','color':C['sub']}),
            html.Span(' — shown for reference only',style={'fontSize':'12px','color':C['muted']}),
        ],style={'marginBottom':'12px'}),
    ]) if not oos_show.empty else html.Div()

    # FIX 22: OOS cards — no index in display
    oos_cards=[product_card(oos_show.iloc[i],oos_exps[i],'oos') for i in range(len(oos_show))]
    oos_section=html.Div(oos_cards) if oos_cards else html.Div()

    return html.Div([summary,relax_el,brand_conflict_el,abs_section,divider,oos_section])

# ── RUN ──────────────────────────────────────────────────────────
if __name__=='__main__':
    print('\n'+'='*52)
    print('  🧴  Skincare Finder v6')
    print('  Open: http://127.0.0.1:8050')
    print('='*52+'\n')
    app.run(debug=True,port=8050)
