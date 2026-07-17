from sklearn.linear_model import LogisticRegression

#

def probability(df):
    df['open']=(df['coverage_dist']>3).astype(int)
    model=LogisticRegression()
    X=df[['coverage_dist']].fillna(0)
    y=df['open']
    model.fit(X,y)
    df['probability_for_open']=model.predict_proba(X)[:,1]
    return df


