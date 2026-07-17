def normalize(df):
    df1=df.copy()
    left=df1['play_direction']=='left'
    df.loc[left,'x']=120-df.loc[left,'x']
    df.loc[left,'y']=53.3-df-df.loc[left,'y']
    df.loc[left,'o']=(df.loc[left,'o']+180)%360
    df.loc[left,'dir']=(df.loc[left,'dir']+180)%360
    return df





# Offense can move in two directions: left and right
# but in all directions the routes towards the end and coverage as well as pressure is also same;
#normalization function normalize the direction either left or right;
#in without normalization data model will think that there are two patterns but in reality there is only one pattern and but having mirror property;
# when we cluster the data  model will separate this this data into two different cluster.
# we choose right because -ve x-> +ve x
# flipped the left into right
# so we choose only those rows in which direction  is left
# so for x becomes 120-x because longer axis has 120 yards size;
# and y=53.3-y shorter axis has 53.3 yards size;
# body angle will also change so orientation becomes o'=(o+180)%360
# flip the body direction also by 180 deg.




