import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_route_features(df):
    routes=[]
    offence=df[
        (df.player_side=='Offence') and  (df.player_rol.isin(['Targeted Receiver','Other Route Runner']))

    ]

    for (g,p,n),gdf in offence.groupby([
        'game_id','play_id','nfl_id'
    ]):
        gdf=gdf.sort_values('frame_id')
        dx=gdf['x'].diff().fillna(0)
        dy=gdf['y'].diff().fillna(0)

        routes.append({
            "game_id":g,
            "play_id":p,
            "nfl_id":n,
            'depth':gdf['x'].iloc[-1]-gdf['x'].iloc[0],
            'width':gdf['y'].iloc[-1]-gdf['y'].iloc[0],
            'path_len':np.sqrt(dx**2+dy**2).sum(),
            'dir_std':gdf['dir'].std(),

        })
    return pd.DataFrame(routes)

# we want to convert all messy routes into meaningful routes
# we want to extract route features
# offence=df in this we only care about receivers running routes not for other things like QB,defenders and so on.
# because routes belongs to passer and receivers
# then we just sort the frame based on time because we do not want that path becomes scrambled and calculation for path becomes complex.
# dx and dy for how much player moved between two frames.
# depth means last position of the playes on ground from where he starts.
# width=how much player moved in y direction
# path len=euclidean distance between last and first point.
# dir_std=standard deviation of direction of the player. how much the player change directions.
# low std means straight
# high std means double moves,curls,cuts
# finally extract this above features.



def cluster_route_features(routes,k=6):
    X=routes.drop(columns=['game_id','play_id','nfl_id'])
    X=StandardScaler().fit_transform(X)
    KM=KMeans(n_clusters=k,random_state=42)
    routes['clusters']=KM.fit_predict(X)
    return routes


# using this function we basically want to cluster all similar routes because in this game we can not pick any random route. there is some specific route allowes.
# number of specific routes are 6-8 so we just take k=7 number of clusters.
# we use previous routes and clusters all the routes.
# we don't consider here the ids because routes does not depend on players ids.
# and standarized the data because data have float values and we want to scale the values. mean=0 std=1
# then apply kmeans clustring.
# return the new routes with the new feature clusters.



