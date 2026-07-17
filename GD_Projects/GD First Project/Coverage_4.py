from os import WCONTINUED

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances


def coverage(df):
    defense=df[df.player_side=='Defense'],
    offence=df[df.player_side=='Offense'],
    records=[]
    for keys,off in offence.groupby(['game_id','play_id','nfl_id']):
        defn=defense[
            (defense.game_id==keys[0]) and (defense.play_id==keys[1] ) and (defense.nfl_id==keys[2])
        ]
        if defn.empty:
            continue

        dist=paired_distances(off['x','y'],defn['x','y'])
        for  i,off_row in off.iterrows():
            j=dist[off.index.get_loc(i)].argmin()
            records.append({
                'game_id':off_row.game_id,
                'play_id':off_row.play_id,
                'nfl_id':off_row.nfl_id,
                'frame_id':off_row.frame_id,
                'nearest_defender':defn.iloc[j].nfl_id,
                'coverage_dist':dist[off.index.get_loc(i),j]

            })
    return pd.DataFrame(records)







# we want to know that at each moment which defender is covering each offensive player.
# compute distance between off and def.
# assign the closest def
# and maintain the record that how far they are.
# a defender close in one frame not necessary that he will also close in next frame.
# select only those defenders those are close to off at the same time.
# dist compute the euclidean distance between def and off.
# then find the closest defender and record it.


