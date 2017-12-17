import pandas as pd

OD = pd.read_csv('OD.csv')
OD['flow'] = OD[['flow_in', 'flow_out']].min(axis=1)
STATION = pd.read_csv('STATION.csv', index_col=0)
idx = STATION.index
ret = pd.DataFrame(0, index=idx, columns=idx)
for i in OD.index:
    o = OD.loc[i, 'o']
    d = OD.loc[i, 'd']
    ret.loc[o, d] = OD.loc[i, 'flow']
