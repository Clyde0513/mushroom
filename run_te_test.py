import pandas as pd
from mushroom_transformers import TargetEncoderOOF

def run_test():
    df = pd.DataFrame({'cat':['a','a','b','b','c','c','c'],'num':[1,0,1,0,1,0,1]})
    y = df['num']
    te = TargetEncoderOOF(cols=['cat'], n_splits=3, random_state=0)
    te.fit(df[['cat']], y)
    expected = df.groupby('cat')['num'].mean().to_dict()
    Xnew = pd.DataFrame({'cat':['a','b','c','d']})
    out = te.transform(Xnew)
    global_mean = y.mean()
    print('out:', out.flatten().tolist())
    print('expected a,b,c,global:', expected['a'], expected['b'], expected['c'], global_mean)
    assert abs(out[0,0] - expected['a']) < 1e-6
    assert abs(out[1,0] - expected['b']) < 1e-6
    assert abs(out[2,0] - expected['c']) < 1e-6
    assert abs(out[3,0] - global_mean) < 1e-6
    print('basic transformer check: PASS')

if __name__ == '__main__':
    run_test()
