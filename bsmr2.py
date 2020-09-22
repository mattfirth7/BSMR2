# CHANGE FROM ORTHOGONAL TO MINIMUM MIC for feature selection
# mic(xcurrent,y)/product_across_previous(mic(xcurrent,xprevious))

import numpy as np
import pandas as pd
import minepy as mp
from sklearn.model_selection import train_test_split



def proj(b, a):
    numerator = np.dot(b, a)
    denominator = np.sqrt(sum(a**2))
    
    proj = (numerator / denominator) * a
    
    return proj

def two_var_orthog(b, a):
    projection_b_to_a = proj(b, a)
    
    orthogonal_comp = b - projection_b_to_a
    
    return orthogonal_comp

def max_info_coef(x, y):
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    
    x_test = np.asarray(x)
    y_test = np.asarray(y)
    
    mine.compute_score(x_test, y_test)
    mic_val = mine.mic()
    
    return mic_val

def norm_cols(data):
    for column in df:
        data[column] = data[column] / np.linalg.norm(data[column])
        
    return data

def find_max_mic(data, y):
    mic_arr = []
    col_arr = []
    i=0
    for column in data:
        mic_arr.append(max_info_coef(column[0], y))
        col_arr.append(i)
        i+=1
        
    return col_arr[np.argmax(mic_arr)], np.max(mic_arr)

def bootstrapped_mean_vectr(x):
    y=x
    bootstrapped_vectrs = []
    for i in range(10):
        x_train, x_test = train_test_split(y, test_size=0.1)
        bootstrapped_vectrs.append(np.array([x_test]).astype(np.float))
        y=x
        
    return np.average(np.array(bootstrapped_vectrs).astype(np.float), axis=0)


def main():
    data = pd.read_csv('C:/Users/Matt/Documents/max_relevancy_min_redundancy/heart_gm.csv')
    y = data.pop('target')
    x = data
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

        
    y_train_bs_vctr = bootstrapped_mean_vectr(y_train)
    
    bs_vctrs = []
    
    for column in x_train:
        bs_vctr = bootstrapped_mean_vectr(x_train[column])
        bs_vctrs.append(bs_vctr)
        
    
    col, mic = find_max_mic(bs_vctrs, y_train_bs_vctr[0])
    print(col)
    print(mic)
    
if __name__ == '__main__':
    main()