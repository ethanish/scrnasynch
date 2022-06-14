import matplotlib
from matplotlib.ticker import AutoMinorLocator
from matplotlib_venn import venn3, venn3_circles
from matplotlib.patches import Circle
import venn
#%matplotlib inline
import matplotlib.pyplot as plt 
from matplotlib.pyplot import rc_context
from matplotlib.patches import ConnectionPatch

from fastdtw import fastdtw
import numpy as np
import pandas as pd

class sync_artist:
    def __init__(self, _object,out_dir="./"):
        self._object=_object
        self._out_dir = out_dir
        self.check_obj_qc()


    def check_obj_qc(self):
        try:
            self._object.get_norm_df()
            #self._df
        except :
            raise ValueError("Please use artist object with normalized sequence object")
        

    def get_DTW_distance_list(self):
        distance_combination_list = []
        distance_left_list = []
        distance_right_list = []
        distance_value_list = []
        
        for _i in range(self._object.dflen):
            for _j in range(self._object.dflen):
                if _j <= _i:
                    continue
                distance_combination_list.append(( self._object.patients_name_list[_i], self._object.patients_name_list[_j]))
                #distance_left_list.append(self._object.patients_name_list[_i])
                #distance_right_list.append(self._object.patients_name_list[_j])
                distance_value_list.append(self._object.DTW_distances.iloc[_i,_j])
        
        DTW_distance_list = pd.DataFrame()
        DTW_distance_list['comb'] = distance_combination_list
        #DTW_distance_list['left'] = distance_left_list
        #DTW_distance_list['right'] = distance_right_list
        DTW_distance_list['value'] = distance_value_list
        #DTW_distance_list.sort_values(by=["left","right"],inplace=True,ascending=[True,True])
        DTW_distance_list.sort_values(by=["value"],inplace=True,ascending=[True])
        DTW_distance_list.reset_index(drop=True)
        self.DTW_distance_list = DTW_distance_list        
        return DTW_distance_list




    def get_Distance_plot_df(self):
        
        print("h")
        
