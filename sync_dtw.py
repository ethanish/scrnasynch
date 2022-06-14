import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from sklearn.cluster import AgglomerativeClustering
import networkx as nx


class sync_analysis:
    def __init__(self, ndf, rna_point_df, appendframe=False):
        self.rawdf = ndf
        self.rnadf = rna_point_df
        self.appendframe_flag = appendframe
        
        self.patients_name_list = ndf.index.to_list()
        self.dflen = len(self.rawdf)
        
        self.get_sampling_df()
        
        self.set_empty_array()


    def get_norm_df(self):
        return self.normdf 
    
    
    
    
    def set_empty_array(self):
        # 1. make empty 2d array
        self.DTW_distances = np.zeros((self.dflen, self.dflen))
        self.DTW_distances_nonorm = np.zeros((self.dflen, self.dflen))
        
        self.DTW_paths = []
        self.DTW_paths_nonorm = []
    
    def normalize_sequences(self, method = "swm"):
        # 2. normalization / trimming / append first and last
        new_df_list = []
        nonorm_df_list = []
        max_score_list = []
        score_length_list = []
        # 2.0. for the DTW algorithm, add -1 to first and last / it could go to the end of normalization
        
        
        for _x in self.rawdf.to_numpy():
            _y = _x[~np.isnan(_x)]
            #_ny = append_frame(_y, val=-1)
            _ny = _y

            if self.appendframe_flag == True:
                _ny = append_frame(_ny,0)
            nonorm_df_list.append(_ny)
        
        
        
        if method == "swm":
            for _x in self.rawdf.to_numpy():
                _y = _x[~np.isnan(_x)]
                if self.appendframe_flag == True:
                    _ny = append_frame(_y, val=0)
                else:
                    _ny = _y
                _ny = ish_sliding_window_mean_normalization(_ny, window_size = 3)
                new_df_list.append(_ny)
                
        elif method == "mm":
            for _x in self.rawdf.to_numpy():
                _y = _x[~np.isnan(_x)]
                if self.appendframe_flag == True:
                    _ny = append_frame(_y, val=-1)
                else:
                    _ny = _y
                _ny = ish_minmax_normalization(_ny)
                new_df_list.append(_ny)
                
        elif method == "swmm":
            for _x in self.rawdf.to_numpy():
                _y = _x[~np.isnan(_x)]
                if self.appendframe_flag == True:
                    _ny = append_frame(_y, val=-1)
                else:
                    _ny = _y
                _ny = ish_sliding_window_minmax_normalization(_ny, window_size = 3)
                new_df_list.append(_ny)
        elif method == "none":
            new_df_list = nonorm_df_list
        else:
            raise Exception("Sorry, please use proper method")
            
        
        for _x in self.rawdf.to_numpy():
            _y = _x[~np.isnan(_x)]
            if self.appendframe_flag == True:
                _y = append_frame(_y, val= 0)
            max_score_list.append(np.max(_y))
            score_length_list.append(len(_y))
            
        
        
        # Properties
        self.max_score_list = max_score_list
        self.score_length_list = score_length_list
        
        # New score df
        new_df = pd.DataFrame(new_df_list)
        new_df.index = self.patients_name_list
        self.normdf = new_df
        
        # Raw score df
        nonorm_df = pd.DataFrame(nonorm_df_list)
        nonorm_df.index = self.patients_name_list
        self.nonormdf = nonorm_df



    def DTW_distance_matrix(self):
        # 4. move next warped time point and do step 3 again
        # 4. make distance matrix for classification after minmax normalization
        # 4.1. With Minmax normalization
        for _i, _row in zip(range(self.dflen), self.normdf.to_numpy()):
            temp_DTW_path = []
            for _j, _col in zip(range(self.dflen), self.normdf.to_numpy()):
                # i as row
                # j as column
                _x = _row[~np.isnan(_row)]
                _y = _col[~np.isnan(_col)]
                _distance, _path = fastdtw(_x, _y)
                temp_DTW_path.append( _path )
                if self.DTW_distances[_j,_i] != 0:
                    self.DTW_distances[_i,_j] = self.DTW_distances[_j,_i]
                else:
                    self.DTW_distances[_i,_j] = _distance
            self.DTW_paths.append(temp_DTW_path)
        
        
        self.DTW_distances = pd.DataFrame(self.DTW_distances)
        self.DTW_distances.index = self.patients_name_list
        self.DTW_distances.columns = self.patients_name_list
        
        self.DTW_paths = pd.DataFrame(self.DTW_paths)
        self.DTW_paths.index = self.patients_name_list
        self.DTW_paths.columns = self.patients_name_list
        
        # 4.2. Without Normalization just in case
        for _i, _row in zip(range(self.dflen), self.rawdf.to_numpy()):
            temp_DTW_path = []
            for _j, _col in zip(range(self.dflen),self.rawdf.to_numpy()):
                # i as row
                # j as column
                # print()
                _x = _row[~np.isnan(_row)]
                _y = _col[~np.isnan(_col)]
                _distance, _path = fastdtw(_x, _y)
                temp_DTW_path.append(_path)
                if self.DTW_distances_nonorm[_j,_i] != 0:
                    self.DTW_distances_nonorm[_i,_j] = self.DTW_distances_nonorm[_j,_i]
                else:
                    self.DTW_distances_nonorm[_i,_j] = _distance
            self.DTW_paths_nonorm.append(temp_DTW_path)
        
        
        self.DTW_distances_nonorm = pd.DataFrame(self.DTW_distances_nonorm)
        self.DTW_distances_nonorm.index = self.patients_name_list
        self.DTW_distances_nonorm.columns = self.patients_name_list
        
        self.DTW_paths_nonorm = pd.DataFrame(self.DTW_paths_nonorm)
        self.DTW_paths_nonorm.index = self.patients_name_list
        self.DTW_paths_nonorm.columns = self.patients_name_list

    def get_sampling_df(self, window = 1):
        

        samples_list = []
        indexes_list = []
        patients_list = []
        names_list = []
        columns_list = ["samples","index","patients","names"]



        # Sampling points
        for _x, _p in zip(self.rnadf.to_numpy(), self.rnadf.index.to_list()):
            _y = _x[~pd.isnull(_x)]
            
            if self.appendframe_flag == True: 
                _y = append_frame(_y, False)
            _index_list = np.where(_y != False)[0]
            
            # loop with with index of sampling points
            for _s, _name in zip(_index_list, _y[_y != False]):
                
                #sample_list.append((str(_p)+_name[0], _s,_p, _name)) # Sample name/ Time Index/ Label
                
                samples_list.append(str(_p)+_name[0])
                indexes_list.append(_s)
                patients_list.append(_p)
                names_list.append(_name)

        sampling_df = pd.DataFrame()
        sampling_df['samples'] = samples_list
        sampling_df['indexes'] = indexes_list
        sampling_df['patients'] = patients_list
        sampling_df['names'] = names_list

        self.sampling_df = sampling_df





    def DTW_distance_hierarchical_clustering(self, cluster_n):
        # n_clusters_list = [1,2,3,4,5,10,15] # The number of conditions
        # linage_list = [‘complete’, ‘average’, ‘single’] # The list of parameter
        _clustering = AgglomerativeClustering(n_clusters=cluster_n, linkage='complete',affinity='precomputed').fit(self.DTW_distances)
        # print(clustering)
        
        _dendrogram = pd.Series(_clustering.labels_, index=self.patients_name_list)
        _dendrogram.name = "hierarchi"
        self.hierarchical_label = _dendrogram
        
        

    def DTW_distance_comparing_set(self):
        try:
            _label_group = self.hierarchical_label.groupby(self.hierarchical_label)
        except:
            raise Exception("Sorry, please use after 'DTW_distance_hierarchical_clustering() function'")
        
        try:
            self.sampling_df
        except:
            raise Exception("Sorry, please use after 'get_sampling_df() function'")
        
        self.condition_name_list = []
        self.condition_set_list = []
        self.sample_distance_list = []
        self.sample_graph_list = []
        self.used_sample_list = []
        
        self.comparing_set = []
        
        
        
        for _idx, _g in _label_group:
            self.condition_name_list.append(_idx)
            self.condition_set_list.append(_g.index.to_list())
            
            temp_sample_df = self.sampling_df[self.sampling_df['patients'].isin(_g.index.to_list())]
            self.used_sample_list.append(temp_sample_df)
            
            # After dtw path has warped and warped time points are OMICS sampoing points
            # Both sampoing index points is considered as connected edge of graph
            
            # Check every sampling points
            _idx1 = 0 # index of 'temp_sample_distance' matrix
            temp_sample_distances = np.zeros((len(temp_sample_df),len(temp_sample_df)))
            temp_sample_graph = nx.Graph()
            temp_sample_name_list = []
            temp_comparing_set = []
            
            append_graph_list = []
            append_graph_values_list = []
            # comparing left list loop
            for _s1 in temp_sample_df.to_numpy():
                
                _idx2 = 0
                temp_sample_name_list.append(_s1[0])
                
                # comparing right list loop
                for _s2 in temp_sample_df.to_numpy():
                    # if aleady cacluated, just copy value
                    if temp_sample_distances[_idx2,_idx1]!=0:
                        temp_sample_distances[_idx1,_idx2] = temp_sample_distances[_idx2,_idx1]
                    else:
                        # print(_s1,_s2)
                        # temp_sample_distances[_idx1,_idx2]
                        
                        # First, if comparing two samples from different patients, not from same patient samples
                        if _s1[2] == _s2[2]:
                            temp_sample_distances[_idx1,_idx2] = np.inf
                            
                        # Second, if there are matched via DTW algoritm / sencond option will be the navie window allowment
                        else:
                            _connected_flag = False
                            for _itp, _jtp in self.DTW_paths.loc[_s1[2],_s2[2]]:
                                if _s1[1] == _itp and _s2[1] == _jtp :
                                    _connected_flag = True
                            
                            # 2.1. connecting exists, DTW Distance between sample is DTW distance between seqeunce
                            if _connected_flag == True:
                                temp_sample_distances[_idx1,_idx2] = self.DTW_distances.loc[_s1[2],_s2[2]]
                                append_graph_list.append((_s1[0],_s2[0], self.DTW_distances.loc[_s1[2],_s2[2]]))
                                append_graph_values_list.append(self.DTW_distances.loc[_s1[2],_s2[2]])
                                
                            # 2.2. No connecting point
                            else:
                                temp_sample_distances[_idx1,_idx2] = np.inf
                    _idx2 += 1
                _idx1 += 1
            
            
            
            for _sample1, _sample2, _distance in append_graph_list:
                ##### No options
                temp_sample_graph.add_edge(_sample1,_sample2, distance = _distance)
                
                # ##### Option: cut edge under mean
                # if (_distance < np.mean(append_graph_values_list)):
                #     Sampling_Graph.add_edge(_sample1,_sample2, distance = _distance)
                    
                    
                # ##### Option: cut edge under mean
                # if (_distance < np.median(append_graph_values_list)):
                #     Sampling_Graph.add_edge(_sample1,_sample2, distance = _distance)
            
            temp_sample_distances = pd.DataFrame(temp_sample_distances)
            temp_sample_distances.index = temp_sample_name_list
            temp_sample_distances.columns = temp_sample_name_list
            
            self.sample_distance_list.append(temp_sample_distances)
            
            
            # Finally linked sampling points
            
            _sum_num = 0
            _sum_set = set()
            
            for _c in nx.connected_components(temp_sample_graph):
                temp_comparing_set.append(set(_c))
                _sum_set = _sum_set | _c
            
            _missed_set = set(temp_sample_df['samples']) - _sum_set
            for _mc in _missed_set:
                temp_comparing_set.append(set([_mc]))
                
                temp_sample_graph.add_node(_mc)
                
            self.sample_graph_list.append(temp_sample_graph)
            
            self.comparing_set.append(temp_comparing_set)
        
        
        
        # lavel_group





    def synchronization(self):
        print("this")



