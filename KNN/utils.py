import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    
    assert len(real_labels) == len(predicted_labels)
    temp = list(zip(real_labels,predicted_labels))
    tp = float(sum([i==1 and j==1 for i,j in temp]))
    fp = float(sum([i==0 and j==1 for i,j in temp]))
    fn = float(sum([i==1 and j==0 for i,j in temp]))
    if 2 * tp + fp + fn == 0:
        f1 = 0
    else:
        f1 = 2 * tp / float(2 * tp + fp + fn)

    # recall = np.dot(real_labels,predicted_labels)/ np.count_nonzero(real_labels)
    # precision= np.dot(real_labels,predicted_labels)/ np.count_nonzero(predicted_labels)
    # f1= 2* precision*recall/(precision+recall)
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.linalg.norm(np.subtract(point1,point2),3).tolist()

    @staticmethod
    # TODO
        
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.linalg.norm(np.subtract(point1,point2)).tolist()


    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if np.linalg.norm(point1)*np.linalg.norm(point2) == 0:
            return 0.0
        else:
            return 1.0 - (np.dot(point1, point2) / (np.linalg.norm(point1)*np.linalg.norm(point2)))



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = 0
        self.best_distance_function = None
        self.best_model = None
        best_score=-float("inf")
        dis_list=['euclidean','minkowski','cosine_dist']
        for k in dis_list:
            for i in range(1,min(len(x_train),30)):
                model = KNN(i,distance_funcs[k])
                model.train(x_train,y_train)
                temp_f1 = f1_score(y_val,model.predict(x_val))
                if temp_f1> best_score:
                    best_score=temp_f1
                    self.best_k=i
                    self.best_distance_function=k
                    self.best_model=model
                    # else:
                    #     if k == self.best_distance_function:
                    #         continue
                    #     elif len(k) == len(self.best_distance_function):
                    #         if k=="euclidean":
                    #             self.best_k=i
                    #             self.best_distance_function=k
                    #             self.best_model=model
                    #     elif len(k)<len(self.best_distance_function):
                    #         self.best_k=i
                    #         self.best_distance_function=k
                    #         self.best_model=model




        #     f1_scores =[self.helper(i,v,x_train,y_train,x_val,y_val) for i in range(1,30)]
            
        #     temp_k = np.argmax(f1_scores)
        #     temp_score = f1_scores[temp_k]

        #     best_list.append((temp_score,temp_k+1,k))
        # def custom_sort(data):
        #     if len(data)>1:
        #         midpoint = len(data)//2
        #         L=data[:midpoint]
        #         R=data[midpoint:]
        #         custom_sort(L)
        #         custom_sort(R)
        #         i = j =k =0

        #         while i<len(L) and j<len(R):
        #             l = L[i]
        #             r = R[j]

        #             if l[0] > r[0]:
        #                 data[k]=l
        #                 i+=1
        #             elif r[0] > l[0]:
        #                 data[k]=r
        #                 j+=1
        #             else:                        
        #                 if l[2]==r[2]:
        #                     if l[1]<=r[1]:
        #                         data[k]=l
        #                         i+=1
        #                     else:
        #                         data[k]=r
        #                         j+=1
        #                 else:
        #                     if len(l[2]) == len(r[2]):
        #                         if l[2]<r[2]:
        #                             data[k] =l
        #                             i+=1
        #                         else:
        #                             data[k]=r
        #                             j+=1
                                
        #                     elif len(l[2])>len(r[2]):
        #                         data[k]=r
        #                         j+=1
        #                     else:
        #                         data[k]=l
        #                         i+=1
        #             k+=1
        #         while i < len(L): 
        #             data[k] = L[i] 
        #             i+= 1
        #             k+= 1
                
        #         while j < len(R): 
        #             data[k] = R[j] 
        #             j+= 1
        #             k+= 1
        # custom_sort(best_list)
        # temp= best_list[0]
        # self.best_k = temp[1]
        # self.best_distance_function=temp[2]
        # self.best_model = KNN(self.best_k,distance_funcs[self.best_distance_function])
        # self.best_model.train(x_train,y_train)
        self.best_scaler = None
    
    def helper(self,k,dis_function,x_train,y_train,x_val,y_val):
        knn = KNN(k,dis_function)
        knn.train(x_train,y_train)
        return f1_score(y_val,knn.predict(x_val))
        

        
         
                

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        dis_list=['euclidean','minkowski','cosine_dist']
        scal_list=['min_max_scale','normalize']
        best_score=-float("inf")
        for scal in scal_list:
            scalModel = scaling_classes[scal]()
            scaled_x_train = scalModel(x_train)
            scaled_x_val =scalModel(x_val)
            for k in dis_list:
                for i in range(1,min(30,len(scaled_x_train))):
                    model = KNN(i,distance_funcs[k])
                    model.train(scaled_x_train,y_train)
                    temp_f1 = f1_score(y_val,model.predict(scaled_x_val))
                    if temp_f1>best_score:
                        # if temp_f1>best_score:
                            best_score=temp_f1
                            self.best_k=i
                            self.best_distance_function=k
                            self.best_model=model
                            self.best_scaler=scal
                        # else:
                        #     if self.best_scaler is not None:
                        #         if len(scal) > len(self.best_scaler):
                        #             self.best_k=i
                        #             self.best_distance_function=k
                        #             self.best_model=model
                        #             self.best_scaler=scal
                        #         if scal == self.best_scaler:
                        #             if k == self.best_distance_function:
                        #                 continue
                        #             elif len(k) == len(self.best_distance_function):
                        #                 if k<self.best_distance_function:
                        #                     self.best_k=i
                        #                     self.best_distance_function=k
                        #                     self.best_model=model
                        #                     self.best_scaler=scal
                        #             elif len(k)<len(self.best_distance_function):
                        #                 self.best_k=i
                        #                 self.best_distance_function=k
                        #                 self.best_model=model
                        #                 self.best_scaler=scal




        #         f1_scores =[self.helper(i,v,scaled_x_train,y_train,scaled_x_val,y_val) for i in range(1,30)]
        #         temp_k = np.argmax(f1_scores)
        #         temp_score = f1_scores[temp_k]
        #         best_list.append((temp_score,temp_k+1,k,scal))
        # def custom_sort(data):
        #     if len(data)>1:
        #         midpoint = len(data)//2
        #         L=data[:midpoint]
        #         R=data[midpoint:]
        #         custom_sort(L)
        #         custom_sort(R)
        #         i = j =k =0
                
        #         while i<len(L) and j<len(R):
        #             l = L[i]
        #             r = R[j]
                    
        #             if l[0] > r[0]:
        #                 data[k]=l
        #                 i+=1
        #             elif r[0] > l[0]:
        #                 data[k]=r
        #                 j+=1
        #             else:
        #                 if len(l[3])>len(r[3]):
        #                     data[k]=l
        #                     i+=1
        #                 elif len(l[3])<len(r[3]):
        #                     data[k]=r
        #                     j+=1
        #                 else:
        #                     if l[2]==r[2]:
        #                         if l[1]<=r[1]:
        #                             data[k]=l
        #                             i+=1
        #                         else:
        #                             data[k]=r
        #                             j+=1
        #                     else:
        #                         if len(l[2]) == len(r[2]):
        #                             if l[2]<r[2]:
        #                                 data[k] =l
        #                                 i+=1
        #                             else:
        #                                 data[k]=r
        #                                 j+=1                                    
        #                         elif len(l[2])>len(r[2]):
        #                             data[k]=r
        #                             j+=1
        #                         else:
        #                             data[k]=l
        #                             i+=1
        #             k+=1
        #         while i < len(L): 
        #             data[k] = L[i] 
        #             i+= 1
        #             k+= 1
                
        #         while j < len(R): 
        #             data[k] = R[j] 
        #             j+= 1
        #             k+= 1
        # custom_sort(best_list)
        # temp = best_list[0]
        # self.best_k = temp[1]
        # self.best_distance_function=temp[2]
        # self.best_model = KNN(self.best_k,distance_funcs[self.best_distance_function])
        # self.best_scaler = temp[3]
        # temp_scal_model = scaling_classes[self.best_scaler]()
        # self.best_model.train(temp_scal_model(x_train),y_train)


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norms = np.linalg.norm(features,axis=1)
        features=np.array(features).astype("float64")
        
        for i in range(norms.shape[0]):
            if norms[i] != 0 :
                
                features[i,:] = features[i,:]/norms[i]
        
        return features.tolist()


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features_to_zero = np.argwhere(np.equal(np.max(features,axis=0),np.min(features,axis=0))==True)
        features=np.array(features)
        features=features.astype(float)
        for i in range(features.shape[1]):
            if i in features_to_zero:
                features[:,i]=0
            else:
                temp = np.subtract(features[:,i],np.min(features[:,i]))/(np.max(features[:,i])-np.min(features[:,i]))
                features[:,i]=temp
                
        return features.tolist()