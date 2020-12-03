import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels= np.array(labels).astype(int).tolist()

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbours.
        :param point: List[float]
        :return:  List[int]
        """
        distances = list(map(lambda x:self.distance_function(x,point),self.features))
        distancesWithIndex = np.array(sorted(list(zip(distances,range(len(distances)))),key=lambda x:(x[0],x[1])))
        k_index = distancesWithIndex[:self.k,1].astype(int)
        # k_index = np.argpartition(distances,self.k)[:self.k]
        # k_index_distance = np.take(distances,k_index)
        # temp= sorted(list(zip(k_index_distance,k_index)))
        # k_index = np.sort(k_index) 
        # print(type(np.take(self.labels,k_index).tolist()[0]))
        result=np.take(self.labels,k_index).tolist()
        if np.unique(np.bincount(result)).size==1 and np.bincount(result).shape[0] >1 and self.k>1:
            # k_index = np.argpartition(distances,self.k-1)[:self.k-1]
            
            k_index = distancesWithIndex[:self.k-1,1]
            k_index = k_index.astype(int)
            result=np.take(self.labels,k_index).tolist()


        return result
		
	# TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        # result=[]
        # for i in features:
        #     k_near = self.get_k_neighbors(i)
        #     counts = np.bincount(k_near)
        #     # if np.unique(counts).size ==1 and len(k_near)>1:             
        #     #     # withoutLast=k_near[:-1]
        #     #     # result.append(np.bincount(withoutLast).argmax().tolist())
        #     #     result.append(0)
        #     # else:
        #     result.append(np.argmax(counts).tolist())
        # return result
                

        return list(map(lambda x:np.bincount(x).argmax().tolist(),map(lambda x:self.get_k_neighbors(x),features)))


if __name__ == '__main__':
    print(np.__version__)
