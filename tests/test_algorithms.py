"""Self-contained tests written for the portfolio (independent of UAB fixtures)."""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from KNN import KNN
from Kmeans import KMeans, distance
from my_labeling import Retrieval_by_shape, get_shape_accuracy, get_color_accuracy, Retrieval_combined

class AlgorithmsTest(unittest.TestCase):
    def test_knn_euclidean(self):
        data = np.array([[1., 1.], [10., 10.], [10., 11.]])
        knn = KNN(data, ['a', 'b', 'b'])
        self.assertEqual(knn.predict(np.array([[1., 1.]]), 1)[0], 'a')

    def test_knn_other_metrics(self):
        data = np.array([[1., 1.], [10., 10.], [0., 2.]])
        knn = KNN(data, ['a', 'b', 'c'])
        for metric in ['euclidean','cityblock','cosine']:
            prediction = knn.predict(np.array([[9., 10.]]), 1, distance=metric)
            self.assertEqual(prediction.shape,(1,))

    def test_kmeans_centroids_and_wcd(self):
        colors = np.array([[255.,0,0]]*5 + [[0.,0,255]]*5)
        model = KMeans(colors, K=2, options={'km_init':'first','max_iter':30,'tolerance':1e-6})
        model.fit()
        self.assertAlmostEqual(model.withinClassDistance(),0.)
        self.assertEqual(model.centroids.shape,(2,3))

    def test_kmeans_distance(self):
        out = distance([[0.,0.],[3.,4.]],[[0.,0.]])
        np.testing.assert_allclose(out,[[0.],[5.]])

    def test_kmeans_best_k(self):
        data = np.array([[0.,0.],[0.2,0.],[20.,20.],[20.2,20.],[40.,40.],[40.2,40.]])
        for heuristic in ('WCD','FISHER','INTER'):
            km=KMeans(data,K=2,options={'km_init':'first','max_iter':40,'tolerance':1e-6,'fitting':heuristic})
            result=km.find_bestK(4)
            self.assertIn(result,(2,3,4))
            self.assertEqual(km.K,result)
            self.assertEqual(km.centroids.shape,(result,2))

    def test_retrieval_by_shape(self):
        images=np.array([[[1]],[[2]],[[3]]])
        result=Retrieval_by_shape(images,['x','y','x'],'x',[25,90,75])
        np.testing.assert_equal(result[:,0,0],[3,1])

    def test_shape_accuracy(self):
        predictions={'euclidean':[np.array(['a','b']),np.array(['a','a'])]}
        actual=np.array(['a','b'])
        res=get_shape_accuracy(predictions,actual,[1,3])
        self.assertEqual(res['euclidean'],[100.,50.])
        plt.close('all')

    def test_color_accuracy(self):
        predicted=[['red'],['blue']]
        truth=[['red'],['blue','white']]
        self.assertEqual(get_color_accuracy(predicted,truth,threshold=0.5),1.)

    def test_combined_retrieval(self):
        imgs=np.array([[[1]],[[2]],[[3]]])
        result=Retrieval_combined(imgs,['shirt','shirt','shoes'],[['red'],['red'],['blue']],'shirt','red',[50.,90.,100.],[[60.],[70.],[100.]])
        self.assertEqual(len(result),2)
        self.assertEqual(int(result[0][0][0]),2)

if __name__ == '__main__':
    unittest.main()
