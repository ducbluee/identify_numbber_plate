from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np 
import tensorflow as tf 

import argparse
import sys
import time

class NeuralNetwork():
    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile
        
        self.label = self.load_labels(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image(self, image, imageSizeOuput):
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation = cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data,axis=0)
        return np_final

    def label_image(self, tensor):
        #### for MobileNet
        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top_k = results.argsort()[-1:][::-1]
        return labels[top_k[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            plate = plate + self.label_image(self.read_tensor_from_image(img, imageSizeOuput))
            # print(plate)
            return plate, len(plate)
