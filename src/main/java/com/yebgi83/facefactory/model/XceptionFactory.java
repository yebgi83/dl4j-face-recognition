package com.yebgi83.facefactory.model;

import lombok.experimental.UtilityClass;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.model.Xception;
import org.nd4j.linalg.activations.Activation;

@UtilityClass
public class XceptionFactory {
    public static final String PREDICTIONS = "predictions";

    public ComputationGraph computationGraph(int numClasses, Activation activation) {
        ComputationGraph baseModel = Xception
                .builder()
                .inputShape(new int[]{3, 299, 299})
                .numClasses(numClasses)
                .build()
                .init();

        return new TransferLearning
                .GraphBuilder(baseModel)
                .removeVertexKeepConnections(PREDICTIONS)
                .addLayer(
                        PREDICTIONS,
                        new DenseLayer.Builder()
                                .nIn(2048)
                                .nOut(numClasses)
                                .activation(activation)
                                .build(),
                        "avg_pool"
                )
                .build();
    }
}