package com.konantech.facefactory.model;


import com.konantech.facefactory.constants.HyperParameter;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelFactory {
    private static final String CLASSIFICATIONS = "classifications";

    private final int numClasses;

    public ModelFactory(int numClasses) {
        this.numClasses = numClasses;
    }

    public ComputationGraph softMax() {
        return new TransferLearning
                .GraphBuilder(XceptionFactory.computationGraph(1024, Activation.RELU))
                .fineTuneConfiguration(
                        new FineTuneConfiguration
                                .Builder()
                                .updater(
                                        Adam
                                                .builder()
                                                .learningRate(HyperParameter.LEARNING_RATE.getValue())
                                                .build()
                                )
                                .build()
                )
                .addLayer(
                        CLASSIFICATIONS,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(1024)
                                .nOut(numClasses)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        XceptionFactory.PREDICTIONS
                )
                .setOutputs(CLASSIFICATIONS)
                .build();
    }
}