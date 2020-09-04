package com.konantech.facefactory;

import com.konantech.facefactory.data.ClassesInfo;
import com.konantech.facefactory.data.ImageDataSetLoader;
import com.konantech.facefactory.model.ModelFactory;
import com.konantech.facefactory.util.DataSetUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import static com.konantech.facefactory.util.ResourceUtils.getResourceStream;

public class Application {
    private static final String CLASSES_INFO_PATH = "classes.info";

    private static final String TRAIN_PATH = "train_images_19_small.txt";

    private static final String VALIDATION_PATH = "val_images_19.txt";

    public static void main(String[] args) throws IOException, InterruptedException {
        final int EPOCHS = 10;

        final int BATCH_SIZE = 16;

        ClassesInfo classesInfo = ClassesInfo.fromPath(CLASSES_INFO_PATH);

        ImageDataSetLoader imageDataSetLoader = new ImageDataSetLoader(classesInfo);

        ComputationGraph model = new ModelFactory(classesInfo.size())
                .softMax();

        IteratorDataSetIterator trainDataSetIterator = DataSetUtils.createMiniBatchIterator(
                BATCH_SIZE,
                IOUtils.readLines(getResourceStream(TRAIN_PATH), StandardCharsets.UTF_8),
                imageDataSetLoader
        );

        IteratorDataSetIterator validateDataSetIterator = DataSetUtils.createMiniBatchIterator(
                BATCH_SIZE,
                IOUtils.readLines(getResourceStream(VALIDATION_PATH), StandardCharsets.UTF_8),
                imageDataSetLoader
        );

        model.addListeners(
                new CheckpointListener.Builder("W:/")
                        .keepAll()
                        .saveEveryEpoch()
                        .build(),
                new EvaluativeListener(validateDataSetIterator, 1)
        );

        EarlyStoppingConfiguration<ComputationGraph> configuration = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(EPOCHS))
                .iterationTerminationConditions(
                        new MaxTimeIterationTerminationCondition(2, TimeUnit.HOURS)
                )
                .scoreCalculator(new ClassificationScoreCalculator(Evaluation.Metric.ACCURACY, validateDataSetIterator))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileGraphSaver("W:/"))
                .build();

        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(
                configuration,
                model,
                trainDataSetIterator
        );

        trainer.fit();
    }
}