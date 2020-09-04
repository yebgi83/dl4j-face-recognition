package com.konantech.facefactory.data;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.datavec.api.transform.transform.integer.IntegerToOneHotTransform;
import org.datavec.image.loader.ImageLoader;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

public class ImageDataSetLoader implements Loader<DataSet> {
    private final IntegerToOneHotTransform oneHotTransform;

    private final ClassesInfo classesInfo;

    public ImageDataSetLoader(ClassesInfo classesInfo) {
        this.oneHotTransform = new IntegerToOneHotTransform("category", 0, classesInfo.size());
        this.classesInfo = classesInfo;
    }

    @Override
    public DataSet load(Source source) throws IOException {
        String path = source.getPath();
        String label = getLabel(path);

        return new DataSet(
                getImage(path),
                getCategoricalVector(label)
        );
    }

    private String getLabel(String path) {
        String[] parts = StringUtils.split(path, "\\/");

        int lastIndex = parts.length - 1;

        if (lastIndex >= 1) {
            return parts[lastIndex - 1];
        } else {
            return null;
        }
    }

    private INDArray getImage(String path) {
        File file = new File(path);

        try {
            if (file.exists()) {
                return forMiniBatch(new ImageLoader().toBgr(file));
            } else {
                return forMiniBatch(Nd4j.create(3, 299, 299));
            }
        } catch (Exception e) {
            return forMiniBatch(Nd4j.create(3, 299, 299));
        }
    }

    private INDArray getCategoricalVector(String label) {
        INDArray rawResult = Optional
                .empty()
                .map(empty -> {
                    int labelIndex = classesInfo.indexOf(label);

                    if (labelIndex >= 0) {
                        @SuppressWarnings("unchecked")
                        List<Integer> vector = (List<Integer>) oneHotTransform
                                .map(labelIndex);

                        return Nd4j.createFromArray(vector.toArray(new Integer[0]));
                    } else {
                        return null;
                    }
                })
                .orElse(Nd4j.zeros(classesInfo.size()));

        return forMiniBatch(rawResult);
    }

    private INDArray forMiniBatch(INDArray rawResult) {
        return rawResult.reshape(ArrayUtils.addAll(new long[]{1}, rawResult.shape()));
    }
}