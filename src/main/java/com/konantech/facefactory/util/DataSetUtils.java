package com.konantech.facefactory.util;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.LocalFileSourceFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

@UtilityClass
public class DataSetUtils {
    public IteratorDataSetIterator createMiniBatchIterator(int batchSize, List<String> paths, Loader<DataSet> loader) throws IOException {
        DataSetLoaderIterator rawDataSetIterator = new DataSetLoaderIterator(
                paths,
                loader,
                new LocalFileSourceFactory()
        );

        return new MiniBatchDataSetIterator(
                rawDataSetIterator,
                batchSize
        );
    }

    public class MiniBatchDataSetIterator extends IteratorDataSetIterator {
        public MiniBatchDataSetIterator(Iterator<DataSet> iterator, int batchSize) {
            super(iterator, batchSize);
        }

        @Override
        public boolean resetSupported() {
            return Optional
                    .ofNullable(getDataSetIterator())
                    .map(DataSetIterator::resetSupported)
                    .orElse(false);
        }

        @Override
        public void reset() {
            if (resetSupported()) {
                Optional
                        .ofNullable(getDataSetIterator())
                        .ifPresent(DataSetIterator::reset);
            } else {
                super.reset();
            }
        }

        private DataSetIterator getDataSetIterator() {
            @SuppressWarnings("unchecked")
            Iterator<DataSet> iterator = Optional
                    .of(FieldUtils.getField(getClass(), "iterator", true))
                    .map(field -> {
                        try {
                            return (Iterator<DataSet>) field.get(this);
                        } catch (IllegalAccessException e) {
                            return null;
                        }
                    })
                    .orElse(null);

            if (iterator instanceof DataSetIterator) {
                return (DataSetIterator) iterator;
            } else {
                return null;
            }
        }
    }
}