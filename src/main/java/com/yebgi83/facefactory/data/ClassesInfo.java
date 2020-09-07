package com.yebgi83.facefactory.data;

import com.yebgi83.facefactory.util.ResourceUtils;
import org.apache.commons.collections4.ListUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ClassesInfo {
    public static ClassesInfo fromPath(String path) throws IOException {
        List<String> classNames = null;

        try (InputStream inputStream = ResourceUtils.getResourceStream(path)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            while (true) {
                String line = reader.readLine();

                if (line != null) {
                    if (classNames == null) {
                        classNames = new ArrayList<>(Integer.parseInt(line));
                    } else {
                        classNames.add(line);
                    }
                } else {
                    return new ClassesInfo(classNames);
                }
            }
        }
    }

    private final List<String> classNames;

    private ClassesInfo(List<String> classNames) {
        this.classNames = ListUtils.unmodifiableList(classNames);
    }

    public int indexOf(String className) {
        return classNames.indexOf(className);
    }

    public int size() {
        return classNames.size();
    }

    public Map<Integer, String> toMap() {
        AtomicInteger index = new AtomicInteger();
        return classNames
                .stream()
                .distinct()
                .collect(
                        Collectors.toMap(
                                string -> index.getAndIncrement(),
                                Function.identity()
                        )
                );
    }
}