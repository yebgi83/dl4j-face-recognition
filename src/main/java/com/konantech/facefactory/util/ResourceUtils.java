package com.konantech.facefactory.util;

import java.io.InputStream;

public class ResourceUtils {
    public static InputStream getResourceStream(String path) {
        return Thread
                .currentThread()
                .getContextClassLoader()
                .getResourceAsStream(path);
    }
}