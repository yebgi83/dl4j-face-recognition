package com.yebgi83.facefactory.util;

import java.io.InputStream;

public class ResourceUtils {
    public static InputStream getResourceStream(String path) {
        return Thread
                .currentThread()
                .getContextClassLoader()
                .getResourceAsStream(path);
    }
}