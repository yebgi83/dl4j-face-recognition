package com.yebgi83.facefactory.constants;

public enum HyperParameter {
    LEARNING_RATE(0.001);

    private Object value;

    @SuppressWarnings("unchecked")
    public <T> T getValue() {
        return (T) value;
    }

    HyperParameter(Object value) {
        this.value = value;
    }
}