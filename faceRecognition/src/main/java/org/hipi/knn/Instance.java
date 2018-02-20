package org.hipi.knn;


public class Instance {
    private double[] features;
    private int label;

    public Instance(String line) {
        String[] value = line.split(" ");
        features = new double[value.length - 1];
        for (int i = 0; i < features.length; i++) {
            features[i] = Double.parseDouble(value[i]);
        }
        label = Integer.parseInt(value[value.length - 1]);
    }

    public double[] getFeatures() {
        return features;
    }

    public int getLabel() {
        return label;
    }
}
