package org.hipi.knn;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.hipi.util.helper;

public class KNNMapper
        extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
    private int k;
    private ArrayList<Instance> trainSet;

    private final static IntWritable zero = new IntWritable(0);
    private IntWritable outValue = new IntWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        k = context.getConfiguration().getInt("k", 1);
        trainSet = new ArrayList<>();

        Path[] trainFile = DistributedCache.getLocalCacheFiles(context.getConfiguration());
        BufferedReader br;
        String line;
        for (int i = 0; i < trainFile.length; i++) {
            br = new BufferedReader(new FileReader(trainFile[0].toString()));
            while ((line = br.readLine()) != null) {
                Instance trainInstance = new Instance(line);
                trainSet.add(trainInstance);
            }
        }
    }

    @Override
    public void map(LongWritable textIndex, Text textLine, Context context)
            throws IOException, InterruptedException {
        ArrayList<Double> k_distances = new ArrayList<>(k);
        ArrayList<Integer> k_labels = new ArrayList<>(k);

        for (int i = 0; i < k; i++) {
            k_distances.add(Double.MAX_VALUE);
            k_labels.add(-1);
        }

        Instance testInstance = new Instance(textLine.toString());

        for (int i = 0; i < trainSet.size(); i++) {
            double dist = helper.EuclideanDistance(trainSet.get(i).getFeatures(), testInstance.getFeatures());
            int index = helper.indexOfMax(k_distances);
            if (dist < k_distances.get(index)) {
                k_distances.remove(index);
                k_labels.remove(index);
                k_distances.add(dist);
                k_labels.add(trainSet.get(i).getLabel());
            }
        }

        int predictedLabel = helper.getMostFrequentValue(k_labels), groundLabel = testInstance.getLabel();

        if (predictedLabel == groundLabel) {
            outValue.set(1);
        } else {
            outValue.set(0);
        }

        context.write(zero, outValue);
    }
}
