package org.hipi.reduction;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class ReductionReducer extends
        Reducer<IntWritable, Text, Text, Text> {
    private final static Text k = new Text("");

    @Override
    public void reduce(IntWritable key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        for (Text val : values) {
            context.write(k, val);
        }
    }
}