package org.hipi.knn;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class KNNReducer
        extends Reducer<IntWritable, IntWritable, NullWritable, DoubleWritable> {

    @Override
    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int trueCnt = 0, sum = 0;
        for (IntWritable val: values) {
            sum++;
            if (val.get() == 1) {
                trueCnt++;
            }
        }
        context.write(NullWritable.get(), new DoubleWritable((double)trueCnt/sum));
    }
}
