package org.hipi.knn;


import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.net.URI;

public class runKNN {

    public static int run(String trainFilePath, String testFilePath, String outputDir, int k)
            throws ClassNotFoundException, IllegalStateException, InterruptedException, IOException {

        System.out.println("Running KNN.");

        Job job = Job.getInstance();

        job.setJarByClass(KNN.class);

        DistributedCache.addCacheFile(URI.create(trainFilePath), job.getConfiguration());

        job.setInputFormatClass(TextInputFormat.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(KNNMapper.class);
        job.setReducerClass(KNNReducer.class);
        job.setNumReduceTasks(1);

        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job, new Path(testFilePath));
        FileOutputFormat.setOutputPath(job, new Path(outputDir));

        job.getConfiguration().setInt("k", k);

        return job.waitForCompletion(true) ? 0 : 1;
    }
}