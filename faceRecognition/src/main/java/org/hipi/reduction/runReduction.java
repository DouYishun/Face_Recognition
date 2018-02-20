package org.hipi.reduction;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.hipi.imagebundle.mapreduce.HibInputFormat;
import org.hipi.mapreduce.BinaryOutputFormat;
import org.hipi.opencv.OpenCVMatWritable;

import java.io.IOException;

public class runReduction {

    public static int run(String[] args, String inputHibPath, String transformMatrixPath,
                          String meanPath, String outputDir)
            throws ClassNotFoundException, IllegalStateException, InterruptedException, IOException {

        System.out.println("Running reduction.");

        Job job = Job.getInstance();

        job.setJarByClass(Reduction.class);

        job.setInputFormatClass(HibInputFormat.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(ReductionMapper.class);
        job.setReducerClass(ReductionReducer.class);
        job.setNumReduceTasks(1);

        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job, new Path(inputHibPath));
        FileOutputFormat.setOutputPath(job, new Path(outputDir));

        job.getConfiguration().setStrings("hipi.reduction.transformMatrix.path", transformMatrixPath);
        job.getConfiguration().setStrings("hipi.reduction.mean.path", meanPath);

        return job.waitForCompletion(true) ? 0 : 1;
    }
}