package org.hipi.pca;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import org.hipi.util.util;

public class Covariance extends Configured implements Tool {

    public static final float sigma = 10;  // Standard deviation of Gaussian weighting function

    public int run(String[] args) throws Exception {
        Configuration conf = Job.getInstance().getConfiguration();

        util.validateArgs(args, 2);

        // Build I/O path strings
        String inputHibPath = args[0];
        String outputBaseDir = args[1];
        String outputMeanDir = outputBaseDir + "/mean-output/";
        String outputCovarianceDir = outputBaseDir + "/covariance-output/";
        String inputMeanPath = outputMeanDir + "part-r-00000"; //used to access ComputeMean result

        // Set up directory structure
        util.rmdir(outputBaseDir, conf);
        util.mkdir(outputBaseDir, conf);

        // Run compute mean
        if (ComputeMean.run(inputHibPath, outputMeanDir) == 1) {
            System.out.println("Compute mean job failed to complete.");
            return 1;
        }

        util.validatePath(inputMeanPath, conf);

        // Run compute covariance
        if (ComputeCovariance.run(inputHibPath, outputCovarianceDir, inputMeanPath) == 1) {
            System.out.println("Compute covariance job failed to complete.");
            return 1;
        }

        return 0;
    }


    public static void main(String[] args) throws Exception {
        /*
            args: inputHibPath outputBaseDir
         */
        int res = ToolRunner.run(new Covariance(), args);
        System.exit(res);
    }
}
