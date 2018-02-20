package org.hipi.pca;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import org.hipi.util.helper;

public class Covariance extends Configured implements Tool {

    public static final int patchSize = 32;  // Patch dimensions: patchSize x patchSize
    public static final float sigma = 10;  // Standard deviation of Gaussian weighting function

    public int run(String[] args) throws Exception {

        // Used for initial argument validation and hdfs configuration before jobs are run
        Configuration conf = Job.getInstance().getConfiguration();

        // Validate arguments before any work is done
        helper.validateArgs(args, 2);

        // Build I/O path strings
        String inputHibPath = args[0];
        String outputBaseDir = args[1];
        String outputMeanDir = outputBaseDir + "/mean-output/";
        String outputCovarianceDir = outputBaseDir + "/covariance-output/";
        String inputMeanPath = outputMeanDir + "part-r-00000"; //used to access ComputeMean result

        // Set up directory structure
        helper.mkdir(outputBaseDir, conf);
        helper.rmdir(outputMeanDir, conf);
        helper.rmdir(outputCovarianceDir, conf);

        // Run compute mean
        if (ComputeMean.run(args, inputHibPath, outputMeanDir) == 1) {
            System.out.println("Compute mean job failed to complete.");
            return 1;
        }

        helper.validatePath(inputMeanPath, conf);

        // Run compute covariance
        if (ComputeCovariance.run(args, inputHibPath, outputCovarianceDir, inputMeanPath) == 1) {
            System.out.println("Compute covariance job failed to complete.");
            return 1;
        }

        // Indicate success
        return 0;
    }


    // Main driver for full covariance computation
    public static void main(String[] args) throws Exception {
        /*
            args: inputHibPath outputBaseDir
         */
        int res = ToolRunner.run(new Covariance(), args);
        System.exit(res);
    }
}
