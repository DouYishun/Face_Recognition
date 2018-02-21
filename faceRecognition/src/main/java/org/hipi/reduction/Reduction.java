package org.hipi.reduction;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.hipi.util.util;


public class Reduction extends Configured implements Tool {

    public int run(String[] args) throws Exception {
        Configuration conf = Job.getInstance().getConfiguration();

        util.validateArgs(args, 3);

        // Build I/O path strings
        String inputHibPath = args[0];
        String coefficientBaseDir = args[1];
        String transformMatrixPath = coefficientBaseDir + "/eigen-output/transformMatrix";
        String meanPath = coefficientBaseDir + "/mean-output/part-r-00000";
        String outputDir = args[2];

        // Set up directory structure
        util.rmdir(outputDir, conf);

        // Run reduction
        if (runReduction.run(inputHibPath, transformMatrixPath, meanPath, outputDir) == 1) {
            System.out.println("Reduction job failed to complete.");
            return 1;
        }

        return 0;
    }

    public static void main(String[] args) throws Exception {
        /*
            args: inputHibPath coefficientBaseDir outputDir
         */
        int res = ToolRunner.run(new Reduction(), args);
        System.exit(res);
    }
}
