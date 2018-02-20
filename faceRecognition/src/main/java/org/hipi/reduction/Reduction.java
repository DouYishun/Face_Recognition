package org.hipi.reduction;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.hipi.util.helper;


public class Reduction extends Configured implements Tool {

    public static final int patchSize = 32;  // Patch dimensions: patchSize x patchSize

    public int run(String[] args) throws Exception {

        // Used for initial argument validation and hdfs configuration before jobs are run
        Configuration conf = Job.getInstance().getConfiguration();

        // Validate arguments before any work is done
        helper.validateArgs(args, 3);

        // Build I/O path strings
        String inputHibPath = args[0];
        String coefficientBaseDir = args[1];
        String transformMatrixPath = coefficientBaseDir + "/transform_matrix-output/part-r-00000";
        String meanPath = coefficientBaseDir + "/mean-output/part-r-00000";
        String outputDir = args[2];

        // Set up directory structure
        helper.rmdir(outputDir, conf);
        helper.mkdir(outputDir, conf);

        // Run reduction
        if (runReduction.run(inputHibPath, transformMatrixPath, meanPath, outputDir) == 1) {
            System.out.println("Reduction job failed to complete.");
            return 1;
        }

        // Indicate success
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
