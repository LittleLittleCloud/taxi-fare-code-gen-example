using ConsoleApp2.Model;
using ConsoleApp2.Pipeline;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace ConsoleApp2
{
    class Program
    {
        static void Main(string[] args)
        {
            var input = ConsumeModel.GetSampleData();
            var output = ConsumeModel.Predict(input);
            Console.WriteLine($"fare_amount: {output.Score}");

            RetrainModel();
        }


        static void RetrainModel()
        {
            var context = new MLContext();
            var dataset = context.Data.LoadFromTextFile<ModelInput>(@"C:\Users\xiaoyuz\Desktop\taxi-fare-train.csv", hasHeader: true, separatorChar: ',');
            var trainTestSplit = context.Data.TrainTestSplit(dataset);

            var model = ModelBuilder.RetrainPipeline(context, trainTestSplit.TrainSet);
            var test = model.Transform(trainTestSplit.TestSet);
            var eval = context.Regression.Evaluate(test, "fare_amount");
            PrintRegressionMetrics(eval);
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }
    }
}
