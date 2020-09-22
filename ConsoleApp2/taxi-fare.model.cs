using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp2.Model
{
    #region model input class
    public class ModelInput
    {
        [ColumnName("vendor_id"), LoadColumn(0)]
        public string Vendor_id { get; set; }


        [ColumnName("rate_code"), LoadColumn(1)]
        public float Rate_code { get; set; }


        [ColumnName("passenger_count"), LoadColumn(2)]
        public float Passenger_count { get; set; }


        [ColumnName("trip_time_in_secs"), LoadColumn(3)]
        public float Trip_time_in_secs { get; set; }


        [ColumnName("trip_distance"), LoadColumn(4)]
        public float Trip_distance { get; set; }


        [ColumnName("payment_type"), LoadColumn(5)]
        public string Payment_type { get; set; }


        [ColumnName("fare_amount"), LoadColumn(6)]
        public float Fare_amount { get; set; }


    }
    #endregion

    #region model output class
    public class ModelOutput
    {
        public float Score { get; set; }
    }
    #endregion

    public class ConsumeModel
    {
        public static string MLNetModelPath = Path.GetFullPath("taxi-fare.zip");

        public static ModelOutput Predict(ModelInput input)
        {
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            ModelOutput result = predEngine.Predict(input);
            return result;
        }

        public static ModelInput GetSampleData()
        {
            ModelInput sampleData = new ModelInput()
            {
                Vendor_id = @"CMT",
                Rate_code = 1F,
                Passenger_count = 1F,
                Trip_time_in_secs = 1271F,
                Trip_distance = 3.8F,
                Payment_type = @"CRD",
            };

            return sampleData;
        }
    }
}
