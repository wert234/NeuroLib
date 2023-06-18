using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Trainers;
using System.IO;

namespace NeuroLib.Models.TextModel
{
    public class TextModel
    {
        private MLContext _MlContext;

        public TextModel() => _MlContext = new MLContext();

        public ITransformer LoadModel(string MLNetModelPath) =>
            _MlContext.Model.Load(MLNetModelPath, out var _);
        public void CreatModel(string data, string SaveModelPath) =>
            new MLContext().Model.Save(TrainModel(_MlContext.Data.LoadFromTextFile<ModelInput>(data, separatorChar: ';')),  _MlContext.Data.LoadFromTextFile<ModelInput>(data, separatorChar: ';').Schema, SaveModelPath);
        public ITransformer TrainModel(IDataView trainData) =>
            Pipeline(_MlContext).Fit(trainData);
        public ITransformer RetrainModel(MLContext context, IDataView trainData) =>
            Pipeline(context).Fit(trainData);
        public float Predict(ITransformer transformer, ModelInput input) =>
            _MlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(transformer).Predict(input).PredictedLabel;
        

        private EstimatorChain<KeyToValueMappingTransformer> Pipeline(MLContext context)
            =>  context.Transforms.Text.FeaturizeText(inputColumnName: @"Sentence", outputColumnName: @"Sentence")
                        .Append(context.Transforms.Concatenate(@"Features", new[] { @"Sentence" }))
                        .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label"))
                        .Append(context.Transforms.NormalizeMinMax(@"Features", @"Features"))
                        .Append(context.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator: context.BinaryClassification.Trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options() { L1Regularization = 0.03379069F, L2Regularization = 0.7905437F, LabelColumnName = @"Label", FeatureColumnName = @"Features" }), labelColumnName: @"Label"))
                        .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));  
    }
}
