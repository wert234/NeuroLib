using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroLib.Models.TextModel
{
    public class ModelOutput
    {
        [ColumnName(@"Sentence")]
        public float[] Sentence { get; set; }

        [ColumnName(@"Label")]
        public uint Label { get; set; }

        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public float PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }
    }
}
