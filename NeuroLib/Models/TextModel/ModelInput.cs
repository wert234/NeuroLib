using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroLib.Models.TextModel
{
    public class ModelInput
    {
        [ColumnName(@"Sentence")]
        [LoadColumn(0)]
        public string Sentence { get; set; }

        [ColumnName(@"Label")]
        [LoadColumn(1)]
        public float Label { get; set; }
    }
}
