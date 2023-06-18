using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using Microsoft.ML;
using NeuroLib.Models;
using NeuroLib.Models.TextModel;

namespace TestNeuroLib_2
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            TextModel model = new TextModel();
           
            var transformer = model.LoadModel(Path.GetFullPath("NeuroLibModel.zip"));

           var predict = model.Predict(transformer, new ModelInput() { Sentence = Input.Text });

            Output.Text = predict.ToString(); 
        }
    }
}
