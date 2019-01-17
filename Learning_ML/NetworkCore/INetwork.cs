using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning_ML.NetworkCore
{
    interface INetwork
    {
        void Train(double result, double trainSpeed);
        double Test();
        void AddHiddenLayer(int countNodes);
        void CreateSampleNetwork(int countInputNodes, int countLayers);
    }
}
