using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Learning_ML.NetworkCore;

namespace Learning_ML
{
    class Program
    {
        static void Main(string[] args)
        {              
            Random r = new Random();
            Network network = new Network();
            network.CreateSampleNetwork(3, 3);
            network.AddHiddenLayer(2);
            double n = 0.01;
            double x1, x2, x3, result;
            x1 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            x2 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            x3 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            result = Network.LogisticFunction(x1 + x2 + x3 * 2);
            network.layers[0].nodes[0].sum = x1;
            network.layers[0].nodes[1].sum = x2;
            network.layers[0].nodes[2].sum = x3;
            network.Train(result, n);

            double y1 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            double y2 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            double y3 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            double result2 = Network.LogisticFunction(y1 + y2 + y3 * 2);
            network.layers[0].nodes[0].sum = y1;
            network.layers[0].nodes[1].sum = y2;
            network.layers[0].nodes[2].sum = y3;
            double networkTest = network.Test();
            Console.WriteLine("Работа сети: {0}; Реальный результат: {1}", networkTest, result2);
            Console.WriteLine("Точность: " + networkTest / result2);

            Console.Read();
        }
    }
}
