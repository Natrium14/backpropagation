using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning_ML
{
    class Program
    {
        static void Main(string[] args)
        {
            Random r = new Random();
            double x1 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            double x2 = Network.LogisticFunction(r.Next(1, 100) + r.NextDouble());
            double result = Network.LogisticFunction(x1 + x2);
            double n = 0.2;
            Network network = new Network(x1,x2);
            network.Train(result, n);

            double y1 = r.Next(1, 100) + r.NextDouble();
            double y2 = r.Next(1, 100) + r.NextDouble();
            y1 = Network.LogisticFunction(y1);
            y2 = Network.LogisticFunction(y2);
            double result2 = Network.LogisticFunction(y1 + y2);
            double networkTest = network.Test(y1, y2);
            Console.WriteLine("Работа сети: {0}; Реальный результат: {1}", networkTest, result2);
            Console.WriteLine("Точность: " + networkTest / result2);

            Console.Read();
        }
    }    

    class Network
    {
        public class Node
        {
            public double s { get; set; }
            public List<double> w = new List<double>();
            public double q { get; set; }
        }

        public class Layer
        {
            public List<Node> nodes = new List<Node>();
        }

        public List<Layer> layers { get; set; }

        public Network(double x1, double x2)
        {
            Layer layerInput = new Layer();
            Layer layerHidden1 = new Layer();
            Layer layerHidden2 = new Layer();
            Layer layerOutput = new Layer();

            layerInput.nodes.Add(new Node() { s = x1 });
            layerInput.nodes.Add(new Node() { s = x2 });
            layerHidden1.nodes.Add(new Node());
            layerHidden1.nodes.Add(new Node());
            layerHidden1.nodes.Add(new Node());
            layerHidden2.nodes.Add(new Node());
            layerHidden2.nodes.Add(new Node());
            layerOutput.nodes.Add(new Node());

            layers = new List<Layer>();
            layers.Add(layerInput);
            layers.Add(layerHidden1);
            layers.Add(layerHidden2);
            layers.Add(layerOutput);

            layerOutput.nodes[0].q = 1;
        }
        public static double LogisticFunction(double x)
        {
            return 1 / (1 + Math.Exp(-2 * 0.1 * x));
        }
        public static double ReverseLogisticFunction(double x)
        {
            return Math.Log(1 / x - 1) / -0.2;
        }

        public void Train(double result, double n)
        {
            bool firstTime = true;
            Random r = new Random();

            while (Math.Abs(this.layers[layers.Count-1].nodes[0].q) > 0.001)
            {
                for (int i = 0; i < this.layers.Count - 1; i++)
                {
                    for (int j = 0; j < this.layers[i].nodes.Count; j++)
                    {
                        for (int k = 0; k < this.layers[i + 1].nodes.Count; k++)
                        {
                            if (firstTime)
                            {
                                double w = r.NextDouble();
                                this.layers[i + 1].nodes[k].w.Add(w);
                                this.layers[i + 1].nodes[k].s += w * this.layers[i].nodes[j].s;
                            }
                            else
                            {
                                double w = this.layers[i + 1].nodes[k].w.ElementAt(j);
                                double newW = w + w * this.layers[i + 1].nodes[k].q * n;
                                this.layers[i + 1].nodes[k].w[j] = newW;
                                this.layers[i + 1].nodes[k].s += newW * this.layers[i].nodes[j].s;
                            }
                        }
                    }

                    for (int k = 0; k < this.layers[i + 1].nodes.Count; k++)
                    {
                        this.layers[i + 1].nodes[k].s = LogisticFunction(this.layers[i + 1].nodes[k].s);
                    }
                }

                for (int i = this.layers.Count - 1; i >= 0; i--)
                {
                    for (int j = this.layers[i].nodes.Count - 1; j >= 0; j--)
                    {
                        double outResutlt = this.layers[i].nodes[j].s;
                        double q = (result - outResutlt) * outResutlt * (1 - outResutlt);
                        this.layers[i].nodes[j].q = q;
                    }
                }

                firstTime = false;
            }
        }

        public double Test(double x1, double x2)
        {
            double y1 = LogisticFunction(x1);
            double y2 = LogisticFunction(x2);
            Network myNetwork = new Network(y1, y2);

            for (int i = 0; i < myNetwork.layers.Count - 1; i++)
            {
                for (int j = 0; j < myNetwork.layers[i].nodes.Count; j++)
                {
                    for (int k = 0; k < myNetwork.layers[i + 1].nodes.Count; k++)
                    {
                        double w = this.layers[i + 1].nodes[k].w.ElementAt(j);
                        myNetwork.layers[i + 1].nodes[k].s += w * myNetwork.layers[i].nodes[j].s;
                    }
                }

                for (int k = 0; k < myNetwork.layers[i + 1].nodes.Count; k++)
                {
                    myNetwork.layers[i + 1].nodes[k].s = LogisticFunction(myNetwork.layers[i + 1].nodes[k].s);
                }
            }

            return myNetwork.layers[layers.Count-1].nodes[0].s;
        }
    }
}
