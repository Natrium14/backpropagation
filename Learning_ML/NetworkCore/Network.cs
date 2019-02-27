using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning_ML.NetworkCore
{
    public class Network: INetwork
    {
        public List<Layer> layers { get; set; }

        public Network()
        {

        }
        public static double LogisticFunction(double x)
        {
            return 1 / (1 + Math.Exp(-2 * 0.1 * x));
        }

        public void CreateSampleNetwork(int countInputNodes, int countLayers)
        {
            if (countInputNodes > 0 && countLayers > 2)
            {
                layers = new List<Layer>();
                for (int i = 0; i < countLayers; i++)
                {
                    Layer layer = new Layer();
                    if (i == 0)
                    {
                        for (int j = 0; j < countInputNodes; j++)
                        {
                            layer.nodes.Add(new Node());
                        }
                    }
                    else
                    {
                        layer.nodes.Add(new Node());
                    }

                    layers.Add(layer);
                }

                layers[countLayers - 1].nodes[0].errorValue = 1;
            }
        }

        public void AddHiddenLayer(int countNodes)
        {
            if (countNodes > 0)
            {
                Layer layerHidden = new Layer();
                for(int i = 0; i < countNodes; i++)
                {
                    layerHidden.nodes.Add(new Node());
                }

                layers.Insert(1, layerHidden);
            }
        }

        public void Train(double result, double trainSpeed)
        {
            bool firstTime = true;
            Random r = new Random();

            while (Math.Abs(layers[layers.Count - 1].nodes[0].errorValue) > 0.001)
            {
                for (int i = 0; i < layers.Count - 1; i++)
                {
                    for (int j = 0; j < layers[i].nodes.Count; j++)
                    {
                        for (int k = 0; k < layers[i + 1].nodes.Count; k++)
                        {
                            if (firstTime)
                            {
                                double w = r.NextDouble();
                                layers[i + 1].nodes[k].weight.Add(w);
                                layers[i + 1].nodes[k].sum += w * layers[i].nodes[j].sum;
                            }
                            else
                            {
                                double w = layers[i + 1].nodes[k].weight.ElementAt(j);
                                double newW = w + w * layers[i + 1].nodes[k].errorValue * trainSpeed;
                                layers[i + 1].nodes[k].weight[j] = newW;
                                layers[i + 1].nodes[k].sum += newW * layers[i].nodes[j].sum;
                            }
                        }
                    }

                    for (int k = 0; k < layers[i + 1].nodes.Count; k++)
                    {
                        layers[i + 1].nodes[k].sum = LogisticFunction(layers[i + 1].nodes[k].sum);
                    }
                }

                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    for (int j = layers[i].nodes.Count - 1; j >= 0; j--)
                    {
                        //double outResutlt = layers[i].nodes[j].sum;
                        //double q = (result - outResutlt) * outResutlt * (1 - outResutlt);
                        //layers[i].nodes[j].errorValue = q;

                        if (i == layers.Count - 1)
                        {
                            double outResutlt = layers[i].nodes[j].sum;
                            double q = (result - outResutlt) * outResutlt * (1 - outResutlt);
                            layers[i].nodes[j].errorValue = q;
                        }
                        else
                        {
                            double outResutlt = layers[i].nodes[j].sum;

                            double q = outResutlt * (1 - outResutlt);
                            double tmpW = 0;
                            foreach (var item in layers[i].nodes[j].weight)
                            {
                                tmpW += item * layers[i].nodes[j].errorValue;
                            }
                            q *= tmpW;
                            layers[i].nodes[j].errorValue = q;
                        }
                    }
                }

                firstTime = false;
            }
        }

        public double Test()
        {
            Network testNetwork = this;
            
            for (int i = 0; i < testNetwork.layers.Count - 1; i++)
            {
                for (int j = 0; j < testNetwork.layers[i].nodes.Count; j++)
                {
                    for (int k = 0; k < testNetwork.layers[i + 1].nodes.Count; k++)
                    {
                        double w = layers[i + 1].nodes[k].weight.ElementAt(j);
                        testNetwork.layers[i + 1].nodes[k].sum += w * testNetwork.layers[i].nodes[j].sum;
                    }
                }

                for (int k = 0; k < testNetwork.layers[i + 1].nodes.Count; k++)
                {
                    testNetwork.layers[i + 1].nodes[k].sum = LogisticFunction(testNetwork.layers[i + 1].nodes[k].sum);
                }
            }

            return testNetwork.layers[layers.Count - 1].nodes[0].sum;
        }
    }
}
