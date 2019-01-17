using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning_ML.NetworkCore
{
    public class Node
    {
        public double sum { get; set; }
        public List<double> weight = new List<double>();
        public double errorValue { get; set; }
    }
}
