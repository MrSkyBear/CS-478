import java.util.ArrayList;
import java.util.Random;

public class Perceptron extends SupervisedLearner
{
    ArrayList<double[]> machines;

    ArrayList<double[]> best_machines;

    double LEARNING_RATE = .1;

    public ArrayList<double[]> copy_array(ArrayList<double[]> a)
    {
        ArrayList<double[]> b = new ArrayList<double[]>();

        for (int i = 0; i < a.size(); i++)
        {
            b.add(new double[a.get(i).length]);

            for (int x = 0; x < a.get(i).length; x++)
            {
                b.get(i)[x] = a.get(i)[x];
            }
        }

        return b;
    }

    public void initialize_weights(Matrix features, int num_classes)
    {
        machines = new ArrayList<double[]>();

        Random r = new Random();

        for (int x = 0; x < num_classes; x++)
        {
            double[] weights = new double[features.cols() + 1];

            for (int i = 0; i < features.cols() + 1; i++)
            {
                weights[i] = r.nextDouble();
            }

            machines.add(weights);
        }

        best_machines = copy_array(machines);
    }

    public void update_weights(double[] weights, double[] inputs, double target, double output)
    {
        double learning_coefficient = LEARNING_RATE * (target - output);

        for (int i = 0; i < inputs.length; i++)
        {
            double weight_delta = learning_coefficient * inputs[i];
            weights[i] += weight_delta;
        }

        // Update Bias Weight
        weights[inputs.length] = learning_coefficient;
    }

    public double run_input(double[] inputs, double[] weights)
    {
        double output = 0;

        for (int i = 0; i < inputs.length; i++)
        {
            output += inputs[i] * weights[i];
        }

        // Add bias
        output += weights[weights.length - 1];

        return output;
    }

    public int classify(ArrayList<Double> outputs)
    {
        // If there is only one machine, simply run the threshold function
        // Otherwise, determine which machine had the largest net value
        if (outputs.size() == 1)
        {
            return (outputs.get(0) > 0) ? 1 : 0;
        }

        double max = outputs.get(0);
        int classification = 0;

        for (int i = 1; i < outputs.size(); i++)
        {
            if (outputs.get(i) > max)
            {
                max = outputs.get(i);
                classification = i;
            }
        }

        return classification;
    }

    public void train(Matrix features, Matrix labels) throws Exception
    {
        // pass in data and number of classes (machines)
        int num_classes = labels.m_enum_to_str.get(0).size();

        initialize_weights(features, num_classes);

        int non_improving_epochs = 0;
        double best_accuracy = 0;

        do
        {
            Random r = new Random();
            features.shuffle(r, labels);

            for (int i = 0; i < features.rows(); i++)
            {
                double target = labels.get(i, 0);
                double[] input_list = features.m_data.get(i);

                for (int x = 0; x < machines.size(); x++)
                {
                    double[] weights = machines.get(x);
                    double net_output = run_input(input_list, weights);

                    // Machine X detects class X
                    int expected = (x == target) ? 1 : 0;
                    int prediction = (net_output > 0) ? 1 : 0;

                    if (prediction != expected)
                    {
                        update_weights(weights, input_list, expected, prediction);
                    }
                }
            }

            double current_accuracy = measureAccuracy(features, labels, null);

            if (current_accuracy > best_accuracy)
            {
                best_accuracy = current_accuracy;
                non_improving_epochs = 0;

                // Store weights with best accuracy for use in prediction
                best_machines = copy_array(machines);
            }
            else
            {
                non_improving_epochs++;
            }
        }
        while(non_improving_epochs < 5);

        for (int i = 0; i < best_machines.size(); i++)
        {
            System.out.println("Machine: " + labels.m_enum_to_str.get(0).get(i));
            for (int j = 0; j < best_machines.get(i).length; j++)
            {
                System.out.println(best_machines.get(i)[j]);
            }

            System.out.println("");
        }
    }

    public void predict(double[] features, double[] labels) throws Exception
    {
        ArrayList<Double> net_values = new ArrayList<Double>();
        for (int x = 0; x < best_machines.size(); x++)
        {
            double net_value = run_input(features, best_machines.get(x));
            net_values.add(net_value);
        }

        labels[0] = classify(net_values);
    }
}