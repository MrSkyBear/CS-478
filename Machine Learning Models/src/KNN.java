import java.util.ArrayList;
import java.util.Random;

public class KNN extends SupervisedLearner
{
    private Matrix stored_data;
    private int k;

    public KNN(int k)
    {
        this.k = k;
    }

    // Should probably return a map of distance:list of points
    public double calculate_distances()
    {
        // Function will take in new data instance
        // For each instance in the stored data
        //    if feature is continuous, use euclidean distance
        //    else if feature matches, distance is 0, else 1

        return 0.0;
    }

    public int classify(ArrayList<Double> outputs)
    {
        // Should I store new instances in the "training data?"
        return 0;
    }


    // Combine matrices so target is last element of feature row? Or Store separately
    public void train(Matrix feature_values, Matrix targets) throws Exception
    {
        // Store data along with list of feature types for each value
        // i.e.
        // [4.0, Red, 6.3]
        // [C, N, C]
        this.stored_data = feature_values;
    }

    public void predict(double[] feature_values, double[] targets) throws Exception
    {
        
    }
}