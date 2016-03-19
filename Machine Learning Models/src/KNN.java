import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class KNN extends SupervisedLearner
{
    private Matrix stored_data;
    private Matrix targets;
    private int k;
    private int num_output_classes;

    HashMap<Integer,String> feature_types;

    public KNN(int k)
    {
        this.k = k;
        feature_types = new HashMap<Integer,String>();
    }

    public double single_distance(double[] instance_a, double[] instance_b)
    {
        double distance = 0.0;

        for (int i = 0; i < instance_a.length; i++)
        {
            if (feature_types.get(i).equals("CONTINUOUS"))
            {

            }
            else
            {

            }
        }

        return 0.0;
    }

    // Function will take in new data instance
    // For each instance in the stored data
    //    if feature is continuous, use euclidean distance
    //      (Matrix.valueCount(column_number): 0 == continuous, else nominal?
    //      Nominal values in matrix are represented with enum
    //    else if feature matches, distance is 0, else 1

    // Return a mapping of distance : list of data_point row numbers
    public HashMap<Double, ArrayList<Integer>> calculate_distances(double[] data_instance)
    {
        HashMap<Double, ArrayList<Integer>> distances = new HashMap<Double, ArrayList<Integer>>();

        for (int i = 0; i < stored_data.rows(); i++)
        {
            double distance = single_distance(stored_data.row(i), data_instance);

            if (!distances.containsKey(distance))
            {
                distances.put(distance, new ArrayList<Integer>());
            }

            distances.get(distance).add(i);
        }

        return distances;
    }

    // Store data along with list of feature types for each value
    // i.e.
    // [4.0, Red, 6.3]
    // [CONTINUOUS, NOMINAL, CONTINUOUS]
    public void train(Matrix feature_values, Matrix targets) throws Exception
    {
        this.stored_data = feature_values;
        this.targets = targets;
        this.num_output_classes = targets.m_enum_to_str.get(0).size();

        for (int i = 0; i < stored_data.cols(); i++)
        {
            int type = stored_data.valueCount(i);

            if (type == 0)
            {
                feature_types.put(i, "CONTINUOUS");
            }
            else
            {
                feature_types.put(i, "NOMINAL");
            }
        }
    }

    // Returns a mapping of output_class : number of votes
    public HashMap<Double, Integer> get_votes(HashMap<Double, ArrayList<Integer>> distances)
    {
        HashMap<Double, Integer> votes = new HashMap<>();

        // DOUBLE CHECK THAT OUTPUT CLASSES START AT 0
        for (double x = 0; x < this.num_output_classes; x++)
        {
            votes.put(x, 0);
        }

        int tallied_instances = 0;
        boolean found_n = false;

        // Need to sort keyset in descending order
        for (double d : distances.keySet())
        {
            ArrayList<Integer> instances = distances.get(d);

            for (Integer i : instances)
            {
                double instance_class = this.targets.get(0, i);
                votes.put(instance_class, votes.get(instance_class) + 1);
                tallied_instances++;

                if (tallied_instances == this.k)
                {
                    found_n = true;
                    break;
                }
            }

            if (found_n)
            {
                break;
            }
        }

        return votes;
    }



    public void predict(double[] feature_values, double[] targets) throws Exception
    {
        HashMap<Double, Integer> votes = get_votes(calculate_distances(feature_values));

        double majority_class = 0;
        double highest_votes = -1;

        // Find instance with largest number of votes
        for (Map.Entry<Double, Integer> e : votes.entrySet())
        {
            if (e.getValue() > highest_votes)
            {
                majority_class = e.getKey();
                highest_votes = e.getValue();
            }
        }

        targets[0] = majority_class;
    }
}