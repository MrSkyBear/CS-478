import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;

public class KNN extends SupervisedLearner
{
    private Matrix stored_data;
    private Matrix targets;
    private int k;
    private int num_output_classes;

    private HashMap<Integer,String> feature_types;
    private String output_type;

    public KNN(int k)
    {
        this.k = k;
        feature_types = new HashMap<>();
    }


    public double manhattan_distance(double[] instance_a, double[] instance_b)
    {
        double distance = 0.0;

        for (int i = 0; i < instance_a.length; i++)
        {
            if (feature_types.get(i).equals("CONTINUOUS"))
            {
                distance += Math.abs((instance_a[i] - instance_b[i]));
            }
        }

        return distance;
    }

    public double single_distance(double[] instance_a, double[] instance_b)
    {
        double distance_squared = 0.0;

        for (int i = 0; i < instance_a.length; i++)
        {
            if (feature_types.get(i).equals("CONTINUOUS"))
            {
                distance_squared += Math.pow((instance_a[i] - instance_b[i]), 2.0);
            }
            else
            {
                // If either a or b is unknown, distance is 1
                // Matrix class uses Double.MAX_VALUE for unknown/missing values
                if (instance_a[i] == Double.MAX_VALUE || instance_b[i] == Double.MAX_VALUE)
                {
                    distance_squared += 1;
                }
                else
                {
                    distance_squared += instance_a[i] == instance_b[i] ? 0 : 1;
                }
            }
        }

        return Math.sqrt(distance_squared);
    }

    // Return a mapping of distance : list of data_point row numbers
    public HashMap<Double, ArrayList<Integer>> calculate_distances(double[] data_instance)
    {
        HashMap<Double, ArrayList<Integer>> distances = new HashMap<>();

        for (int i = 0; i < stored_data.rows(); i++)
        {
            double distance = single_distance(stored_data.row(i), data_instance);

            // Used to test manhattan distance function for data sets with all continuous features
            //double distance = manhattan_distance(stored_data.row(i), data_instance);

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

        if (targets.valueCount(0) == 0)
        {
            this.output_type = "CONTINUOUS";
        }
        else
        {
            this.output_type = "NOMINAL";
        }
    }

    // Returns a mapping of the neighbor's row in the data set : distance
    public HashMap<Integer, Double> nearest_neighbors(HashMap<Double, ArrayList<Integer>> distances)
    {
        HashMap<Integer, Double> neighbors = new HashMap<>();

        Double[] distance_keys = new Double[distances.size()];
        distances.keySet().toArray(distance_keys);
        Arrays.sort(distance_keys);

        int distance_index = 0;

        while (neighbors.size() != this.k)
        {
            double distance = distance_keys[distance_index];
            ArrayList<Integer> instances = distances.get(distance);

            for (Integer i : instances)
            {
                neighbors.put(i, distance);

                if (neighbors.size() == this.k)
                {
                    break;
                }
            }

            distance_index++;
        }

        return neighbors;
    }

    // Returns a mapping of output_class : number of votes
    // Each of the nearest neighbors will get a vote for a single output class
    public HashMap<Double, Double> get_votes(HashMap<Integer, Double> neighbors, boolean weighted)
    {
        HashMap<Double, Double> votes = new HashMap<>();

        for (double x = 0; x < this.num_output_classes; x++)
        {
            votes.put(x, 0.0);
        }

        for (Map.Entry<Integer, Double> e : neighbors.entrySet())
        {
            double instance_class = this.targets.get(e.getKey(), 0);

            double vote = 1;

            // If distance is 0, an exact match was found, so it's class should always be chosen
            if (weighted && e.getValue() != 0)
            {
                vote = (1.0 / (Math.pow(e.getValue(), 2.0)));
            }
            else if (weighted && e.getValue() == 0)
            {
                vote = Double.MAX_VALUE;
                votes.put(instance_class, vote);
            }

            //if (vote != Double.MAX_VALUE)
            if (votes.get(instance_class) != Double.MAX_VALUE && vote != Double.MAX_VALUE)
            {
                votes.put(instance_class, votes.get(instance_class) + vote);
            }
        }

        return votes;
    }

    public double regression_vote(HashMap<Integer, Double> neighbors, boolean weighted)
    {
        double raw_score = 0;
        double total_weight = 0.0;

        for (Map.Entry<Integer, Double> e : neighbors.entrySet())
        {
            double weight = 1;

            if (weighted && e.getValue() != 0)
            {
                weight = (1 / (Math.pow(e.getValue(), 2.0)));
            }

            total_weight += weight;

            raw_score += (weight * this.targets.get(e.getKey(), 0));
        }

        if (weighted)
        {
            return raw_score / total_weight;
        }
        else
        {
            return (raw_score / this.k);
        }
    }


    public void predict(double[] feature_values, double[] targets) throws Exception
    {

        HashMap<Integer, Double> neighbors = nearest_neighbors(calculate_distances(feature_values));

        // If nominal output classes, do normal Classification, else use regression
        if (this.output_type.equals("NOMINAL"))
        {
            HashMap<Double, Double> votes = get_votes(neighbors, true);

            double majority_class = 0;
            double highest_votes = -1;

            // Find instance with largest number of votes
            for (Map.Entry<Double, Double> e : votes.entrySet())
            {
                if (e.getValue() > highest_votes)
                {
                    majority_class = e.getKey();
                    highest_votes = e.getValue();
                }
            }

            targets[0] = majority_class;
        }
        else
        {
            targets[0] = regression_vote(neighbors, true);
        }
    }
}