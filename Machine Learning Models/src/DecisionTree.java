import java.util.ArrayList;
import java.util.Random;

public class DecisionTreeModel extends SupervisedLearner
{   

    private ID3Node tree_root;



    public DecisionTreeModel(Matrix data, Matrix features)
    {
        tree_root = create_tree(data, features);
    }

    public ID3Node create_tree(Matrix data, Matrix features)
    {
        ID3Node root = new ID3Node();

        // If all data instances are pure, set label to that class and return root

        // If number of predicting attributes is 0, return root with label of most common target attribute

        int attribute = 0; // Choose feature that represents majority (Best info gain)
    }


    public int classify(ArrayList<Double> outputs)
    {
        
    }

    public void train(Matrix features, Matrix labels) throws Exception
    {
        
    }

    public void predict(double[] features, double[] labels) throws Exception
    {
        
    }
}