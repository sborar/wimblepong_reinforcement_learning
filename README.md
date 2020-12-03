# Wimblepong reinforcement learning
A3C approach
To train run train.py 
To load existing models while training, use --load_model_path to pass model path argument to train.py
Currently trained model provide 90+ win rate against simple AI.

To test, run test_pong_ai.py
To load existing models for testing, use --load_model_path to pass model path argument to test_pong_ai.py

Some strategies tried - 
1. Batch normalization for more stable training
2. Entropy maximization for encouraging exploration
3. Switching opponents while training to prevent overfitting
