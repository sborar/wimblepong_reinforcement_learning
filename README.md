# Wimblepong reinforcement learning
A3C approach<br/>
To train, run train.py<br/>
To load existing models while training, use --load_model_path to pass model path argument to train.py<br/>
Currently trained model provide 90+ win rate against simple AI<br/>

To test, run test_pong_ai.py<br/>
To load existing models for testing, use --load_model_path to pass model path argument to test_pong_ai.py<br/>

Some strategies tried - <br/>
1. Batch normalization for more stable training<br/>
2. Entropy maximization for encouraging exploration<br/>
3. Switching opponents while training to prevent overfitting
