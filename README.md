# E8125 Course Project: Wimblepong reinforcement learning
Aalto University, Reinforcement Learning<br/>
OpenAI GYM Pong-v0 Atari game<br/>
A2C approach

## Developers
Sheetal Borar<br/>
Hossein Firooz


## Train
Run train.py<br/>
To load existing models while training, use --load_model_path to pass model path argument to train.py<br/>
Currently trained model provide 90+ win rate against simple AI<br/>

## Test
Run test_pong_ai.py<br/>
To load existing models for testing, use --load_model_path to pass model path argument to test_pong_ai.py<br/>

## Strategies tried
1. Batch normalization for more stable training<br/>
2. Entropy maximization for encouraging exploration<br/>
3. Switching opponents while training to prevent overfitting<br/>

Find the details in the report.
