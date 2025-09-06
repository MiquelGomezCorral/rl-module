# lr-module
Reinforcement learning module for projects

This module is completely based on the Hugging Face reinforcement learning course [here](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction) and three videos made by wights and biases: [1](https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases), [2](https://www.youtube.com/watch?v=05RMTj-2K_Y&t=2s&ab_channel=Weights%26Biases), [3](https://www.youtube.com/watch?v=BvZvx7ENZBw&ab_channel=Weights%26Biases).
Feel free to use it and learn as much as I did (more or less hehe). 

Is based and built around gymnassium environments. My recommendation is to stick to them. But you can always adapt anything to your own use case :)


# Setup env
```bash
python3.12 -m venv venv
source venv/bin/activate

pip install uv
uv pip install -r requirements.txt
pip install -e ./app
```

# Choosing envs and -cnn parameter
When choosing the env take into account that if the output state is an image it will need the -cnn parameter to use the correct model.