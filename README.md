# lr-module
Reinforcement learning module for projects


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