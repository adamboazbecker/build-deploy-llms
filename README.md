# Build and Deploy LLM Workshop


### Step 1: Install the requirements
Type `pip install -r requirements.txt`.

Wait for the modules to be installed.

### Step 2: Get your OpenAI Key
You can find them in the link: https://platform.openai.com/account/api-keys

### Step 3: Create your `.env` file
Do you see the file titled `.env.example?`, you should duplicate it and name
the copied file `.env` (this is a file you should never commit to your source control).

Inside your `.env` file, fill in the credentials.

```
# .env
OPENAI_API_KEY=sk-....
```

### Step 4: Get your WandB Key
Sign up for an account to find your key: https://wandb.ai/.

### Step 5: Update the logging config
Inside the file `src/config.py`, you'll see a `vector_store_artifact` key.
Replace the username placeholder with your WandB username.
